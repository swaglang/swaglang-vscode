import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "swaglang"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import enum
import logging
import re
from typing import Dict, List, Optional

import attrs
from antlr4 import InputStream, CommonTokenStream
from compiler.lexer.SwagLangLexer import SwagLangLexer
from compiler.lexer.SwagLangParser import SwagLangParser
from compiler.ast.builder import ASTBuilder
from compiler.errors.listener import SwagErrorListener
from compiler.semantic.analyzer import SemanticAnalyzer
from compiler.semantic.symbols import Symbol, SymbolKind
from lsprotocol import types
from pygls.cli import start_server
from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument

ADDITION = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)?\s*$")

TOKEN_TYPES = [
    "keyword",
    "type",
    "variable",
    "function",
    "operator",
    "string",
    "number",
    "comment",
    "parameter",
]
TOKEN_TYPE_INDEX = {t: i for i, t in enumerate(TOKEN_TYPES)}

ANTLR_TO_LSP: Dict[str, str] = {
    # keywords
    "IF": "keyword",
    "ELSE": "keyword",
    "ELSE_IF": "keyword",
    "WHILE": "keyword",
    "FOR": "keyword",
    "RETURN": "keyword",
    "LET": "keyword",
    "ACCESS_MOD": "keyword",
    "CONST": "keyword",
    "INTERFACE": "keyword",
    "EXTENDS": "keyword",
    "DO": "keyword",
    "BREAK": "keyword",
    "CONTINUE": "keyword",
    "DEFER": "keyword",
    "IN": "keyword",
    "VOID": "keyword",
    "BOOL": "keyword",
    "NULL": "keyword",
    # types
    "TYPE": "type",
    "MAP": "type",
    "SET": "type",
    # literals
    "STRING": "string",
    "INT": "number",
    "FLOAT": "number",
    # identifiers (fallback — overridden per-token by symbol table)
    "IDENT": "variable",
    "function": "function",
    "parameter": "parameter",
    "variable": "variable",
    "type": "type",
    # operators
    "ASSIGN": "operator",
    "ADD_ASSIGN": "operator",
    "SUB_ASSIGN": "operator",
    "MUL_ASSIGN": "operator",
    "DIV_ASSIGN": "operator",
    "MOD_ASSIGN": "operator",
    "EQ": "operator",
    "NEQ": "operator",
    "LT": "operator",
    "GT": "operator",
    "LTE": "operator",
    "GTE": "operator",
    "PLUS": "operator",
    "MINUS": "operator",
    "MUL": "operator",
    "DIV": "operator",
    "MOD": "operator",
    "EXP": "operator",
    "AND": "operator",
    "OR": "operator",
    "NOT": "operator",
    "INC": "operator",
    "DEC": "operator",
    # comments
    "COMMENT": "comment",
    "INLINE_COMMENT": "comment",
}


class TokenModifier(enum.IntFlag):
    deprecated = enum.auto()
    readonly = enum.auto()
    defaultLibrary = enum.auto()
    definition = enum.auto()


@attrs.define
class Token:
    line: int
    offset: int
    text: str
    tok_type: str = ""
    tok_modifiers: List[TokenModifier] = attrs.field(factory=list)


_SYMBOL_KIND_TO_LSP: Dict[SymbolKind, str] = {
    SymbolKind.FUNCTION: "function",
    SymbolKind.PARAMETER: "parameter",
    SymbolKind.VARIABLE: "variable",
    SymbolKind.INTERFACE: "type",
}


def _build_name_kind_map(snapshot: Dict[str, Symbol]) -> Dict[str, str]:
    """Build a flat name - LSP token type map from the symbol snapshot."""
    return {
        name: _SYMBOL_KIND_TO_LSP.get(sym.kind, "variable")
        for name, sym in snapshot.items()
    }


def _analyze_capturing(analyzer: SemanticAnalyzer, ast) -> tuple:
    """Run semantic analysis while capturing every symbol at define-time,
    before local scopes are popped."""
    snapshot: Dict[str, Symbol] = {}
    original_define = analyzer.symbols.define

    def _capturing_define(symbol: Symbol) -> bool:
        # Functions take priority over variables of the same name
        if symbol.name not in snapshot or symbol.kind == SymbolKind.FUNCTION:
            snapshot[symbol.name] = symbol
        return original_define(symbol)

    analyzer.symbols.define = _capturing_define
    symbol_table, type_table, errors = analyzer.analyze(ast)
    return symbol_table, type_table, errors, snapshot


def _format_document(
    document: TextDocument, range_: Optional[types.Range] = None
) -> List[types.TextEdit]:
    def _skip_line(line: int) -> bool:
        if range_ is None:
            return False
        return line < range_.start.line or line > range_.end.line

    edits: List[types.TextEdit] = []
    depth = 0
    for linum, line in enumerate(document.lines):
        if _skip_line(linum):
            continue
        line = line.lstrip()
        if "}" in line:
            depth -= 1
        edited_line = "\t" * depth + line
        edits.append(
            types.TextEdit(
                range=types.Range(
                    start=types.Position(line=linum, character=0),
                    end=types.Position(line=linum + 1, character=0),
                ),
                new_text=edited_line,
            )
        )
        if "{" in line:
            depth += 1
    return edits


class SwaglangServer(LanguageServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics: Dict[str, tuple] = {}
        self.tree = None
        self.parser_errors: list = []
        self._sem_errors: list = []
        self.tokens: Dict[str, List[Token]] = {}
        self.all_symbols: Dict[str, Dict[str, Symbol]] = {}

    def get_parser_results(self, document: TextDocument):
        stream = InputStream(document.source)

        lexer = SwagLangLexer(stream)
        error_listener = SwagErrorListener(document.filename or document.uri)
        lexer.removeErrorListeners()
        lexer.addErrorListener(error_listener)

        token_stream = CommonTokenStream(lexer)
        parser = SwagLangParser(token_stream)
        parser.removeErrorListeners()
        parser.addErrorListener(error_listener)

        tree = parser.prog()

        if error_listener.errors:
            return None, token_stream, parser, error_listener.errors
        return tree, token_stream, parser, []

    def _collect_tokens(
        self,
        token_stream: CommonTokenStream,
        parser,
        name_kind: Optional[Dict[str, str]] = None,
    ) -> List[Token]:
        tokens = []
        for t in token_stream.tokens:
            symbolic = parser.symbolicNames[t.type]
            if symbolic == "IDENT" and name_kind:
                symbolic = name_kind.get(t.text, "IDENT")
            tokens.append(
                Token(
                    line=t.line,
                    offset=t.column,
                    text=t.text,
                    tok_type=symbolic,
                )
            )
        return tokens

    def create_diagnostics(self, document: TextDocument) -> List[types.Diagnostic]:
        diagnostics = []
        parse_errors = self.parser_errors or []
        sem_errors: list = getattr(self, "_sem_errors", [])

        for error in parse_errors + sem_errors:
            line = error.line - 1  # ANTLR is 1-based, LSP is 0-based
            if line < 0 or line >= len(document.lines):
                continue
            diagnostics.append(
                types.Diagnostic(
                    message=error.message,  # type: ignore
                    severity=types.DiagnosticSeverity.Error,
                    range=types.Range(
                        start=types.Position(line=line, character=0),
                        end=types.Position(
                            line=line, character=len(document.lines[line]) - 1
                        ),
                    ),
                )
            )
        return diagnostics

    def parse(self, document: TextDocument) -> None:
        self.tree, token_stream, parser, self.parser_errors = self.get_parser_results(
            document
        )

        name_kind: Optional[Dict[str, str]] = None
        sem_errors: list = []
        if self.tree is not None:
            ast = ASTBuilder().visit(self.tree)
            if ast is not None:
                filename = document.filename or document.uri
                analyzer = SemanticAnalyzer(filename)
                symbol_table, _, sem_errors, snapshot = _analyze_capturing(
                    analyzer, ast
                )
                name_kind = _build_name_kind_map(snapshot)
                self.all_symbols[document.uri] = snapshot

        self._sem_errors = sem_errors
        self.tokens[document.uri] = self._collect_tokens(
            token_stream, parser, name_kind
        )
        self.diagnostics[document.uri] = (
            document.version,
            self.create_diagnostics(document),
        )

    def report_server_error(self, error: Exception, source) -> None:
        self.window_show_message(
            types.ShowMessageParams(
                message=f"Error in server: {error}",
                type=types.MessageType.Error,
            )
        )
        data = getattr(error, "data", {}) or {}
        tb = "".join(data.get("traceback", []))
        self.window_log_message(
            types.LogMessageParams(
                message=f"\nError in server: {error} \n {tb}",
                type=types.MessageType.Error,
            )
        )


server = SwaglangServer("swaglang-server", "v1")


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: SwaglangServer, params: types.DidOpenTextDocumentParams) -> None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse(doc)
    for uri, (version, diagnostics) in ls.diagnostics.items():
        ls.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=uri, version=version, diagnostics=diagnostics
            )
        )


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: SwaglangServer, params: types.DidChangeTextDocumentParams) -> None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse(doc)
    for uri, (version, diagnostics) in ls.diagnostics.items():
        ls.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=uri, version=version, diagnostics=diagnostics
            )
        )


@server.feature(types.TEXT_DOCUMENT_FORMATTING)
def format_document(
    ls: SwaglangServer, params: types.DocumentFormattingParams
) -> List[types.TextEdit]:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return _format_document(doc)


@server.feature(types.TEXT_DOCUMENT_RANGE_FORMATTING)
def format_range(
    ls: SwaglangServer, params: types.DocumentRangeFormattingParams
) -> List[types.TextEdit]:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return _format_document(doc, params.range)


@server.feature(
    types.TEXT_DOCUMENT_ON_TYPE_FORMATTING,
    types.DocumentOnTypeFormattingOptions(first_trigger_character="|"),
)
def format_on_type(
    ls: SwaglangServer, params: types.DocumentOnTypeFormattingParams
) -> List[types.TextEdit]:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return _format_document(doc)


@server.feature(
    types.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
    types.SemanticTokensLegend(
        token_types=TOKEN_TYPES,
        token_modifiers=[m.name for m in TokenModifier if m.name is not None],
    ),
)
def semantic_tokens_full(
    ls: SwaglangServer, params: types.SemanticTokensParams
) -> types.SemanticTokens:
    data: List[int] = []
    tokens = ls.tokens.get(params.text_document.uri, [])
    prev_line, prev_offset = 1, 0

    for token in tokens:
        lsp_type = ANTLR_TO_LSP.get(token.tok_type)
        if lsp_type is None:
            continue
        delta_line = token.line - prev_line
        delta_char = token.offset if delta_line > 0 else token.offset - prev_offset
        data.extend(
            [
                delta_line,
                delta_char,
                len(token.text),
                TOKEN_TYPE_INDEX[lsp_type],
                0,
            ]
        )
        prev_line = token.line
        prev_offset = token.offset

    return types.SemanticTokens(data=data)


def _word_at(document: TextDocument, position: types.Position) -> Optional[str]:
    line = document.lines[position.line]
    char = position.character
    start = char
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
        start -= 1
    end = char
    while end < len(line) and (line[end].isalnum() or line[end] == "_"):
        end += 1
    word = line[start:end]
    return word if word else None


@server.feature(types.TEXT_DOCUMENT_DEFINITION)
def goto_definition(
    ls: SwaglangServer, params: types.DefinitionParams
) -> Optional[types.Location]:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    symbols = ls.all_symbols.get(params.text_document.uri)
    if symbols is None:
        return None

    word = _word_at(doc, params.position)
    if not word:
        return None

    sym = symbols.get(word)
    if sym is None or sym.decl_node is None:
        return None

    decl_line = sym.decl_node.line - 1
    decl_col = sym.decl_node.col

    return types.Location(
        uri=params.text_document.uri,
        range=types.Range(
            start=types.Position(line=decl_line, character=decl_col),
            end=types.Position(line=decl_line, character=decl_col + len(word)),
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_server(server)
