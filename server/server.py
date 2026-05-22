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
from compiler.ast.nodes import (
    ArrayType, BaseType, FuncDecl, MapType, MultiReturnType,
    SetType, SingleReturnType, UserType, VoidReturnType,
)
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


def _fmt_type(t) -> str:
    if t is None:
        return "unknown"
    if isinstance(t, BaseType):
        return t.value
    if isinstance(t, UserType):
        return t.name
    if isinstance(t, ArrayType):
        return f"{_fmt_type(t.element)}[]"
    if isinstance(t, MapType):
        return f"map<{_fmt_type(t.key)}, {_fmt_type(t.value)}>"
    if isinstance(t, SetType):
        return f"set<{_fmt_type(t.element)}>"
    return str(t)


# Human-readable signatures for builtins
# Update this when builtins.py changes
_BUILTIN_DISPLAY: Dict[str, str] = {
    "println": "fn println(value: any)",
    "print":   "fn print(value: any)",
    "len":     "fn len(value: string | T[] | set<T> | map<K,V>) -> int",
    "range":   "fn range(n: int) -> int[]\nfn range(start: int, end: int) -> int[]\nfn range(start: int, end: int, step: int) -> int[]",
    "input":   "fn input(prompt: string) -> string",
}


def _fmt_func(sym: Symbol) -> str:
    if not isinstance(sym.decl_node, FuncDecl):
        display = _BUILTIN_DISPLAY.get(sym.name)
        if display:
            return display
        return f"fn {sym.name}(...)"
    decl: FuncDecl = sym.decl_node
    params = ", ".join(
        f"{p.name}: {_fmt_type(p.type_ann)}" for p in decl.params
    )
    if isinstance(decl.return_type, VoidReturnType):
        ret = ""
    elif isinstance(decl.return_type, SingleReturnType):
        ret = f" -> {_fmt_type(decl.return_type.type_ann)}"
    elif isinstance(decl.return_type, MultiReturnType):
        ret = " -> (" + ", ".join(_fmt_type(t) for t in decl.return_type.types) + ")"
    else:
        ret = ""
    return f"fn {sym.name}({params}){ret}"


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
    if sym is None or sym.decl_node is None or sym.decl_node.line == 0:
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


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(ls: SwaglangServer, params: types.HoverParams) -> Optional[types.Hover]:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    symbols = ls.all_symbols.get(params.text_document.uri)
    if symbols is None:
        return None

    word = _word_at(doc, params.position)
    if not word:
        return None

    sym = symbols.get(word)
    if sym is None:
        return None

    kind_label = sym.kind.value  # "function" | "variable" | "parameter" | "interface"
    mut = "" if sym.kind in (SymbolKind.FUNCTION, SymbolKind.INTERFACE) else (
        "let " if sym.is_mutable else "const "
    )

    if sym.kind == SymbolKind.FUNCTION:
        type_str = _fmt_func(sym)
    elif sym.kind == SymbolKind.INTERFACE:
        type_str = f"interface {sym.name}"
    else:
        type_str = f"{mut}{sym.name}: {_fmt_type(sym.type)}"

    content = f"```swaglang\n{type_str}\n```\n*{kind_label}*"

    return types.Hover(
        contents=types.MarkupContent(
            kind=types.MarkupKind.Markdown,
            value=content,
        )
    )


_SYMBOL_KIND_TO_LSP_KIND: Dict[SymbolKind, types.SymbolKind] = {
    SymbolKind.FUNCTION:  types.SymbolKind.Function,
    SymbolKind.VARIABLE:  types.SymbolKind.Variable,
    SymbolKind.PARAMETER: types.SymbolKind.Variable,
    SymbolKind.INTERFACE: types.SymbolKind.Interface,
}


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(
    ls: SwaglangServer, params: types.DocumentSymbolParams
) -> List[types.SymbolInformation]:
    symbols = ls.all_symbols.get(params.text_document.uri, {})
    result: List[types.SymbolInformation] = []

    for sym in symbols.values():
        if sym.decl_node is None or sym.decl_node.line == 0:
            continue
        decl_line = sym.decl_node.line - 1
        decl_col = sym.decl_node.col
        lsp_kind = _SYMBOL_KIND_TO_LSP_KIND.get(sym.kind, types.SymbolKind.Variable)
        result.append(
            types.SymbolInformation(
                name=sym.name,
                kind=lsp_kind,
                location=types.Location(
                    uri=params.text_document.uri,
                    range=types.Range(
                        start=types.Position(line=decl_line, character=decl_col),
                        end=types.Position(line=decl_line, character=decl_col + len(sym.name)),
                    ),
                ),
            )
        )

    return result


_KEYWORDS = [
    "if", "else", "while", "for", "do", "in", "break", "continue",
    "return", "defer", "interface", "extends", "void", "let", "const",
    "true", "false", "null", "map", "set", "int", "float", "string",
    "bool", "error",
]

_SYMBOL_KIND_TO_COMPLETION_KIND: Dict[SymbolKind, types.CompletionItemKind] = {
    SymbolKind.FUNCTION:  types.CompletionItemKind.Function,
    SymbolKind.VARIABLE:  types.CompletionItemKind.Variable,
    SymbolKind.PARAMETER: types.CompletionItemKind.Variable,
    SymbolKind.INTERFACE: types.CompletionItemKind.Interface,
}


@server.feature(
    types.TEXT_DOCUMENT_COMPLETION,
    types.CompletionOptions(trigger_characters=["."]),
)
def completion(
    ls: SwaglangServer, params: types.CompletionParams
) -> types.CompletionList:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    symbols = ls.all_symbols.get(params.text_document.uri, {})

    word = _word_at(doc, params.position) or ""
    items: List[types.CompletionItem] = []

    # Symbol completions
    for name, sym in symbols.items():
        if word and not name.startswith(word):
            continue
        kind = _SYMBOL_KIND_TO_COMPLETION_KIND.get(sym.kind, types.CompletionItemKind.Variable)
        if sym.kind == SymbolKind.FUNCTION:
            detail = _fmt_func(sym)
        elif sym.kind == SymbolKind.INTERFACE:
            detail = f"interface {name}"
        else:
            detail = f"{name}: {_fmt_type(sym.type)}"
        items.append(
            types.CompletionItem(  # type: ignore[call-arg]
                label=name,
                kind=kind,
                detail=detail,
            )
        )

    # Keyword completions
    for kw in _KEYWORDS:
        if word and not kw.startswith(word):
            continue
        items.append(
            types.CompletionItem(  # type: ignore[call-arg]
                label=kw,
                kind=types.CompletionItemKind.Keyword,
            )
        )

    return types.CompletionList(is_incomplete=False, items=items)


def _active_param_index(line: str, char: int) -> int:
    """Count commas at nesting depth 1 to find active parameter index."""
    depth = 0
    commas = 0
    for i, ch in enumerate(line[:char]):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 1:
            commas += 1
    return commas


def _func_name_before_paren(line: str, char: int) -> Optional[str]:
    """Find the function name that opened the innermost unclosed '('."""
    depth = 0
    for i in range(char - 1, -1, -1):
        ch = line[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            if depth == 0:
                # scan backwards for identifier
                j = i - 1
                while j >= 0 and line[j] in " \t":
                    j -= 1
                end = j + 1
                while j >= 0 and (line[j].isalnum() or line[j] == "_"):
                    j -= 1
                return line[j + 1:end] or None
            depth -= 1
    return None


@server.feature(
    types.TEXT_DOCUMENT_SIGNATURE_HELP,
    types.SignatureHelpOptions(trigger_characters=["(", ","]),
)
def signature_help(
    ls: SwaglangServer, params: types.SignatureHelpParams
) -> Optional[types.SignatureHelp]:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    symbols = ls.all_symbols.get(params.text_document.uri, {})

    line_text = doc.lines[params.position.line]
    char = params.position.character

    func_name = _func_name_before_paren(line_text, char)
    if not func_name:
        return None

    sym = symbols.get(func_name)
    if sym is None or sym.kind != SymbolKind.FUNCTION:
        return None
    if not isinstance(sym.decl_node, FuncDecl):
        return None

    decl: FuncDecl = sym.decl_node
    params_info = [
        types.ParameterInformation(
            label=f"{p.name}: {_fmt_type(p.type_ann)}"
        )
        for p in decl.params
    ]

    label = _fmt_func(sym)
    sig = types.SignatureInformation(
        label=label,
        parameters=params_info,
    )

    active_param = _active_param_index(line_text, char)

    return types.SignatureHelp(
        signatures=[sig],
        active_signature=0,
        active_parameter=min(active_param, max(len(decl.params) - 1, 0)),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_server(server)
