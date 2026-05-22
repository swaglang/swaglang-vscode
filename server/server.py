import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "swaglang"))

import random

from diagnostics import DIAGNOSTICS_SERVER
from formating import FORMATING_SERVER

import logging
from pygls.cli import start_server
from pygls.lsp.server import LanguageServer
from pygls.exceptions import PyglsError, JsonRpcException
import enum
from functools import reduce
import operator

import argparse
from antlr4 import InputStream, CommonTokenStream

# from compiler.ast.printer import print_ast
from compiler.lexer.SwagLangLexer import SwagLangLexer
from compiler.lexer.SwagLangParser import SwagLangParser
from compiler.ast.builder import ASTBuilder
from compiler.errors.listener import SwagErrorListener
from compiler.llvm.llvm import LLVMCompiler
from compiler.semantic.analyzer import SemanticAnalyzer
from compiler.semantic.transformer import ASTTransformer

import logging
import re
import io

from lsprotocol import types

from pygls.cli import start_server
from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument

from typing import Dict
from typing import List
from typing import Optional

import attrs
from lsprotocol import types

from pygls.cli import start_server
from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument

"""
DIAGNOSTICS AND INIT
"""

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
ANTLR_TO_LSP = {
    # keywords
    "IF": "keyword",
    "ELSE": "keyword",
    "WHILE": "keyword",
    "FOR": "keyword",
    "RETURN": "keyword",
    "LET": "keyword",
    "CONST": "keyword",
    "INTERFACE": "keyword",
    "EXTENDS": "keyword",
    "DO": "keyword",
    "BREAK": "keyword",
    "CONTINUE": "keyword",
    "DEFER": "keyword",
    "IN": "keyword",
    "VOID": "keyword",
    # types
    "TYPE": "type",
    "MAP": "type",
    "SET": "type",
    # literals
    "STRING": "string",
    "INT": "number",
    "FLOAT": "number",
    "BOOL": "keyword",
    "NULL": "keyword",
    # identifiers
    "IDENT": "variable",
    # operators
    "ASSIGN": "operator",
    "ADD_ASSIGN": "operator",
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


class SwaglangServer(LanguageServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics = {}
        self.tree, self.parser_errors = None, None
        self.tokens: Dict[str, List[Token]] = {}

    def get_parser_results(self, document: TextDocument):
        stream = InputStream(document.source)

        lexer = SwagLangLexer(stream)
        error_listener = SwagErrorListener(document.uri)

        lexer.removeErrorListeners()
        lexer.addErrorListener(error_listener)

        tokens = CommonTokenStream(lexer)
        parser = SwagLangParser(tokens)

        parser.removeErrorListeners()
        parser.addErrorListener(error_listener)

        tree = parser.prog()
        tokens = self.format_tokens(tokens, parser)
        return tree if not error_listener.errors else [], tokens, error_listener.errors

    def create_diagnostics(self, document: TextDocument):
        diagnostics = []
        parse_errors, tree = self.parser_errors, self.tree
        sem_errors = []
        if tree and tree != []:
            ast = ASTBuilder().visit(tree)
            symbols, sem_types, sem_errors = SemanticAnalyzer(
                document.filename
            ).analyze(ast)

        for error in parse_errors + sem_errors:
            severity = types.DiagnosticSeverity.Error
            line = error.line
            # TODO: solve why line number can be outside our document range
            if line >= len(document.lines):
                continue

            diagnostics.append(
                types.Diagnostic(
                    message=error.message,
                    severity=severity,
                    range=types.Range(
                        start=types.Position(line=line, character=0),
                        end=types.Position(
                            line=line, character=len(document.lines[line]) - 1
                        ),
                    ),
                )
            )
        return diagnostics

    def create_test_diagnostics(self, document: TextDocument):
        diagnostics = []
        for idx, line in enumerate(document.lines):
            match = ADDITION.match(line)
            if match is not None:
                left = int(match.group(1))
                right = int(match.group(2))

                expected_answer = left + right
                actual_answer = match.group(3)
                if actual_answer is not None and expected_answer == int(actual_answer):
                    continue

                if actual_answer is None:
                    message = "Missing answer"
                    severity = types.DiagnosticSeverity.Warning
                else:
                    message = f"Incorrect answer: {actual_answer}"
                    severity = types.DiagnosticSeverity.Error

                diagnostics.append(
                    types.Diagnostic(
                        message=message,
                        severity=severity,
                        range=types.Range(
                            start=types.Position(line=idx, character=0),
                            end=types.Position(line=idx, character=len(line) - 1),
                        ),
                    )
                )
        return diagnostics

    def format_tokens(self, inp_tokens, parser):
        tokens = []
        # i = 0
        for t in inp_tokens.tokens:
            # i += 1
            # if i == 4:
            #     raise Exception(f""
            #                     f"{t.line}\n"
            #                     f"{t.column}\n"
            #                     f"{t.text}\n"
            #                     f"{parser.symbolicNames[t.type]}")
            tokens.append(
                Token(
                    line=t.line,
                    offset=t.column,
                    text=t.text,
                    tok_type=parser.symbolicNames[t.type],
                )
            )

        return tokens

    def parse(self, document: TextDocument):
        self.tree, tokens, self.parser_errors = self.get_parser_results(document)
        self.tokens[document.uri] = tokens
        self.diagnostics[document.uri] = (
            document.version,
            self.create_diagnostics(document),
        )

    def format_document(
        document: TextDocument, range_: Optional[types.Range] = None
    ) -> List[types.TextEdit]:

        def skip_line(line: int, range_: Optional[types.Range]) -> bool:
            if range_ is None:
                return False
            return any([line < range_.start.line, line > range_.end.line])

        edits: List[types.TextEdit] = []
        depth = 0
        for linum, line in enumerate(document.lines):
            if skip_line(linum, range_):
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

    """
    Report error in server execution, not from language analysis
    """

    def report_server_error(
        self, error: Exception, source: PyglsError | JsonRpcException
    ):
        self.window_show_message(
            types.ShowMessageParams(
                message=f"Error in server: {error}",
                type=types.MessageType.Error,
            )
        )
        tb = "".join(error.data["traceback"])
        self.window_log_message(
            types.ShowMessageParams(
                message=f"\nError in server: {error} \n {tb}",
                type=types.MessageType.Error,
            )
        )


server = SwaglangServer("swaglang-server", "v1")


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: SwaglangServer, params: types.DidOpenTextDocumentParams):
    """Parse each document when it is opened"""
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse(doc)

    for uri, (version, diagnostics) in ls.diagnostics.items():
        ls.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=uri,
                version=version,
                diagnostics=diagnostics,
            )
        )


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: SwaglangServer, params: types.DidOpenTextDocumentParams):
    """Parse each document when it is changed"""
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse(doc)

    for uri, (version, diagnostics) in ls.diagnostics.items():
        ls.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=uri,
                version=version,
                diagnostics=diagnostics,
            )
        )


"""
FORMATING
"""


@attrs.define
class Row:
    """Represents a row in the table"""

    cells: List[str]
    cell_widths: List[int]
    line_number: int


@server.feature(types.TEXT_DOCUMENT_FORMATTING)
def format_document(ls: LanguageServer, params: types.DocumentFormattingParams):
    """Format the entire document"""
    logging.debug("%s", params)

    doc = ls.workspace.get_text_document(params.text_document.uri)
    return ls.format_document(doc)


@server.feature(types.TEXT_DOCUMENT_RANGE_FORMATTING)
def format_range(ls: LanguageServer, params: types.DocumentRangeFormattingParams):
    """Format the given range within a document"""
    logging.debug("%s", params)

    doc = ls.workspace.get_text_document(params.text_document.uri)
    return ls.format_document(doc, params.range)


@server.feature(
    types.TEXT_DOCUMENT_ON_TYPE_FORMATTING,
    types.DocumentOnTypeFormattingOptions(first_trigger_character="|"),
)
def format_on_type(ls: LanguageServer, params: types.DocumentOnTypeFormattingParams):
    """Format the document while the user is typing"""
    logging.debug("%s", params)

    doc = ls.workspace.get_text_document(params.text_document.uri)
    return ls.format_document(doc)


"""Tokens"""


@server.feature(
    types.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
    types.SemanticTokensLegend(
        token_types=TOKEN_TYPES,
        token_modifiers=[m.name for m in TokenModifier if m.name is not None],
    ),
)
def semantic_tokens_full(ls: SwaglangServer, params):
    data = []
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
                0,  # no modifiers for now
            ]
        )
        prev_line = token.line
        prev_offset = token.offset

    return types.SemanticTokens(data=data)


"""
ENTRYPOINT
"""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_server(server)
