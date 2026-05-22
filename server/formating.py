import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "swaglang"))

import argparse
from antlr4 import FileStream, CommonTokenStream

# from compiler.ast.printer import print_ast
from compiler.lexer.SwagLangLexer import SwagLangLexer
from compiler.lexer.SwagLangParser import SwagLangParser
from compiler.ast.builder import ASTBuilder
from compiler.errors.listener import SwagErrorListener
from compiler.llvm.llvm import LLVMCompiler
from compiler.semantic.analyzer import SemanticAnalyzer
from compiler.semantic.transformer import ASTTransformer


import logging
from typing import Dict
from typing import List
from typing import Optional

import attrs
from lsprotocol import types

from pygls.cli import start_server
from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument


server = LanguageServer("formatting-server", "v1")
FORMATING_SERVER = server


@server.feature(types.TEXT_DOCUMENT_FORMATTING)
def format_document(ls: LanguageServer, params: types.DocumentFormattingParams):
    """Format the entire document"""
    logging.debug("%s", params)

    doc = ls.workspace.get_text_document(params.text_document.uri)
    return format_document(doc)


@server.feature(types.TEXT_DOCUMENT_RANGE_FORMATTING)
def format_range(ls: LanguageServer, params: types.DocumentRangeFormattingParams):
    """Format the given range within a document"""
    logging.debug("%s", params)

    doc = ls.workspace.get_text_document(params.text_document.uri)
    return format_document(doc, params.range)


@server.feature(
    types.TEXT_DOCUMENT_ON_TYPE_FORMATTING,
    types.DocumentOnTypeFormattingOptions(first_trigger_character="|"),
)
def format_on_type(ls: LanguageServer, params: types.DocumentOnTypeFormattingParams):
    """Format the document while the user is typing"""
    logging.debug("%s", params)

    doc = ls.workspace.get_text_document(params.text_document.uri)
    return format_document(doc)


def format_document(
    document: TextDocument, range_: Optional[types.Range] = None
) -> List[types.TextEdit]:
    """Parse the given document into a list of table rows.

    If range_ is given, only consider lines within the range part of the table.
    """
    edits: List[types.TextEdit] = []
    depth = 0
    for linum, line in enumerate(document.lines):
        if skip_line(linum, range_):
            continue
        line = line.lstrip()

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
        if "}" in line:
            depth -= 1
    return edits


def skip_line(line: int, range_: Optional[types.Range]) -> bool:
    """Given a range, determine if we should skip the given line number."""

    if range_ is None:
        return False

    return any([line < range_.start.line, line > range_.end.line])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_server(server)
