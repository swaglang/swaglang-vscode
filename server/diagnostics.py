import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'swaglang'))

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

ADDITION = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)?\s*$")

class PublishDiagnosticServer(LanguageServer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics = {}


    def get_parser_errors(self, document: TextDocument):
            stream = InputStream(document.source)
            lexer = SwagLangLexer(stream)

            lexer = SwagLangLexer(stream)
            error_listener = SwagErrorListener(document.uri)

            lexer.removeErrorListeners()
            lexer.addErrorListener(error_listener)

            tokens = CommonTokenStream(lexer)
            parser = SwagLangParser(tokens)

            parser.removeErrorListeners()
            parser.addErrorListener(error_listener)

            tree = parser.prog()

            return error_listener.errors, None if not error_listener.errors else []

    def create_diagnostics(self, document: TextDocument):
        diagnostics = []
        parse_errors, tree = self.get_parser_errors(document)
        sem_errors = []
        if tree:
            ast = ASTBuilder().visit(tree)
            symbols, sem_types, sem_errors = SemanticAnalyzer(document.filename).analyze(ast)

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
                        end=types.Position(line=line, character=len(document.lines[line]) - 1),
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


    def parse(self, document: TextDocument):
        self.diagnostics[document.uri] = (document.version, self.create_diagnostics(document))


server = PublishDiagnosticServer("diagnostic-server", "v1")
DIAGNOSTICS_SERVER = server

@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: PublishDiagnosticServer, params: types.DidOpenTextDocumentParams):
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
def did_change(ls: PublishDiagnosticServer, params: types.DidOpenTextDocumentParams):
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_server(server)
