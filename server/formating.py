import logging
from typing import List, Optional

from lsprotocol import types
from pygls.cli import start_server
from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument


server = LanguageServer("formatting-server", "v1")
FORMATTING_SERVER = server


def _format_document(
    document: TextDocument, range_: Optional[types.Range] = None
) -> List[types.TextEdit]:
    edits: List[types.TextEdit] = []
    depth = 0
    for linum, line in enumerate(document.lines):
        if _skip_line(linum, range_):
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


def _skip_line(line: int, range_: Optional[types.Range]) -> bool:
    if range_ is None:
        return False
    return any([line < range_.start.line, line > range_.end.line])


@server.feature(types.TEXT_DOCUMENT_FORMATTING)
def format_document(ls: LanguageServer, params: types.DocumentFormattingParams):
    logging.debug("%s", params)
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return _format_document(doc)


@server.feature(types.TEXT_DOCUMENT_RANGE_FORMATTING)
def format_range(ls: LanguageServer, params: types.DocumentRangeFormattingParams):
    logging.debug("%s", params)
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return _format_document(doc, params.range)


@server.feature(
    types.TEXT_DOCUMENT_ON_TYPE_FORMATTING,
    types.DocumentOnTypeFormattingOptions(first_trigger_character="|"),
)
def format_on_type(ls: LanguageServer, params: types.DocumentOnTypeFormattingParams):
    logging.debug("%s", params)
    doc = ls.workspace.get_text_document(params.text_document.uri)
    return _format_document(doc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_server(server)
