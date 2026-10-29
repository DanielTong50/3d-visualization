"""Acquisition boundary: use discovered official links, bound work, preserve provenance."""

import hashlib
import io
import json
from email.message import Message

import pytest

from darius.lego import documents
from darius.lego.instructions import InstructionLookupError, lookup_instructions


class Response(io.BytesIO):
    def __init__(self, body: bytes, content_type: str = "application/pdf"):
        super().__init__(body)
        self.headers = Message()
        self.headers["Content-Type"] = content_type


def setup_download(monkeypatch, body=b"%PDF-1.7\nmock", content_type="application/pdf"):
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append((request.full_url, timeout))
            return Response(body, content_type)

    def render(pdf_path, destination, pages):
        assert pdf_path.read_bytes() == body
        path = destination / "page-003.png"
        path.write_bytes(b"rasterized-page")
        return 10, [{"page": 3, "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]

    monkeypatch.setattr(documents.urllib.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(documents, "_render_pdf", render)
    return requests


def test_official_selection_has_hashes_and_explicit_partial_document(monkeypatch, tmp_path):
    requests = setup_download(monkeypatch)
    output = tmp_path / "booklet"
    result = documents.fetch_booklet("10698", output, booklet_index=4, pages=[3])
    manifest = result["manifest"]
    assert requests == [
        ("https://www.lego.com/cdn/product-assets/product.bi.additional.extra.pdf/10698_X_Camera.pdf", 30)
    ]
    assert manifest["training_rights"] == "permission-required"
    assert manifest["semantic_status"] == "not_parsed"
    assert manifest["partial_document"] and manifest["selected_pages"] == [3]
    assert hashlib.sha256((output / "source.pdf").read_bytes()).hexdigest() == manifest["sha256"]
    assert json.loads((output / "acquisition.json").read_text()) == manifest
    assert result["page_paths"] == [str(output / "page-003.png")]
    with pytest.raises(ValueError, match="not empty"):
        documents.fetch_booklet("10698", output)


def test_bad_payload_does_not_leave_an_acquisition_directory(monkeypatch, tmp_path):
    setup_download(monkeypatch, b"<html>not a PDF</html>")
    with pytest.raises(ValueError, match="not a PDF"):
        documents.fetch_booklet("10698", tmp_path / "bad", booklet_index=4)
    assert not (tmp_path / "bad").exists()
    setup_download(monkeypatch, b"%PDF-1.7\nmock", "text/html")
    with pytest.raises(ValueError, match="content type"):
        documents.fetch_booklet("10698", tmp_path / "bad", booklet_index=4)


def test_unofficial_and_oversized_booklets_are_rejected_before_download(monkeypatch, tmp_path):
    discovered = lookup_instructions("10698")
    discovered.booklets[0].pdf_url = "https://example.com/instructions.pdf"
    monkeypatch.setattr(documents, "lookup_instructions", lambda *a, **k: discovered)
    with pytest.raises(InstructionLookupError, match="official HTTPS"):
        documents.fetch_booklet("10698", tmp_path / "bad")
    discovered.booklets[0].pdf_url = "https://www.lego.com/cdn/example.pdf"
    discovered.booklets[0].size_bytes = documents.MAX_PDF_BYTES + 1
    with pytest.raises(ValueError, match="32MB"):
        documents.fetch_booklet("10698", tmp_path / "large")


def test_render_failures_leave_no_partial_output(monkeypatch, tmp_path):
    setup_download(monkeypatch)

    def failed(*args):
        raise ValueError("rendering failed")

    monkeypatch.setattr(documents, "_render_pdf", failed)
    with pytest.raises(ValueError, match="rendering failed"):
        documents.fetch_booklet("10698", tmp_path / "failed", booklet_index=4)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".darius-booklet-*"))


@pytest.mark.parametrize("pages", [[], [0], [11], [2, 1], [1, 1], [True], list(range(1, 18))])
def test_render_selection_is_bounded_unique_ordered_and_in_range(pages):
    with pytest.raises(ValueError):
        documents._selected_pages(10, pages)
    assert documents._selected_pages(100, None) == list(range(1, 17))
