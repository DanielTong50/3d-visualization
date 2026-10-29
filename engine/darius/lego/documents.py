"""Acquire a selected official booklet and render bounded, reviewable page images.

Only URLs returned by set-number discovery on the official LEGO service are
accepted. Acquisition records provenance, not training permission. Downloads and
rendered pages belong in ignored engine/outputs/, never in the demo corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from .instructions import InstructionLookupError, _NoRedirect, _official_asset, lookup_instructions

MAX_PDF_BYTES = 32 * 2**20
MAX_RENDER_PAGES = 16
MAX_DOCUMENT_PAGES = 1000
MAX_EDGE_PIXELS = 1600


def _selected_pages(page_count: int, pages: list[int] | None) -> list[int]:
    if not 1 <= page_count <= MAX_DOCUMENT_PAGES:
        raise ValueError("Booklet must have between 1 and 1000 pages")
    chosen = list(range(1, min(MAX_RENDER_PAGES, page_count) + 1)) if pages is None else pages
    if not 1 <= len(chosen) <= MAX_RENDER_PAGES:
        raise ValueError("Choose between 1 and 16 pages per draft")
    if any(not isinstance(p, int) or isinstance(p, bool) or not 1 <= p <= page_count for p in chosen):
        raise ValueError(f"Page indices must be integers between 1 and {page_count}")
    if chosen != sorted(set(chosen)):
        raise ValueError("Page indices must be unique and increasing")
    return chosen


def _render_pdf(pdf_path: Path, destination: Path, pages: list[int] | None) -> tuple[int, list[dict]]:
    try:
        import pypdfium2 as pdfium
    except ImportError as error:
        raise RuntimeError("PDF rendering needs the manuals extra: cd engine && uv sync --extra manuals") from error
    rendered = []
    try:
        with pdfium.PdfDocument(pdf_path) as document:
            page_count = len(document)
            chosen = _selected_pages(page_count, pages)
            for index in chosen:
                page = document[index - 1]
                try:
                    width, height = page.get_size()
                    if not all(math.isfinite(d) and d > 0 for d in (width, height)):
                        raise ValueError("PDF page has invalid dimensions")
                    scale = min(2.0, MAX_EDGE_PIXELS / max(width, height))
                    bitmap = page.render(scale=scale)
                    try:
                        image = bitmap.to_pil().convert("RGB")
                        relative = f"page-{index:03d}.png"
                        path = destination / relative
                        image.save(path)
                        rendered.append(
                            {
                                "page": index,
                                "path": relative,
                                "width": image.width,
                                "height": image.height,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        )
                        image.close()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"The selected PDF could not be rendered: {type(error).__name__}") from error
    return page_count, rendered


def fetch_booklet(
    set_number: str,
    output: Path,
    booklet_index: int = 1,
    pages: list[int] | None = None,
    refresh: bool = False,
) -> dict:
    """Download an exact 1-based booklet choice, hash it and render at most 16 pages.

    A short selection is explicitly a partial document. No calls to a paid vision
    model are made. The output manifest remains permission-required/not-parsed.
    """
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Booklet output directory is not empty: {output}")
    result = lookup_instructions(set_number, refresh=refresh)
    if (
        not isinstance(booklet_index, int)
        or isinstance(booklet_index, bool)
        or not 1 <= booklet_index <= len(result.booklets)
    ):
        raise ValueError(f"booklet_index must be between 1 and {len(result.booklets)}")
    booklet = result.booklets[booklet_index - 1]
    pdf_url = _official_asset(booklet.pdf_url)
    if pdf_url is None or not urllib.parse.urlsplit(pdf_url).path.lower().endswith(".pdf"):
        raise InstructionLookupError("The selected booklet does not have an official HTTPS PDF URL")
    if booklet.size_bytes and booklet.size_bytes > MAX_PDF_BYTES:
        raise ValueError("Selected booklet exceeds the 32MB acquisition limit")
    # Official additional-booklet URLs occasionally contain literal spaces.
    parsed = urllib.parse.urlsplit(pdf_url)
    safe_url = urllib.parse.urlunsplit(
        parsed._replace(path=urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/"))
    )
    request = urllib.request.Request(
        safe_url, headers={"User-Agent": "Darius-InstructionDiscovery/1.0", "Accept": "application/pdf"}
    )
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=30) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"application/pdf", "application/octet-stream"}:
                raise ValueError("Official booklet did not return a PDF content type")
            data = response.read(MAX_PDF_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise InstructionLookupError(f"Official booklet acquisition failed: {type(error).__name__}") from error
    if len(data) > MAX_PDF_BYTES:
        raise ValueError("Official booklet exceeds the 32MB acquisition limit")
    if not data.startswith(b"%PDF-"):
        raise ValueError("Official booklet response is not a PDF")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".darius-booklet-", dir=output.parent) as temporary:
        working = Path(temporary)
        pdf_path = working / "source.pdf"
        pdf_path.write_bytes(data)
        page_count, rendered = _render_pdf(pdf_path, working, pages)
        manifest = {
            "schema_version": "darius.manual-acquisition.v1",
            "set_number": set_number,
            "set_title": result.title,
            "official_url": result.official_url,
            "pdf_url": pdf_url,
            "booklet_index": booklet_index,
            "booklet": booklet.model_dump(mode="json"),
            "acquired_at": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "document_pages": page_count,
            "selected_pages": [r["page"] for r in rendered],
            "partial_document": len(rendered) < page_count,
            "pages": rendered,
            "semantic_status": "not_parsed",
            "training_rights": "permission-required",
            "notice": (
                "Acquisition and rasterization only. Review booklet/alternate-build selection. "
                "No reconstructed poses or training permission implied."
            ),
        }
        (working / "acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Commit a completed sibling directory atomically, avoiding a partially
        # copied acquisition if rendering or I/O fails. Recheck before replacing.
        if output.exists():
            if not output.is_dir() or any(output.iterdir()):
                raise ValueError(f"Booklet output directory is not empty: {output}")
            output.rmdir()
        working.rename(output)
    return {
        "manifest": manifest,
        "pdf_path": str(output / "source.pdf"),
        "page_paths": [str(output / r["path"]) for r in rendered],
    }
