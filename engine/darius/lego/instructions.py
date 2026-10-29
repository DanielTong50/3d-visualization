"""Discover official public instruction booklets by a validated set number.

Only the fixed LEGO.com instructions endpoint is fetched, with bounded response
size, a timeout, and redirects disabled. PDF links are metadata, never arbitrary
server-side fetch targets. Public access does not confer dataset/training rights.
The HTML adapter is deliberately isolated because the upstream markup may change.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from .schema import Strict

SET_NUMBER = re.compile(r"^[0-9]{4,7}$")
MAX_PAGE_BYTES = 3 * 2**20
CACHE_DIR = Path(__file__).resolve().parents[3] / "assemblies" / "lego" / "instructions"


class InstructionBooklet(Strict):
    title: str
    pdf_url: str
    cover_url: str | None = None
    size_bytes: int | None = None
    sequence: int = 1
    total: int = 1
    additional: bool = False


class InstructionResult(Strict):
    set_number: str
    title: str
    official_url: str
    year: str | None = None
    piece_count: int | None = None
    theme: str | None = None
    image_url: str | None = None
    booklets: list[InstructionBooklet] = Field(default_factory=list)
    retrieved_at: str
    retrieval: Literal["live", "cached"] = "live"
    semantic_status: Literal["not_parsed"] = "not_parsed"
    training_rights: Literal["permission-required"] = "permission-required"
    notice: str = (
        "Official LEGO instruction links. Availability varies by set. Public viewing is separate from "
        "permission to redistribute or train on manuals. This response has not parsed the PDF; "
        "the optional manual-draft adapter requires review."
    )


class InstructionLookupError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise InstructionLookupError("Official lookup redirected; open the official instruction page instead.")


def official_url(set_number: str) -> str:
    if not SET_NUMBER.fullmatch(set_number):
        raise ValueError("set number must contain 4 to 7 digits")
    return f"https://www.lego.com/en-us/service/building-instructions/{set_number}"


def _official_asset(url: str | None) -> str | None:
    if not isinstance(url, str) or not url:
        return None
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in {"www.lego.com", "lego.com", "www.legocdn.com"}:
            return None
        if parsed.username or parsed.password or parsed.port not in (None, 443):
            return None
    except ValueError:
        return None
    return url


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value, default=None):
    try:
        result = int(value)
        return result if result >= 0 else default
    except (TypeError, ValueError):
        return default


def parse_official_page(set_number: str, html: str) -> InstructionResult:
    """Parse only the set's embedded official structured data, not unrelated page links."""
    url = official_url(set_number)
    match = re.search(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', html, re.S)
    if not match:
        raise InstructionLookupError("LEGO page format changed; use the official link to select booklets.")
    try:
        state = json.loads(match.group(1))
    except (json.JSONDecodeError, RecursionError) as error:
        raise InstructionLookupError("Official page metadata could not be read.") from error
    found = None
    # Traverse bounded parsed page data; only accept exact setNumber and booklet fields.
    pending = [state]
    inspected = 0
    while pending and inspected < 100000:
        item = pending.pop()
        inspected += 1
        if isinstance(item, dict):
            if str(item.get("setNumber")) == set_number and isinstance(item.get("buildingInstructions"), list):
                found = item
                break
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    if found is None:
        raise InstructionLookupError(f"No instruction booklets found for set {set_number}; check the official page.")
    booklets = []
    for item in found.get("buildingInstructions", [])[:100]:
        item = _object(item)
        pdf = _object(item.get("pdf"))
        pdf_url = _official_asset(pdf.get("pdfUrl"))
        if not pdf_url:
            continue
        seq = _object(item.get("sequence"))
        sequence, total = _number(seq.get("element"), 1), _number(seq.get("total"), 1)
        booklets.append(
            InstructionBooklet(
                title=f"{found.get('name', set_number)} · booklet {sequence}/{total}",
                pdf_url=pdf_url,
                cover_url=_official_asset(_object(pdf.get("coverImage")).get("src")),
                size_bytes=_number(pdf.get("fileSize")),
                sequence=sequence,
                total=total,
                additional=bool(item.get("isAdditionalInfoBooklet", False)),
            )
        )
    if not booklets:
        raise InstructionLookupError("The official page has no supported downloadable instruction links.")
    return InstructionResult(
        set_number=set_number,
        title=str(found.get("name") or set_number),
        official_url=url,
        year=str(found["year"]) if found.get("year") else None,
        piece_count=_number(found.get("setPieceCount")),
        theme=_object(found.get("theme")).get("themeName"),
        image_url=_official_asset(_object(found.get("setImage")).get("src")),
        booklets=booklets,
        retrieved_at=datetime.now(UTC).isoformat(),
    )


def lookup_instructions(set_number: str, *, refresh: bool = False) -> InstructionResult:
    """Return checked-in metadata if present, else retrieve the official set page.

    Cache files contain metadata only; request handlers never write to the cache.
    """
    url = official_url(set_number)
    cache = CACHE_DIR / f"{set_number}.json"
    if not refresh and cache.is_file():
        result = InstructionResult.model_validate_json(cache.read_text())
        return result.model_copy(update={"retrieval": "cached"})
    request = urllib.request.Request(
        url, headers={"User-Agent": "Darius-InstructionDiscovery/1.0 (public set lookup)", "Accept": "text/html"}
    )
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=15) as response:
            if response.headers.get_content_type() != "text/html":
                raise InstructionLookupError("The official instructions service did not return an HTML page.")
            body = response.read(MAX_PAGE_BYTES + 1)
            if len(body) > MAX_PAGE_BYTES:
                raise InstructionLookupError("The official instructions page exceeds the 3MB response limit.")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise InstructionLookupError(
            f"Official instructions are temporarily unavailable: {type(error).__name__}"
        ) from error
    return parse_official_page(set_number, body.decode("utf-8", errors="replace"))
