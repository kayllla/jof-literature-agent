"""Build four types of structured chunks from parsed JSON blocks."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

from ingest.metadata_extractor import PaperMetadata

logger = logging.getLogger(__name__)

NOISE_TYPES = {"header", "page_number", "aside_text", "footer"}
TEXT_CHUNK_MAX_CHARS = 1500


@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str
    paper_id: str
    title: str | None
    doi: str | None
    text: str  # the searchable text for embedding

    # optional fields depending on chunk_type
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    page_idx: int | None = None
    img_path: str | None = None
    caption_text: str | None = None
    equation_text: str | None = None
    context_before: str | None = None
    context_after: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Text chunks
# ---------------------------------------------------------------------------

def _build_text_chunks(blocks: list[dict], meta: PaperMetadata) -> list[Chunk]:
    """Concatenate consecutive text blocks per section, splitting at section
    headings and when the cumulative length exceeds the threshold."""

    chunks: list[Chunk] = []
    current_section: str | None = None
    buffer: list[str] = []
    page_start: int | None = None
    page_end: int | None = None

    first_page_title_consumed = False
    abstract_consumed = False

    for block in blocks:
        btype = block.get("type")
        if btype in NOISE_TYPES or btype == "ref_text":
            continue

        page = block.get("page_idx", -1)

        # Skip first-page metadata blocks (title, authors, abstract, DOI)
        if page == 0 and btype == "text":
            text = block.get("text", "").strip()
            if not first_page_title_consumed and block.get("text_level") == 1 and text.upper() != "ABSTRACT":
                first_page_title_consumed = True
                continue
            if text.upper().startswith("ABSTRACT"):
                abstract_consumed = False
                continue
            if not abstract_consumed and first_page_title_consumed:
                # Skip author block(s) and abstract body
                if block.get("text_level") == 1:
                    abstract_consumed = True
                elif "DOI:" in text or "doi:" in text:
                    continue
                else:
                    continue

        if btype == "text" and block.get("text_level") == 1:
            # New section heading → flush buffer
            if buffer:
                chunks.append(_make_text_chunk(
                    buffer, current_section, page_start, page_end, meta
                ))
                buffer = []
            current_section = block.get("text", "").strip()
            page_start = page
            page_end = page
            continue

        if btype in ("text", "list", "page_footnote"):
            txt = _extract_text(block)
            if not txt:
                continue

            if page_start is None:
                page_start = page
            page_end = page

            buffer.append(txt)

            if sum(len(t) for t in buffer) >= TEXT_CHUNK_MAX_CHARS:
                chunks.append(_make_text_chunk(
                    buffer, current_section, page_start, page_end, meta
                ))
                buffer = []
                page_start = None
                page_end = None

    if buffer:
        chunks.append(_make_text_chunk(
            buffer, current_section, page_start, page_end, meta
        ))

    return chunks


def _extract_text(block: dict) -> str:
    btype = block.get("type")
    if btype == "list":
        items = block.get("list_items", [])
        return " ".join(items).strip()
    if btype == "page_footnote":
        return block.get("text", "").strip()
    return block.get("text", "").strip()


def _make_text_chunk(
    buffer: list[str],
    section: str | None,
    page_start: int | None,
    page_end: int | None,
    meta: PaperMetadata,
) -> Chunk:
    return Chunk(
        chunk_id=f"{meta.paper_id}_text_{_uid()}",
        chunk_type="text",
        paper_id=meta.paper_id,
        title=meta.title,
        doi=meta.doi,
        text=" ".join(buffer),
        section=section,
        page_start=page_start,
        page_end=page_end,
    )


# ---------------------------------------------------------------------------
# Figure-caption chunks
# ---------------------------------------------------------------------------

def _build_figure_caption_chunks(blocks: list[dict], meta: PaperMetadata) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in blocks:
        if block.get("type") != "image":
            continue
        captions = block.get("image_caption", [])
        if not captions:
            continue
        caption_text = " ".join(captions).strip()
        if not caption_text:
            continue
        chunks.append(Chunk(
            chunk_id=f"{meta.paper_id}_fig_{_uid()}",
            chunk_type="figure_caption",
            paper_id=meta.paper_id,
            title=meta.title,
            doi=meta.doi,
            text=caption_text,
            page_idx=block.get("page_idx"),
            img_path=block.get("img_path"),
            caption_text=caption_text,
        ))
    return chunks


# ---------------------------------------------------------------------------
# Table-caption chunks
# ---------------------------------------------------------------------------

def _build_table_caption_chunks(blocks: list[dict], meta: PaperMetadata) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in blocks:
        if block.get("type") != "table":
            continue
        captions = block.get("table_caption", [])
        if not captions:
            continue
        caption_text = " ".join(captions).strip()
        if not caption_text:
            continue
        chunks.append(Chunk(
            chunk_id=f"{meta.paper_id}_tab_{_uid()}",
            chunk_type="table_caption",
            paper_id=meta.paper_id,
            title=meta.title,
            doi=meta.doi,
            text=caption_text,
            page_idx=block.get("page_idx"),
            caption_text=caption_text,
        ))
    return chunks


# ---------------------------------------------------------------------------
# Equation-context chunks
# ---------------------------------------------------------------------------

def _build_equation_context_chunks(blocks: list[dict], meta: PaperMetadata) -> list[Chunk]:
    """For each equation block, merge it with the nearest preceding and following
    text blocks to create a context-enriched chunk."""

    text_blocks = [b for b in blocks if b.get("type") == "text"]
    chunks: list[Chunk] = []

    for i, block in enumerate(blocks):
        if block.get("type") != "equation":
            continue

        eq_text = block.get("text", "").strip()
        if not eq_text:
            continue

        ctx_before = _find_nearest_text(blocks, i, direction=-1)
        ctx_after = _find_nearest_text(blocks, i, direction=1)
        merged = f"{ctx_before} {eq_text} {ctx_after}".strip()

        chunks.append(Chunk(
            chunk_id=f"{meta.paper_id}_eq_{_uid()}",
            chunk_type="equation_context",
            paper_id=meta.paper_id,
            title=meta.title,
            doi=meta.doi,
            text=merged,
            page_idx=block.get("page_idx"),
            equation_text=eq_text,
            context_before=ctx_before or None,
            context_after=ctx_after or None,
        ))

    return chunks


def _find_nearest_text(blocks: list[dict], idx: int, direction: int) -> str:
    """Walk from *idx* in *direction* (-1 or +1) to find the nearest text block."""
    i = idx + direction
    while 0 <= i < len(blocks):
        b = blocks[i]
        if b.get("type") == "text" and b.get("text", "").strip():
            return b["text"].strip()
        i += direction
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_chunks(blocks: list[dict], meta: PaperMetadata) -> tuple[list[Chunk], list[Chunk]]:
    """Build all chunks for a single paper.

    Returns:
        A tuple of (main_chunks, equation_chunks) where main_chunks includes
        text + figure-caption + table-caption, and equation_chunks contains
        equation-context chunks.
    """
    text_chunks = _build_text_chunks(blocks, meta)
    fig_chunks = _build_figure_caption_chunks(blocks, meta)
    tab_chunks = _build_table_caption_chunks(blocks, meta)
    eq_chunks = _build_equation_context_chunks(blocks, meta)

    main_chunks = text_chunks + fig_chunks + tab_chunks
    logger.info(
        "%s → %d text, %d fig, %d tab, %d eq chunks",
        meta.paper_id, len(text_chunks), len(fig_chunks),
        len(tab_chunks), len(eq_chunks),
    )
    return main_chunks, eq_chunks
