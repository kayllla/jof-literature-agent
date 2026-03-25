"""Extract paper-level metadata (title, authors, abstract, DOI) from first-page blocks."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    paper_id: str
    title: str | None = None
    authors: str | None = None
    abstract: str | None = None
    doi: str | None = None
    source_path: str | None = None
    is_article: bool = True


_DOI_PATTERN = re.compile(r"(10\.1111/jofi\.\d+)", re.IGNORECASE)


def extract_metadata(paper_id: str, blocks: list[dict], source_path: str = "") -> PaperMetadata:
    """Heuristically extract metadata from the first-page blocks of a paper."""
    meta = PaperMetadata(paper_id=paper_id, source_path=source_path)

    first_page_blocks = [b for b in blocks if b.get("page_idx", -1) == 0 and b.get("type") == "text"]

    title_idx = -1
    abstract_idx = -1

    for i, block in enumerate(first_page_blocks):
        text = block.get("text", "").strip()
        if not text:
            continue

        if block.get("text_level") == 1 and text.upper() != "ABSTRACT" and meta.title is None:
            meta.title = text.rstrip()
            title_idx = i
            continue

        if text.upper().startswith("ABSTRACT"):
            abstract_idx = i
            continue

        m = _DOI_PATTERN.search(text)
        if m:
            meta.doi = m.group(1)

    # Authors: text blocks between title and ABSTRACT (or DOI block)
    if title_idx >= 0:
        end = abstract_idx if abstract_idx > title_idx else len(first_page_blocks)
        author_parts = []
        for block in first_page_blocks[title_idx + 1: end]:
            txt = block.get("text", "").strip()
            if _DOI_PATTERN.search(txt):
                break
            if txt.upper().startswith("ABSTRACT"):
                break
            if block.get("text_level") == 1:
                break
            author_parts.append(txt)
        if author_parts:
            meta.authors = " ".join(author_parts).strip()

    # Abstract: the text block right after ABSTRACT heading
    if abstract_idx >= 0 and abstract_idx + 1 < len(first_page_blocks):
        meta.abstract = first_page_blocks[abstract_idx + 1].get("text", "").strip()

    if meta.title is None:
        logger.warning("Could not extract title for %s", paper_id)
    if meta.doi is None:
        meta.doi = _doi_from_paper_id(paper_id)

    meta.is_article = _check_is_article(meta, blocks)
    return meta


def _check_is_article(meta: PaperMetadata, blocks: list[dict]) -> bool:
    """A real article must have an ABSTRACT; editorials, corrigenda, etc. do not."""
    if meta.abstract:
        return True
    for b in blocks:
        if b.get("type") == "text" and b.get("text", "").strip().upper().startswith("ABSTRACT"):
            return True
    return False


def _doi_from_paper_id(paper_id: str) -> str | None:
    """Fallback: derive DOI from folder name ``10_1111_jofi_XXXXX``."""
    parts = paper_id.split("_", 3)
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}/{parts[2]}.{parts[3]}"
    return None
