"""Recursively scan and load all *_content.json files from the JOF corpus."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RawPaper:
    paper_id: str
    source_path: Path
    blocks: list = field(repr=False)


def load_all_papers(base_dir: Path, limit: int | None = None) -> list[RawPaper]:
    """Scan *base_dir* for paper folders and load their _content.json files.

    Args:
        base_dir: Root directory containing paper folders (e.g. ``10_1111_jofi_*``).
        limit: If set, only load the first *limit* papers (useful for debugging).

    Returns:
        A list of ``RawPaper`` objects sorted by paper_id.
    """
    papers: list[RawPaper] = []
    if not base_dir.exists():
        logger.error("Base directory does not exist: %s", base_dir)
        return papers

    folders = sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("10_1111_jofi_")
    )

    if limit is not None:
        folders = folders[:limit]

    for folder in folders:
        json_file = folder / f"{folder.name}_content.json"
        if not json_file.exists():
            logger.warning("Missing content JSON in %s", folder)
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                blocks = json.load(f)
            papers.append(RawPaper(
                paper_id=folder.name,
                source_path=json_file,
                blocks=blocks,
            ))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", json_file, exc)

    logger.info("Loaded %d papers from %s", len(papers), base_dir)
    return papers
