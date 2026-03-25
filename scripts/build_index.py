#!/usr/bin/env python3
"""One-shot script: load JSONs → extract metadata → build chunks → embed → write FAISS indexes."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

import config
from ingest.json_loader import load_all_papers
from ingest.metadata_extractor import extract_metadata, PaperMetadata
from ingest.chunk_builder import build_chunks, Chunk
from index.index_builder import build_and_save, save_paper_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indexes for JOF literature agent")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N papers (for debugging)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override JSON base directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base_dir = Path(args.data_dir) if args.data_dir else config.JSON_BASE_DIR
    papers = load_all_papers(base_dir, limit=args.limit)
    if not papers:
        logging.error("No papers found in %s", base_dir)
        sys.exit(1)

    all_main: list[Chunk] = []
    all_eq: list[Chunk] = []
    all_meta: list[dict] = []

    for paper in tqdm(papers, desc="Processing papers"):
        meta = extract_metadata(paper.paper_id, paper.blocks, str(paper.source_path))
        all_meta.append({
            "paper_id": meta.paper_id,
            "title": meta.title,
            "authors": meta.authors,
            "abstract": meta.abstract,
            "doi": meta.doi,
            "source_path": meta.source_path,
        })
        main_chunks, eq_chunks = build_chunks(paper.blocks, meta)
        all_main.extend(main_chunks)
        all_eq.extend(eq_chunks)

    logging.info(
        "Total: %d main chunks, %d equation chunks from %d papers",
        len(all_main), len(all_eq), len(papers),
    )

    save_paper_metadata(all_meta)
    build_and_save(all_main, all_eq)
    logging.info("Done.")


if __name__ == "__main__":
    main()
