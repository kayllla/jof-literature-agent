"""Build and persist FAISS indexes + chunk metadata."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

import config
from ingest.chunk_builder import Chunk
from index.embedder import embed_texts, get_client

logger = logging.getLogger(__name__)


def build_and_save(
    main_chunks: list[Chunk],
    equation_chunks: list[Chunk],
) -> None:
    """Embed all chunks, build FAISS indexes, and write everything to disk."""
    client = get_client()

    # --- Main index ---
    logger.info("Embedding %d main chunks …", len(main_chunks))
    main_texts = [c.text for c in main_chunks]
    main_vecs = embed_texts(main_texts, client=client)

    main_index = faiss.IndexFlatIP(main_vecs.shape[1])
    main_index.add(main_vecs)
    faiss.write_index(main_index, str(config.MAIN_INDEX_PATH))

    main_meta = [c.to_dict() for c in main_chunks]
    config.MAIN_CHUNKS_PATH.write_text(
        json.dumps(main_meta, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Main index saved: %d vectors → %s", main_index.ntotal, config.MAIN_INDEX_PATH
    )

    # --- Equation index ---
    if equation_chunks:
        logger.info("Embedding %d equation chunks …", len(equation_chunks))
        eq_texts = [c.text for c in equation_chunks]
        eq_vecs = embed_texts(eq_texts, client=client)

        eq_index = faiss.IndexFlatIP(eq_vecs.shape[1])
        eq_index.add(eq_vecs)
        faiss.write_index(eq_index, str(config.EQUATION_INDEX_PATH))

        eq_meta = [c.to_dict() for c in equation_chunks]
        config.EQUATION_CHUNKS_PATH.write_text(
            json.dumps(eq_meta, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Equation index saved: %d vectors → %s",
            eq_index.ntotal, config.EQUATION_INDEX_PATH,
        )


def save_paper_metadata(metadata_list: list[dict]) -> None:
    """Persist paper-level metadata to JSON."""
    config.PAPER_METADATA_PATH.write_text(
        json.dumps(metadata_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Paper metadata saved: %d papers → %s", len(metadata_list), config.PAPER_METADATA_PATH)
