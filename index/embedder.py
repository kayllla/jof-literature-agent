"""Wrapper around OpenAI embedding API with batching and retry."""
from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
from openai import OpenAI, RateLimitError

import config

logger = logging.getLogger(__name__)

_BATCH_SIZE = 512
_MAX_RETRIES = 5
_RETRY_BASE_WAIT = 2.0


def get_client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


def embed_texts(
    texts: Sequence[str],
    model: str = config.EMBEDDING_MODEL,
    client: OpenAI | None = None,
) -> np.ndarray:
    """Embed a list of texts, returning an (N, dim) float32 numpy array.

    Handles batching and exponential-backoff retry on rate-limit errors.
    """
    if client is None:
        client = get_client()

    all_embeddings: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        emb = _embed_with_retry(client, batch, model)
        all_embeddings.extend(emb)
        if start + _BATCH_SIZE < total:
            logger.info("Embedded %d / %d", start + len(batch), total)

    arr = np.array(all_embeddings, dtype=np.float32)
    # L2-normalize so inner product == cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    arr = arr / norms
    return arr


def _embed_with_retry(
    client: OpenAI,
    texts: list[str],
    model: str,
) -> list[list[float]]:
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.embeddings.create(input=texts, model=model)
            return [d.embedding for d in resp.data]
        except RateLimitError:
            wait = _RETRY_BASE_WAIT * (2 ** attempt)
            logger.warning("Rate limited, waiting %.1fs (attempt %d)", wait, attempt + 1)
            time.sleep(wait)
    raise RuntimeError(f"Failed to embed after {_MAX_RETRIES} retries")
