"""Second-pass verifier: for dataset_usage queries, distinguish 'used' vs 'mentioned'."""
from __future__ import annotations

import json
import logging
from typing import Optional

from openai import OpenAI

import config
from retrieval.retriever import PaperResult

logger = logging.getLogger(__name__)

_VERIFY_PROMPT = """You are verifying whether a research paper actually USES a specific dataset or merely MENTIONS it.

Dataset in question: {dataset}

Evidence text from the paper "{title}":
---
{evidence}
---

Based on the evidence, classify as one of:
- "used": the paper clearly uses this dataset for its analysis / empirical work
- "mentioned": the paper only mentions or cites this dataset (e.g., in a literature review)
- "unclear": cannot determine from this evidence

Respond in JSON: {{"verdict": "used" | "mentioned" | "unclear"}}"""


def verify_dataset_usage(
    results: list[PaperResult],
    dataset_terms: list[str],
    client: OpenAI | None = None,
) -> list[PaperResult]:
    """For each paper result, ask the LLM whether the dataset was actually used.

    Papers classified as 'mentioned' or 'unclear' are moved to the end
    with a penalty on their score.
    """
    if client is None:
        client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    dataset_str = ", ".join(dataset_terms[:3])
    verified: list[PaperResult] = []

    for pr in results:
        evidence = pr.evidence_chunks[0].chunk.get("text", "")[:1500] if pr.evidence_chunks else ""
        verdict = _call_verify(client, dataset_str, pr.title or pr.paper_id, evidence)
        if verdict == "used":
            verified.append(pr)
        elif verdict == "mentioned":
            pr.best_score *= 0.3
            verified.append(pr)
        else:
            pr.best_score *= 0.1
            verified.append(pr)

    verified.sort(key=lambda p: p.best_score, reverse=True)
    return verified


def _call_verify(client: OpenAI, dataset: str, title: str, evidence: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL_MINI,
            messages=[{
                "role": "user",
                "content": _VERIFY_PROMPT.format(dataset=dataset, title=title, evidence=evidence),
            }],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        return result.get("verdict", "unclear")
    except Exception as exc:
        logger.warning("Verification LLM call failed: %s", exc)
        return "unclear"
