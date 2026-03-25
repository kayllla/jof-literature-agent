"""Generate grounded answers from retrieved paper evidence."""
from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

import config
from retrieval.retriever import PaperResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a financial literature research assistant. You answer questions based ONLY on the evidence retrieved from Journal of Finance (JOF) papers published between 2016 and 2026.

Rules:
1. Organize your answer by paper — each relevant paper gets its own section.
2. For each paper, provide:
   - Paper title
   - DOI (format as a clickable link: https://doi.org/DOI)
   - A brief explanation of why this paper is relevant
   - A key evidence passage (quote or close paraphrase)
   - Source type (text / figure_caption / table_caption / equation_context) and page number
3. If the evidence comes from a figure or table caption, present the full caption.
4. If the evidence comes from an equation context, present the equation in LaTeX and its surrounding context.
5. If no relevant papers are found, say so honestly — do not fabricate.
6. Be concise but thorough. Prefer depth over breadth.
7. Answer in the same language as the user's question."""

_MAX_EVIDENCE_CHARS = 1200


def generate_answer(
    query: str,
    paper_results: list[PaperResult],
    client: OpenAI | None = None,
) -> str:
    """Build a context from retrieved results and generate an LLM answer."""
    if client is None:
        client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    if not paper_results:
        return "No relevant papers found in the JOF corpus for this query."

    context = _build_context(paper_results)

    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=_make_messages(query, context),
        temperature=0.2,
        max_tokens=3000,
    )
    return resp.choices[0].message.content


def generate_answer_stream(
    query: str,
    paper_results: list[PaperResult],
    client: OpenAI | None = None,
):
    """Streaming variant — yields text chunks as they arrive."""
    if client is None:
        client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    if not paper_results:
        yield "No relevant papers found in the JOF corpus for this query."
        return

    context = _build_context(paper_results)

    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=_make_messages(query, context),
        temperature=0.2,
        max_tokens=3000,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _make_messages(query: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nRetrieved evidence:\n{context}"},
    ]


def _build_context(results: list[PaperResult]) -> str:
    parts: list[str] = []
    for i, pr in enumerate(results[:8], 1):
        section = f"--- Paper {i} ---\n"
        section += f"Title: {pr.title or 'N/A'}\n"
        section += f"DOI: {pr.doi or 'N/A'}\n"
        section += f"Paper ID: {pr.paper_id}\n"

        for j, ec in enumerate(pr.evidence_chunks[:2], 1):
            chunk = ec.chunk
            ct = chunk.get("chunk_type", "text")
            page = chunk.get("page_idx") or chunk.get("page_start") or "?"
            text = chunk.get("text", "")[:_MAX_EVIDENCE_CHARS]

            section += f"\nEvidence {j} [{ct}, page {page}]:\n{text}\n"

            if ct == "equation_context" and chunk.get("equation_text"):
                section += f"Equation: {chunk['equation_text']}\n"
            if ct in ("figure_caption", "table_caption") and chunk.get("caption_text"):
                section += f"Caption: {chunk['caption_text']}\n"

        parts.append(section)
    return "\n".join(parts)
