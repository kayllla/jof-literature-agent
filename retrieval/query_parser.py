"""Classify user queries and expand dataset aliases."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


class QueryCategory(str, Enum):
    DATASET_USAGE = "dataset_usage"
    TOPIC_SEARCH = "topic_search"
    PAPER_FACT = "paper_fact"
    FIGURE_TABLE = "figure_table"
    EQUATION_MODEL = "equation_model"


@dataclass
class ParsedQuery:
    category: QueryCategory
    original_query: str
    expanded_terms: list[str]
    search_query: str  # the query to use for vector search


DATASET_ALIASES: dict[str, list[str]] = {
    "OptionMetrics": ["OptionMetrics", "option metrics", "IvyDB"],
    "CRSP": ["CRSP", "Center for Research in Security Prices"],
    "Compustat": ["Compustat", "COMPUSTAT"],
    "TAQ": ["TAQ", "Trade and Quote"],
    "BoardEx": ["BoardEx", "board ex"],
    "IBES": ["IBES", "I/B/E/S", "Institutional Brokers Estimate System"],
    "Thomson Reuters": ["Thomson Reuters", "SDC", "SDC Platinum"],
    "Bloomberg": ["Bloomberg"],
    "TRACE": ["TRACE", "Trade Reporting and Compliance Engine"],
    "ExecuComp": ["ExecuComp", "Execucomp"],
    "FRED": ["FRED", "Federal Reserve Economic Data"],
    "FlowOfFunds": ["Flow of Funds", "Financial Accounts"],
    "HMDA": ["HMDA", "Home Mortgage Disclosure Act"],
    "SEC EDGAR": ["SEC EDGAR", "EDGAR"],
    "13F": ["13F", "13-F", "Form 13F"],
    "WRDS": ["WRDS", "Wharton Research Data Services"],
}

_CLASSIFY_PROMPT = """You are a query classifier for a financial literature search system covering Journal of Finance papers.

Classify the user query into exactly ONE category:
- dataset_usage: asking which papers use a specific dataset or data source
- topic_search: asking about a research topic, theme, or finding papers on a subject
- paper_fact: asking for facts about a specific named paper (its DOI, abstract, data, conclusions)
- figure_table: asking about figures, tables, charts, or visual results in papers
- equation_model: asking about mathematical models, equations, lemmas, propositions, proofs

Respond in JSON: {{"category": "...", "search_terms": ["term1", "term2"]}}
where search_terms are the key concepts to search for.

User query: {query}"""


def parse_query(query: str, client: OpenAI | None = None) -> ParsedQuery:
    """Classify a query and expand dataset aliases."""
    if client is None:
        client = OpenAI(api_key=config.OPENAI_API_KEY)

    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL_MINI,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(query=query)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        category = QueryCategory(result.get("category", "topic_search"))
        search_terms = result.get("search_terms", [])
    except Exception as exc:
        logger.warning("LLM classification failed, falling back to topic_search: %s", exc)
        category = QueryCategory.TOPIC_SEARCH
        search_terms = []

    # Expand dataset aliases
    expanded: list[str] = list(search_terms)
    if category == QueryCategory.DATASET_USAGE:
        for term in search_terms:
            term_upper = term.strip()
            for canonical, aliases in DATASET_ALIASES.items():
                if term_upper.lower() in [a.lower() for a in aliases] or term_upper.lower() == canonical.lower():
                    for alias in aliases:
                        if alias not in expanded:
                            expanded.append(alias)

    search_query = query
    if expanded and category == QueryCategory.DATASET_USAGE:
        search_query = f"{query} ({', '.join(expanded)})"

    return ParsedQuery(
        category=category,
        original_query=query,
        expanded_terms=expanded,
        search_query=search_query,
    )
