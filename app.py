"""Streamlit UI for the JOF Structured Literature Agent."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from openai import OpenAI

import config
from retrieval.query_parser import parse_query, QueryCategory
from retrieval.retriever import Retriever
from retrieval.verifier import verify_dataset_usage
from agent.answer_generator import generate_answer_stream


@st.cache_resource
def get_retriever() -> Retriever:
    """Load FAISS indexes once and reuse across queries."""
    return Retriever()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JOF Literature Agent",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("JOF Literature Agent")
    st.markdown(
        "A structured literature research assistant built on **772 Journal of Finance** "
        "papers (2016–2026)."
    )
    st.divider()

    if config.PAPER_METADATA_PATH.exists():
        meta_list = json.loads(config.PAPER_METADATA_PATH.read_text("utf-8"))
        st.metric("Total Papers", len(meta_list))
    else:
        st.warning("Index not built yet. Run `python scripts/build_index.py` first.")
        meta_list = []

    if config.MAIN_INDEX_PATH.exists():
        main_chunks = json.loads(config.MAIN_CHUNKS_PATH.read_text("utf-8"))
        st.metric("Main Index Chunks", len(main_chunks))
    else:
        main_chunks = []

    if config.EQUATION_INDEX_PATH.exists():
        eq_chunks = json.loads(config.EQUATION_CHUNKS_PATH.read_text("utf-8"))
        st.metric("Equation Index Chunks", len(eq_chunks))
    else:
        eq_chunks = []

    st.divider()
    st.caption("Supported Query Types")
    st.markdown("""
- **Dataset Usage** — Which papers use a specific dataset?
- **Topic Search** — Papers on a research topic
- **Paper Fact** — DOI / abstract / data of a paper
- **Figure / Table** — What does a figure or table show?
- **Equation / Model** — Key equations or model mechanisms
    """)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.header("Search the JOF Literature")

query = st.text_input(
    "Enter your research question",
    placeholder="e.g., Which JOF papers use OptionMetrics?",
)

if st.button("Search", type="primary", disabled=not query):
    if not config.MAIN_INDEX_PATH.exists():
        st.error("Index files not found. Please run `python scripts/build_index.py` first.")
        st.stop()

    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    with st.spinner("Classifying query …"):
        parsed = parse_query(query, client=client)
    st.info(
        f"Query type: **{parsed.category.value}**  |  "
        f"Expanded terms: {', '.join(parsed.expanded_terms) or '—'}"
    )

    with st.spinner("Retrieving relevant papers …"):
        retriever = get_retriever()
        results = retriever.search(parsed)

    if parsed.category == QueryCategory.DATASET_USAGE and results:
        with st.spinner("Verifying dataset usage …"):
            results = verify_dataset_usage(results, parsed.expanded_terms, client=client)

    # Streaming answer generation
    st.subheader("Answer")
    st.write_stream(generate_answer_stream(query, results, client=client))

    # Evidence cards
    if results:
        st.subheader("Evidence Details")
        for pr in results[:8]:
            with st.expander(
                f"📄 {pr.title or pr.paper_id}  (score: {pr.best_score:.3f})"
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if pr.doi:
                        st.markdown(f"**DOI:** [{pr.doi}](https://doi.org/{pr.doi})")
                    else:
                        st.markdown("**DOI:** N/A")
                with col2:
                    st.markdown(f"**Paper ID:** `{pr.paper_id}`")

                for ec in pr.evidence_chunks:
                    chunk = ec.chunk
                    ct = chunk.get("chunk_type", "text")
                    page = chunk.get("page_idx") or chunk.get("page_start") or "?"

                    badge_colors = {
                        "text": "blue",
                        "figure_caption": "green",
                        "table_caption": "orange",
                        "equation_context": "red",
                    }
                    badge = badge_colors.get(ct, "gray")
                    st.markdown(f":{badge}[{ct}] &nbsp; page {page} &nbsp; score {ec.score:.3f}")

                    if ct == "equation_context":
                        eq_tex = chunk.get("equation_text", "")
                        if eq_tex:
                            st.latex(eq_tex.replace("$$", "").strip())
                        ctx = chunk.get("context_before", "")
                        if ctx:
                            st.caption(f"Context: {ctx[:300]}")
                    elif ct in ("figure_caption", "table_caption"):
                        st.info(chunk.get("caption_text", chunk.get("text", "")))
                        if chunk.get("img_path"):
                            img_full = config.JSON_BASE_DIR / pr.paper_id / chunk["img_path"]
                            if img_full.exists():
                                st.image(str(img_full), width=400)
                    else:
                        st.markdown(chunk.get("text", "")[:800])

                    st.divider()
