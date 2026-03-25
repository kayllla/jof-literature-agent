import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT.parent / ".env")
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

JSON_BASE_DIR = PROJECT_ROOT.parent / "jof_2016-2026_mineru_extrac"


def _get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then fall back to env vars."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
LLM_MODEL = "gpt-4o-mini"
LLM_MODEL_MINI = "gpt-4o-mini"

MAIN_INDEX_PATH = DATA_DIR / "main_index.faiss"
EQUATION_INDEX_PATH = DATA_DIR / "equation_index.faiss"
MAIN_CHUNKS_PATH = DATA_DIR / "main_chunks.json"
EQUATION_CHUNKS_PATH = DATA_DIR / "equation_chunks.json"
PAPER_METADATA_PATH = DATA_DIR / "paper_metadata.json"

TEXT_CHUNK_MAX_CHARS = 1500

RETRIEVAL_TOP_K = 20
