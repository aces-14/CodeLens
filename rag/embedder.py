"""
HuggingFace embedding model setup.

Uses all-MiniLM-L6-v2 — lightweight, runs on CPU, no API key required.
Model is downloaded on first use (~80MB) and cached by sentence-transformers.
"""

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level singleton — model load is expensive, only do it once per process
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding model instance, loading it on first call."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
