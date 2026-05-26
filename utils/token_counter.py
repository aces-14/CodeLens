"""
Token counting utilities using tiktoken.
Used to enforce budgets before embedding and before sending context to the LLM.
"""

import tiktoken

# cl100k_base is compatible with most modern LLMs including Llama variants
_ENCODING_NAME = "cl100k_base"
_encoder = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoder


def count_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(_get_encoder().encode(text))


def count_chunks_tokens(chunks: list[dict]) -> int:
    """Return total token count across a list of chunk dicts."""
    return sum(count_tokens(chunk["content"]) for chunk in chunks)


def is_within_budget(chunks: list[dict], max_tokens: int = 200_000) -> tuple[bool, int]:
    """
    Check whether the total token count of all chunks fits within a budget.
    Returns (within_budget, total_tokens).
    """
    total = count_chunks_tokens(chunks)
    return total <= max_tokens, total
