"""
In-memory ChromaDB vector store for the current repo session.

One instance per app session. Call build() after scraping + chunking.
Call clear() when the user loads a new repo.
"""

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from rag.embedder import get_embeddings


class CodeVectorStore:
    def __init__(self):
        self._store: Chroma | None = None

    # ── Public API ────────────────────────────────────────────

    def build(self, chunks: list[dict]) -> int:
        """
        Embed all chunks and load them into an in-memory ChromaDB collection.
        Replaces any previously loaded repo.

        Args:
            chunks: List of {content, metadata} dicts from chunk_repo()

        Returns:
            Number of chunks successfully loaded.
        """
        self.clear()

        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # EphemeralClient = in-memory, nothing written to disk
        client = chromadb.EphemeralClient()

        self._store = Chroma.from_texts(
            texts=texts,
            embedding=get_embeddings(),
            metadatas=metadatas,
            client=client,
            collection_name="codelens_session",
        )

        return len(texts)

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search: return the top-k most relevant chunks for a question.

        Returns list of {content, metadata} dicts, ordered by relevance.
        """
        self._assert_ready()
        docs = self._store.similarity_search(question, k=top_k)
        return [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]

    def as_retriever(self, top_k: int = 5) -> VectorStoreRetriever:
        """
        Return a LangChain retriever for use in the ConversationalRetrievalChain.
        This is the interface Phase 4 (RAG pipeline) uses.
        """
        self._assert_ready()
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    def clear(self) -> None:
        """Wipe the current session's vector store. Safe to call when nothing is loaded."""
        if self._store is not None:
            self._store.delete_collection()
            self._store = None

    @property
    def is_ready(self) -> bool:
        """True once build() has been called successfully."""
        return self._store is not None

    # ── Internal ──────────────────────────────────────────────

    def _assert_ready(self) -> None:
        if self._store is None:
            raise RuntimeError("Vector store not built. Call build() first.")
