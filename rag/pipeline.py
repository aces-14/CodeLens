"""
RAG pipeline: wires ChromaDB retrieval to the Groq LLM.

Uses LangChain 1.x compatible approach — manual chat history management
instead of the deprecated ConversationalRetrievalChain / ConversationBufferMemory.

Flow per ask():
  1. If there is chat history, use LLM to reformulate the follow-up question
     into a standalone question (so retrieval works correctly)
  2. Retrieve top-k chunks using the standalone question
  3. Build a message list: system prompt + history + current question
  4. Call the LLM and return {answer, sources}
"""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from rag.vector_store import CodeVectorStore

_GROQ_MODEL = "llama-3.3-70b-versatile"
_MAX_HISTORY_TURNS = 5  # number of past (question, answer) pairs to retain

_QA_SYSTEM = """\
You are a senior software engineer helping a developer understand a codebase.
Use only the code and documentation excerpts provided below to answer the question.

Rules:
- Cite which file(s) your answer comes from, e.g. "(from `src/engine.py`)".
- If the answer is not in the excerpts, say "I couldn't find that in the ingested files." Do not guess or use outside knowledge.
- Be concise and technical. Write for a developer audience.

Code excerpts:
{context}"""

_CONDENSE_TEMPLATE = """\
Given the following conversation history and a follow-up question, \
rewrite the follow-up question as a standalone question that can be understood \
without the conversation history. Output only the standalone question, nothing else.

Conversation history:
{history}

Follow-up question: {question}
Standalone question:"""


class CodeLensPipeline:
    def __init__(self, vector_store: CodeVectorStore):
        """
        Build the RAG pipeline for a loaded repo session.

        Args:
            vector_store: A CodeVectorStore that has already had build() called.

        Raises:
            RuntimeError: If GROQ_API_KEY is not set in the environment.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and add it to your .env file."
            )

        self._llm = ChatGroq(
            model=_GROQ_MODEL,
            temperature=0.1,
            api_key=api_key,
            timeout=30,
        )
        self._retriever = vector_store.as_retriever(top_k=5)

        # Chat history: list of (human_question, ai_answer) tuples, capped at _MAX_HISTORY_TURNS
        self._history: list[tuple[str, str]] = []

    # ── Public API ────────────────────────────────────────────

    def ask(self, question: str) -> dict:
        """
        Ask a natural language question about the loaded codebase.

        Returns:
            {
                "answer":  str       — LLM response with inline source citations,
                "sources": list[str] — deduplicated file paths from retrieved chunks
            }
        """
        # Step 1: Reformulate follow-up questions so retrieval works correctly
        retrieval_question = self._make_standalone_question(question)

        # Step 2: Retrieve relevant chunks
        docs = self._retriever.invoke(retrieval_question)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        sources = sorted({
            doc.metadata.get("file_path", "unknown") for doc in docs
        })

        # Step 3: Build message list with system prompt + history + new question
        messages = self._build_messages(question, context)

        # Step 4: Call LLM (retry once on transient failure)
        try:
            response = self._llm.invoke(messages)
        except Exception:
            try:
                response = self._llm.invoke(messages)
            except Exception as e:
                raise RuntimeError(
                    f"LLM is unavailable — please try again. ({type(e).__name__})"
                ) from e
        answer = response.content

        # Step 5: Store turn in history, enforce window size
        self._history.append((question, answer))
        if len(self._history) > _MAX_HISTORY_TURNS:
            self._history.pop(0)

        return {"answer": answer, "sources": sources}

    def clear_memory(self) -> None:
        """Clear conversation history. Call when the user loads a new repo."""
        self._history.clear()

    # ── Internal ──────────────────────────────────────────────

    def _make_standalone_question(self, question: str) -> str:
        """
        If there is chat history, use the LLM to reformulate the question
        into a standalone version that retrieval can use effectively.
        If no history, the question is returned as-is.
        """
        if not self._history:
            return question

        history_text = "\n".join(
            f"Human: {h}\nAssistant: {a}" for h, a in self._history
        )
        prompt = _CONDENSE_TEMPLATE.format(
            history=history_text,
            question=question,
        )
        response = self._llm.invoke(prompt)
        return response.content.strip()

    def _build_messages(self, question: str, context: str) -> list:
        """
        Build the full message list for the LLM:
          [SystemMessage] + interleaved history + [HumanMessage (current question)]
        """
        messages = [SystemMessage(content=_QA_SYSTEM.format(context=context))]

        for human_q, ai_a in self._history:
            messages.append(HumanMessage(content=human_q))
            messages.append(AIMessage(content=ai_a))

        messages.append(HumanMessage(content=question))
        return messages
