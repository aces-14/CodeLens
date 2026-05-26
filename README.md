---
title: CodeLens
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
---

# CodeLens

**Understand any public GitHub repository — instantly.**

Paste a GitHub URL and chat with the codebase in natural language. Ask questions like a senior engineer who's already read the whole thing. Every answer cites the exact source files it came from.

---

## What It Does

- Paste any public GitHub repo URL
- CodeLens fetches, chunks, and embeds the codebase on the fly
- Ask questions in plain English — "How does authentication work?", "Where is rate limiting implemented?", "What does the DataPipeline class do?"
- Answers include citations back to the actual source files
- Follow-up questions work — context is maintained across the conversation

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq — Llama 3.3 70B |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) |
| Vector store | ChromaDB (in-memory, session-scoped) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` with language-aware separators |
| GitHub ingestion | GitHub REST API (recursive tree endpoint) |
| UI | Gradio 6.x |

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/CodeLens.git
cd CodeLens
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

**3. Set up environment variables**

Copy `.env.example` to `.env` and fill in your keys:
```
GROQ_API_KEY=your_groq_api_key_here
GITHUB_TOKEN=your_github_token_here   # optional but recommended
```

- **GROQ_API_KEY** — get a free key at [console.groq.com](https://console.groq.com)
- **GITHUB_TOKEN** — optional. Without it, GitHub API is limited to 60 requests/hour. With it, the limit is 5,000/hour. [Create a token here](https://github.com/settings/tokens) with `repo` (read-only) scope.

**4. Run**
```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## How It Works

```
User pastes GitHub URL
        ↓
GitHub REST API → file tree (one recursive call) → filter by extension/size
        ↓
LangChain code-aware chunker → splits by functions/classes for code, headings for markdown
        ↓
sentence-transformers → embeds chunks → ChromaDB in-memory store
        ↓
User asks a question
        ↓
Semantic search → top-5 relevant chunks
        ↓
Groq LLM (Llama 3.3 70B) → answer with source citations
```

Multi-turn conversation is supported — follow-up questions are reformulated into standalone queries before retrieval so pronouns and references resolve correctly.

---

## Limitations

- **Public repos only** — private repos require OAuth, which is out of scope for v1
- **500-file cap** — repos larger than 500 supported files are rejected to keep ingestion fast
- **Session-scoped** — embeddings are in-memory and wiped when you load a new repo. Nothing is cached between sessions.
- **English-language questions** — the LLM handles other languages but the system prompt is in English

---

## Project Structure

```
CodeLens/
├── app.py                  # Gradio UI — landing page + chat interface
├── ingestion/
│   ├── github_scraper.py   # GitHub REST API client, file tree fetcher
│   ├── file_filter.py      # Extension allowlist, directory denylist, size cap
│   └── chunker.py          # Language-aware document chunker
├── rag/
│   ├── pipeline.py         # RAG pipeline — retrieval + Groq LLM + chat memory
│   ├── vector_store.py     # ChromaDB in-memory vector store
│   └── embedder.py         # HuggingFace sentence-transformers embedder
├── utils/
│   └── token_counter.py    # tiktoken-based token budget checker
├── requirements.txt
└── .env.example
```
