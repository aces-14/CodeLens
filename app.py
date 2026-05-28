"""
CodeLens — Gradio UI entry point.

Two states: landing (URL input only) → chat (after repo loads).
Simple layout, minimal CSS — just colors and card width.
"""

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from ingestion.github_scraper import GitHubScraperError, scrape_repo
from ingestion.chunker import chunk_repo
from rag.pipeline import CodeLensPipeline
from rag.vector_store import CodeVectorStore

_store: CodeVectorStore | None = None
_pipeline: CodeLensPipeline | None = None


# ─────────────────────────────────────────────────────────────
CSS = """
/* Base */
html, body { background: #0d0d0d !important; color: #e0e0e0 !important; }
body, .gradio-container, .gradio-container > div,
.app, .main, .contain, .wrap, .gap, .block {
    background: #0d0d0d !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
.gradio-container { padding-top: 0 !important; }
footer { display: none !important; }

/* Scanline overlay */
body::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 3px,
        rgba(0,0,0,0.06) 3px,
        rgba(0,0,0,0.06) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Animations ── */
@keyframes glow-pulse {
    0%, 100% { text-shadow: 0 0 12px rgba(0,229,255,0.55), 0 0 28px rgba(0,229,255,0.2); }
    50% { text-shadow: 0 0 22px rgba(0,229,255,0.95), 0 0 55px rgba(0,229,255,0.4), 0 0 90px rgba(0,229,255,0.1); }
}
@keyframes blink-cursor {
    0%, 49% { opacity: 1; }
    50%, 100% { opacity: 0; }
}
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 3px #00e5ff; opacity: 0.7; }
    50% { box-shadow: 0 0 8px #00e5ff, 0 0 16px rgba(0,229,255,0.5); opacity: 1; }
}
@keyframes status-flicker {
    0%, 92%, 100% { opacity: 1; }
    95% { opacity: 0.35; }
    97% { opacity: 1; }
    99% { opacity: 0.5; }
}

/* ── Landing layout ── */
#spacer-l, #spacer-r { padding: 0 !important; min-height: 0 !important; }
#spacer-l *, #spacer-r * { visibility: hidden !important; }
#landing-card { padding-top: 12vh !important; padding-bottom: 4vh !important; }

/* ── Chat section ── */
#chat-col { padding-left: 4vw !important; padding-right: 4vw !important; }

/* URL input — terminal style */
#url-input textarea {
    background: #080808 !important;
    border: 1px solid rgba(0,229,255,0.38) !important;
    color: #00e5ff !important;
    border-radius: 3px !important;
    font-size: 0.9rem !important;
    font-family: 'Courier New', monospace !important;
    padding: 0.65rem 1rem !important;
    letter-spacing: 0.02em !important;
}
#url-input textarea::placeholder { color: #1a3535 !important; }
#url-input textarea:focus {
    box-shadow: 0 0 0 1px rgba(0,229,255,0.3), 0 0 25px rgba(0,229,255,0.12) !important;
    border-color: #00e5ff !important;
    outline: none !important;
}

/* Load button — terminal style */
#load-btn button {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    color: #00e5ff !important;
    border: 1px solid rgba(0,229,255,0.6) !important;
    border-radius: 3px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    font-family: 'Courier New', monospace !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
#load-btn button:hover {
    background: rgba(0,229,255,0.07) !important;
    background-color: rgba(0,229,255,0.07) !important;
    box-shadow: 0 0 22px rgba(0,229,255,0.28), inset 0 0 15px rgba(0,229,255,0.04) !important;
}

/* Status text */
#load-status p {
    color: rgba(0,229,255,0.8) !important;
    font-size: 0.76rem !important;
    text-align: center;
    font-family: 'Courier New', monospace !important;
    letter-spacing: 0.07em !important;
    animation: status-flicker 5s infinite !important;
}

/* ── Chat section ── */
#chat-header {
    background: #080808 !important;
    border-bottom: 1px solid rgba(0,229,255,0.1) !important;
    padding: 0.5rem 1rem !important;
}
#repo-info p, #repo-info * {
    color: #555 !important; font-size: 0.8rem !important;
    margin: 0 !important; font-family: 'Courier New', monospace !important;
}

/* Chatbot bubbles */
.message.bot {
    background: #0c0c0c !important; border: 1px solid rgba(0,229,255,0.08) !important;
    border-radius: 3px !important; color: #d0d0d0 !important;
}
.message.user {
    background: #001419 !important; border: 1px solid rgba(0,229,255,0.18) !important;
    border-radius: 3px !important; color: #e0e0e0 !important;
}

/* Message input */
#msg-input textarea {
    background: #080808 !important; border: 1px solid #181818 !important;
    color: #e0e0e0 !important; border-radius: 3px !important;
    font-size: 0.9rem !important;
}
#msg-input textarea:focus {
    border-color: rgba(0,229,255,0.45) !important;
    box-shadow: 0 0 0 1px rgba(0,229,255,0.15), 0 0 15px rgba(0,229,255,0.07) !important;
}
#msg-input textarea::placeholder { color: #282828 !important; }

/* Send button */
#send-btn button {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    color: #00e5ff !important;
    border: 1px solid rgba(0,229,255,0.48) !important;
    border-radius: 3px !important;
    font-weight: 600 !important;
    font-family: 'Courier New', monospace !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-size: 0.78rem !important;
}
#send-btn button:hover {
    background: rgba(0,229,255,0.07) !important;
    background-color: rgba(0,229,255,0.07) !important;
    box-shadow: 0 0 14px rgba(0,229,255,0.28) !important;
}
#send-btn button:disabled {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    color: #152020 !important;
    border-color: #152020 !important;
}

/* Secondary buttons */
#clear-chat-btn button, #new-repo-btn button {
    background: transparent !important; border-radius: 3px !important;
    font-size: 0.73rem !important;
    font-family: 'Courier New', monospace !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    transition: all 0.15s !important;
}
#clear-chat-btn button { border: 1px solid #1c1c1c !important; color: #444 !important; }
#clear-chat-btn button:hover { border-color: #383838 !important; color: #888 !important; }
#new-repo-btn button { border: 1px solid rgba(0,229,255,0.25) !important; color: rgba(0,229,255,0.7) !important; }
#new-repo-btn button:hover {
    background: rgba(0,229,255,0.05) !important;
    box-shadow: 0 0 8px rgba(0,229,255,0.18) !important;
}

/* Chatbot container */
#chatbot { border: 1px solid rgba(0,229,255,0.08) !important; border-radius: 3px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,229,255,0.25); }

input, textarea { background: #080808 !important; color: #e0e0e0 !important; }
"""

LOGO_HTML = """
<div style="position:relative; margin: 0 0 1.5rem 0;">

  <!-- Tech frame with grid background -->
  <div style="
    position: relative;
    background:
      linear-gradient(rgba(0,229,255,0.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,255,0.035) 1px, transparent 1px),
      #080808;
    background-size: 30px 30px, 30px 30px;
    border: 1px solid rgba(0,229,255,0.22);
    border-radius: 2px;
    padding: 1.8rem 2rem 1.5rem 2rem;
  ">

    <!-- Corner brackets -->
    <div style="position:absolute;top:-2px;left:-2px;width:16px;height:16px;border-top:2px solid #00e5ff;border-left:2px solid #00e5ff;"></div>
    <div style="position:absolute;top:-2px;right:-2px;width:16px;height:16px;border-top:2px solid #00e5ff;border-right:2px solid #00e5ff;"></div>
    <div style="position:absolute;bottom:-2px;left:-2px;width:16px;height:16px;border-bottom:2px solid #00e5ff;border-left:2px solid #00e5ff;"></div>
    <div style="position:absolute;bottom:-2px;right:-2px;width:16px;height:16px;border-bottom:2px solid #00e5ff;border-right:2px solid #00e5ff;"></div>

    <!-- System status bar -->
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1.6rem;
                font-family:'Courier New',monospace;font-size:0.62rem;
                letter-spacing:0.13em;color:rgba(0,229,255,0.45);text-transform:uppercase;">
      <div style="width:5px;height:5px;border-radius:50%;background:#00e5ff;
                  animation:pulse-dot 2.2s ease-in-out infinite;flex-shrink:0;"></div>
      <span>SYS ONLINE</span>
      <span style="color:#222;">│</span>
      <span>RAG READY</span>
      <span style="color:#222;">│</span>
      <span>LLM CONNECTED</span>
      <span style="margin-left:auto;color:#1e3535;">v1.0.0</span>
    </div>

    <!-- Logo -->
    <div style="font-size:3.8rem;font-weight:900;letter-spacing:-2px;color:#00e5ff;
                line-height:1;font-family:'Courier New',monospace;
                animation:glow-pulse 3s ease-in-out infinite;">
      CodeLens
    </div>

    <!-- Tagline -->
    <div style="font-size:0.9rem;color:#888;margin-top:0.55rem;letter-spacing:0.01em;">
      Understand any public GitHub repository — instantly.
    </div>

    <!-- Divider -->
    <div style="margin:1rem 0;height:1px;
                background:linear-gradient(90deg,rgba(0,229,255,0.45) 0%,rgba(0,229,255,0.15) 55%,transparent 100%);"></div>

    <!-- Terminal description -->
    <div style="font-family:'Courier New',monospace;font-size:0.77rem;color:#3a5a5a;line-height:1.95;">
      <div><span style="color:rgba(0,229,255,0.35);user-select:none;">›&nbsp;</span>Paste a repo URL → query it like a senior engineer who already read the whole codebase.</div>
      <div><span style="color:rgba(0,229,255,0.35);user-select:none;">›&nbsp;</span>Every answer cites the exact source files it came from.<span style="color:#00e5ff;animation:blink-cursor 1s step-end infinite;margin-left:2px;">▌</span></div>
    </div>

    <!-- Feature tags -->
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:1.15rem;">
      <span style="font-family:'Courier New',monospace;font-size:0.59rem;letter-spacing:0.13em;
                   color:rgba(0,229,255,0.45);background:rgba(0,229,255,0.04);
                   border:1px solid rgba(0,229,255,0.12);padding:3px 10px;border-radius:2px;">VECTOR SEARCH</span>
      <span style="font-family:'Courier New',monospace;font-size:0.59rem;letter-spacing:0.13em;
                   color:rgba(0,229,255,0.45);background:rgba(0,229,255,0.04);
                   border:1px solid rgba(0,229,255,0.12);padding:3px 10px;border-radius:2px;">RAG PIPELINE</span>
      <span style="font-family:'Courier New',monospace;font-size:0.59rem;letter-spacing:0.13em;
                   color:rgba(0,229,255,0.45);background:rgba(0,229,255,0.04);
                   border:1px solid rgba(0,229,255,0.12);padding:3px 10px;border-radius:2px;">MULTI-LANG</span>
      <span style="font-family:'Courier New',monospace;font-size:0.59rem;letter-spacing:0.13em;
                   color:rgba(0,229,255,0.45);background:rgba(0,229,255,0.04);
                   border:1px solid rgba(0,229,255,0.12);padding:3px 10px;border-radius:2px;">CITED SOURCES</span>
    </div>

  </div>
</div>
"""

_DEFAULT_PLACEHOLDER = (
    "**Ask anything about the codebase.**\n\n"
    "Examples:\n"
    "- How does authentication work in this repo?\n"
    "- What does the main class do?\n"
    "- Where is error handling implemented?"
)


# ─────────────────────────────────────────────────────────────
def _make_example_questions(result: dict) -> str:
    repo  = result["repo_name"]
    langs = {l.lower() for l in result["languages"]}
    name  = repo.lower()

    questions = [
        f"What is the overall architecture of `{repo}`?",
        "What are the main classes or modules?",
    ]

    if any(k in name for k in {"gpt","llm","bert","transformer","model","neural","train","diffusion","clip","vit"}):
        questions += ["How is the model architecture defined?", "How does the training loop work?", "How is data loaded?"]
    elif any(k in name for k in {"api","flask","fastapi","django","server","http","rest","graphql"}):
        questions += ["How is routing implemented?", "How are requests handled?", "How is auth set up?"]
    elif any(k in name for k in {"data","pipeline","etl","spark","pandas","stream","ingest"}):
        questions += ["How is data ingested and transformed?", "What does the main pipeline do?"]
    elif any(k in name for k in {"cli","tool","script","util","cmd","command"}):
        questions += ["How are CLI commands defined?", "What is the main entry point?"]
    else:
        questions += ["How does the core logic work?", "What is the main entry point?"]

    if "python" in langs:        questions.append("Where is error handling implemented?")
    if "typescript" in langs or "javascript" in langs: questions.append("How is state managed?")
    if "jupyter notebook" in langs: questions.append("What does the notebook demonstrate?")

    qs = "\n".join(f"- {q}" for q in questions[:6])
    return f"**`{repo}` loaded.** Here are some questions to start:\n\n{qs}"


# ─────────────────────────────────────────────────────────────
def load_repo(url: str):
    global _store, _pipeline
    _nc = gr.update()
    url = url.strip()

    if not url:
        yield gr.update(visible=True), gr.update(visible=False), \
              gr.update(value="Please enter a GitHub repository URL."), \
              _nc, _nc, gr.update(interactive=False), gr.update(interactive=False)
        return

    yield gr.update(visible=True), gr.update(visible=False), \
          gr.update(value="Fetching repository file tree..."), \
          _nc, _nc, gr.update(interactive=False), gr.update(interactive=False)

    try:
        result = scrape_repo(url)

        yield gr.update(visible=True), gr.update(visible=False), \
              gr.update(value=f"Chunking {result['files_ingested']} files..."), \
              _nc, _nc, gr.update(interactive=False), gr.update(interactive=False)

        chunks = chunk_repo(result["files"])

        yield gr.update(visible=True), gr.update(visible=False), \
              gr.update(value=f"Embedding {len(chunks)} chunks — may take ~30s on first run..."), \
              _nc, _nc, gr.update(interactive=False), gr.update(interactive=False)

        if _store: _store.clear()
        _store = CodeVectorStore()
        _store.build(chunks)
        _pipeline = CodeLensPipeline(_store)

        langs = ", ".join(result["languages"]) or "Unknown"
        info  = (f"**{result['owner']}/{result['repo_name']}** · "
                 f"branch `{result['branch']}` · "
                 f"{result['files_ingested']} files · {len(chunks)} chunks · {langs}")

        yield gr.update(visible=False), gr.update(visible=True), \
              gr.update(value=""), \
              gr.update(value=info, visible=True), \
              gr.update(placeholder=_make_example_questions(result)), \
              gr.update(interactive=True), gr.update(interactive=True)

    except GitHubScraperError as e:
        yield gr.update(visible=True), gr.update(visible=False), \
              gr.update(value=f"GitHub error: {e}"), \
              _nc, _nc, gr.update(interactive=False), gr.update(interactive=False)
    except RuntimeError as e:
        yield gr.update(visible=True), gr.update(visible=False), \
              gr.update(value=f"Configuration error: {e}"), \
              _nc, _nc, gr.update(interactive=False), gr.update(interactive=False)
    except Exception as e:
        yield gr.update(visible=True), gr.update(visible=False), \
              gr.update(value=f"Unexpected error ({type(e).__name__}): {e}"), \
              _nc, _nc, gr.update(interactive=False), gr.update(interactive=False)


def chat(message: str, history: list):
    global _pipeline
    if not message.strip(): return history, ""
    if not _pipeline:
        history.append({"role": "assistant",
                        "content": "No repository loaded. Click **← New Repo** to load one."})
        return history, ""
    history.append({"role": "user", "content": message})
    try:
        resp   = _pipeline.ask(message)
        answer = resp["answer"]
        if resp["sources"]:
            answer += "\n\n---\n**Sources:** " + " · ".join(f"`{s}`" for s in resp["sources"])
    except RuntimeError as e:
        answer = f"Error: {e}"
    except Exception as e:
        answer = f"Unexpected error ({type(e).__name__}): {e}"
    history.append({"role": "assistant", "content": answer})
    return history, ""


def clear_repo():
    global _store, _pipeline
    if _store: _store.clear()
    _store = _pipeline = None
    return (gr.update(visible=True), gr.update(visible=False),
            gr.update(value=""), gr.update(value=""),
            gr.update(value="", visible=False),
            gr.update(value=[], placeholder=_DEFAULT_PLACEHOLDER),
            gr.update(interactive=False), gr.update(interactive=False))


def clear_chat():
    global _pipeline
    if _pipeline: _pipeline.clear_memory()
    return []


# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="CodeLens") as demo:

    # ── Landing ───────────────────────────────────────────────
    with gr.Column(visible=True, elem_id="landing-outer") as landing:
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=0, elem_id="spacer-l"):
                gr.HTML("&nbsp;")
            with gr.Column(scale=2, min_width=320, elem_id="landing-card"):
                gr.HTML(LOGO_HTML)
                url_input = gr.Textbox(
                    label="", show_label=False,
                    placeholder="https://github.com/owner/repo",
                    lines=1, max_lines=1, elem_id="url-input",
                )
                load_btn = gr.Button("Load Repository", elem_id="load-btn")
                status   = gr.Markdown("", elem_id="load-status")
            with gr.Column(scale=1, min_width=0, elem_id="spacer-r"):
                gr.HTML("&nbsp;")

    # ── Chat ──────────────────────────────────────────────────
    with gr.Column(visible=False, elem_id="chat-col") as chat_section:

        with gr.Row(elem_id="chat-header"):
            repo_info      = gr.Markdown("", elem_id="repo-info", visible=False)
            clear_chat_btn = gr.Button("Clear Chat",  variant="secondary",
                                       size="sm", elem_id="clear-chat-btn", scale=0)
            new_repo_btn   = gr.Button("← New Repo", variant="secondary",
                                       size="sm", elem_id="new-repo-btn",  scale=0)

        chatbot = gr.Chatbot(
            height=420, show_label=False,
            placeholder=_DEFAULT_PLACEHOLDER,
            layout="bubble", elem_id="chatbot",
        )

        with gr.Row():
            msg_input = gr.Textbox(
                label="", show_label=False,
                placeholder="Ask a question about the codebase...",
                lines=1, scale=5, interactive=False, elem_id="msg-input",
            )
            send_btn = gr.Button("Send", scale=1,
                                 interactive=False, min_width=80, elem_id="send-btn")

    # ── Wiring ────────────────────────────────────────────────
    _out = [landing, chat_section, status, repo_info, chatbot, msg_input, send_btn]

    load_btn.click(fn=load_repo,   inputs=[url_input], outputs=_out, show_progress=False)
    url_input.submit(fn=load_repo, inputs=[url_input], outputs=_out, show_progress=False)

    send_btn.click(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    msg_input.submit(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])

    new_repo_btn.click(fn=clear_repo,
        outputs=[landing, chat_section, url_input, status, repo_info, chatbot, msg_input, send_btn])
    clear_chat_btn.click(fn=clear_chat, outputs=[chatbot])


if __name__ == "__main__":
    demo.launch(css=CSS)
