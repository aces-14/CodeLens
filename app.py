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

/* ── Landing layout ── */
/* Spacer columns are invisible; the center card takes 50% via scale */
#spacer-l, #spacer-r { padding: 0 !important; min-height: 0 !important; }
#spacer-l *, #spacer-r * { visibility: hidden !important; }
/* Center card vertical padding */
#landing-card { padding-top: 12vh !important; padding-bottom: 4vh !important; }

/* ── Chat section ── */
/* Constrain chat width so it feels focused, not full-screen */
#chat-col { padding-left: 4vw !important; padding-right: 4vw !important; }

/* URL input */
#url-input textarea {
    background: #141414 !important;
    border: 2px solid #00e5ff !important;
    color: #e0e0e0 !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    padding: 0.6rem 0.9rem !important;
}
#url-input textarea::placeholder { color: #444 !important; }
#url-input textarea:focus { box-shadow: 0 0 0 3px rgba(0,229,255,0.2) !important; }

/* Load button */
#load-btn button {
    background: #00e5ff !important;
    background-color: #00e5ff !important;
    background-image: none !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: box-shadow 0.2s !important;
}
#load-btn button:hover { box-shadow: 0 0 18px rgba(0,229,255,0.5) !important; }

/* Status text */
#load-status p { color: #00e5ff !important; font-size: 0.88rem !important; text-align: center; }

/* ── Chat section ── */
#chat-header {
    background: #111 !important;
    border-bottom: 1px solid #1e1e1e !important;
    padding: 0.5rem 1rem !important;
}
#repo-info p, #repo-info * { color: #888 !important; font-size: 0.82rem !important; margin: 0 !important; }

/* Chatbot bubbles */
.message.bot {
    background: #161616 !important; border: 1px solid #1e1e1e !important;
    border-radius: 10px !important; color: #e0e0e0 !important;
}
.message.user {
    background: #002a2e !important; border: 1px solid rgba(0,229,255,0.15) !important;
    border-radius: 10px !important; color: #e0e0e0 !important;
}

/* Message input */
#msg-input textarea {
    background: #141414 !important; border: 1px solid #272727 !important;
    color: #e0e0e0 !important; border-radius: 8px !important;
    font-size: 0.93rem !important;
}
#msg-input textarea:focus { border-color: #00e5ff !important; box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important; }
#msg-input textarea::placeholder { color: #3a3a3a !important; }

/* Send button */
#send-btn button {
    background: #00e5ff !important;
    background-color: #00e5ff !important;
    background-image: none !important;
    color: #0a0a0a !important;
    border: none !important; border-radius: 8px !important; font-weight: 700 !important;
}
#send-btn button:hover { box-shadow: 0 0 12px rgba(0,229,255,0.4) !important; }
#send-btn button:disabled { background: #182828 !important; background-color: #182828 !important; background-image: none !important; color: #254040 !important; }

/* Secondary buttons */
#clear-chat-btn button, #new-repo-btn button {
    background: transparent !important; border-radius: 6px !important;
    font-size: 0.8rem !important; transition: border-color 0.15s, color 0.15s !important;
}
#clear-chat-btn button { border: 1px solid #2e2e2e !important; color: #777 !important; }
#clear-chat-btn button:hover { border-color: #555 !important; color: #aaa !important; }
#new-repo-btn button { border: 1px solid rgba(0,229,255,0.35) !important; color: #00e5ff !important; }
#new-repo-btn button:hover { background: rgba(0,229,255,0.07) !important; }

/* Chatbot container */
#chatbot { border: 1px solid #1e1e1e !important; border-radius: 8px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,229,255,0.25); }

input, textarea { background: #141414 !important; color: #e0e0e0 !important; }
"""

LOGO_HTML = """
<div style="text-align:center; padding: 0 0 2rem 0;">
  <div style="font-size:4rem; font-weight:900; letter-spacing:-3px; color:#00e5ff;
              line-height:1; font-family:'Segoe UI',system-ui,sans-serif;">
    CodeLens
  </div>
  <div style="font-size:1.05rem; color:#aaa; margin-top:0.8rem; line-height:1.5;">
    Understand any public GitHub repository — instantly.
  </div>
  <div style="font-size:0.88rem; color:#555; margin-top:0.5rem; line-height:1.6; max-width:380px; margin-left:auto; margin-right:auto;">
    Paste a repo URL and ask questions like a senior engineer who's already read
    the whole codebase. Every answer cites the exact source files it came from.
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
