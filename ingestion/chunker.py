"""
Splits raw file contents into semantically meaningful chunks with metadata.

Strategy per file type:
  - Code files (Python, JS, TS, Go, etc.) → language-aware splitting (functions/classes stay intact)
  - Markdown / RST                         → heading-based splitting
  - Jupyter notebooks                      → extract cell sources first, then chunk
  - Everything else                        → generic recursive character splitter
"""

import json

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks
MIN_CHUNK_LENGTH = 50  # discard chunks shorter than this — they're usually noise

# Maps the language labels from detect_language() to LangChain's Language enum.
# Only languages that LangChain supports code-aware splitting for are listed here.
# Everything else falls back to the generic splitter.
_LANGUAGE_ENUM_MAP: dict[str, Language] = {
    "Python":             Language.PYTHON,
    "JavaScript":         Language.JS,
    "TypeScript":         Language.TS,
    "Go":                 Language.GO,
    "Ruby":               Language.RUBY,
    "Rust":               Language.RUST,
    "C":                  Language.C,
    "C++":                Language.CPP,
    "C#":                 Language.CSHARP,
    "Scala":              Language.SCALA,
    "Swift":              Language.SWIFT,
    "Kotlin":             Language.KOTLIN,
    "Java":               Language.JAVA,
    "PHP":                Language.PHP,
    "R":                  Language.R,
    "Markdown":           Language.MARKDOWN,
    "reStructuredText":   Language.RST,
    "HTML":               Language.HTML,
    "PowerShell":         Language.POWERSHELL,
    "Latex":              Language.LATEX,
    # SQL has no Language enum entry — falls back to generic splitter
}


def _make_splitter(language: str) -> RecursiveCharacterTextSplitter:
    """
    Return the best splitter for the given language label.
    Falls back to a generic splitter if the language is unsupported or
    LangChain raises an error (e.g. enum value missing in installed version).
    """
    lang_enum = _LANGUAGE_ENUM_MAP.get(language)
    if lang_enum is not None:
        try:
            return RecursiveCharacterTextSplitter.from_language(
                language=lang_enum,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        except Exception:
            pass  # Fall through to generic splitter

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


def _extract_notebook_text(raw_content: str) -> str:
    """
    Pull code and markdown cell sources out of a Jupyter notebook's raw JSON.
    Returns them joined as plain text — one cell per block.
    Skips output cells (stdout, stderr, images) — we only want the source.
    """
    try:
        nb = json.loads(raw_content)
    except json.JSONDecodeError:
        return raw_content  # Malformed notebook — chunk the raw JSON as-is

    blocks = []
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", [])

        # source can be a list of lines or a single string
        if isinstance(source, list):
            text = "".join(source)
        else:
            text = source

        text = text.strip()
        if not text:
            continue

        # Prefix code cells so the LLM knows what it's reading
        if cell_type == "code":
            blocks.append(f"# Code cell\n{text}")
        elif cell_type == "markdown":
            blocks.append(text)

    return "\n\n".join(blocks)


def chunk_file(file: dict) -> list[dict]:
    """
    Split a single file dict (from scrape_repo) into chunks with metadata.

    Input file dict shape:
        {filename, path, language, content, size}

    Returns a list of chunk dicts:
        {content, metadata: {source_file, file_path, language, chunk_index}}
    """
    content = file["content"]
    language = file["language"]

    # Pre-process Jupyter notebooks before chunking
    if language == "Jupyter Notebook":
        content = _extract_notebook_text(content)
        language = "Python"  # Cells are Python — use code-aware splitting

    splitter = _make_splitter(language)
    raw_chunks = splitter.split_text(content)

    chunks = []
    for i, text in enumerate(raw_chunks):
        if len(text.strip()) < MIN_CHUNK_LENGTH:
            continue
        chunks.append({
            "content": text,
            "metadata": {
                "source_file": file["filename"],
                "file_path":   file["path"],
                "language":    file["language"],
                "chunk_index": i,
            },
        })

    return chunks


def chunk_repo(files: list[dict]) -> list[dict]:
    """
    Chunk all files from a scraped repo result.
    Takes the `files` list from scrape_repo() and returns a flat list of
    all chunks across all files, ready to be embedded.
    """
    all_chunks = []
    for file in files:
        all_chunks.extend(chunk_file(file))
    return all_chunks
