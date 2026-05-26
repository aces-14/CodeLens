"""
Decides which files in a GitHub repo to accept or reject.
Rules: extension allowlist + directory denylist + per-file size cap.
"""

ALLOWED_EXTENSIONS = {
    # Code
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rb", ".rs", ".php",
    ".swift", ".kt", ".scala", ".r",
    # Web
    ".html", ".css", ".scss", ".svelte", ".vue",
    # Config / data
    ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".env.example",
    # Docs
    ".md", ".mdx", ".rst", ".txt",
    # Shell
    ".sh", ".bash", ".zsh", ".ps1",
    # SQL
    ".sql",
    # Notebooks
    ".ipynb",
}

DENIED_DIRECTORIES = {
    "node_modules", ".git", "dist", "build", "out",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    "venv", ".venv", "env", ".env",
    "vendor", "target", ".gradle", ".idea", ".vscode",
    "coverage", ".nyc_output", "htmlcov",
    "eggs", ".eggs", "*.egg-info",
    "migrations",  # often auto-generated, noisy
}

# Per-file size cap: skip files larger than this (bytes)
MAX_FILE_SIZE_BYTES = 100_000  # 100 KB


def is_allowed(path: str, size: int = 0) -> bool:
    """
    Return True if the file at `path` should be ingested.

    Args:
        path: File path relative to repo root, e.g. "src/utils/helper.py"
        size: File size in bytes (from GitHub API tree response)
    """
    parts = path.split("/")

    # Reject if any path segment is a denied directory
    for part in parts[:-1]:  # exclude the filename itself
        if part in DENIED_DIRECTORIES:
            return False

    # Reject hidden directories (starting with dot, excluding .env.example etc.)
    for part in parts[:-1]:
        if part.startswith(".") and part not in {".github"}:
            return False

    # Reject lock files and auto-generated files by name
    filename = parts[-1]
    if filename in {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "poetry.lock", "Pipfile.lock", "Gemfile.lock",
        "Cargo.lock", "composer.lock",
    }:
        return False

    # Check extension
    # Handle dotfiles with no extension (e.g. ".gitignore") — skip them
    if filename.startswith(".") and "." not in filename[1:]:
        return False

    # Extract extension (handle multi-part like .env.example)
    if filename.endswith(".env.example"):
        return True

    ext = _get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        return False

    # Size cap
    if size > MAX_FILE_SIZE_BYTES:
        return False

    return True


def _get_extension(filename: str) -> str:
    """Return the lowercase file extension including the dot."""
    dot_index = filename.rfind(".")
    if dot_index == -1:
        return ""
    return filename[dot_index:].lower()


def detect_language(path: str) -> str:
    """Return a human-readable language label based on file extension."""
    ext = _get_extension(path.split("/")[-1])
    language_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".tsx": "TypeScript", ".jsx": "JavaScript",
        ".java": "Java", ".c": "C", ".cpp": "C++", ".h": "C/C++",
        ".cs": "C#", ".go": "Go", ".rb": "Ruby", ".rs": "Rust",
        ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin",
        ".scala": "Scala", ".r": "R",
        ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
        ".svelte": "Svelte", ".vue": "Vue",
        ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
        ".json": "JSON", ".ini": "INI", ".cfg": "Config",
        ".md": "Markdown", ".mdx": "Markdown", ".rst": "reStructuredText",
        ".txt": "Text", ".sh": "Shell", ".bash": "Shell",
        ".zsh": "Shell", ".ps1": "PowerShell", ".sql": "SQL",
        ".ipynb": "Jupyter Notebook",
    }
    return language_map.get(ext, "Unknown")
