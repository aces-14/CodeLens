"""
Fetches file contents from any public GitHub repository.

Flow:
  parse URL → get file tree (one API call) → filter files → fetch raw content
Returns a list of file dicts ready for the chunker.
"""

import base64
import os
import re
import time
from typing import Optional

import requests
from dotenv import load_dotenv

from ingestion.file_filter import detect_language, is_allowed

load_dotenv()

GITHUB_API = "https://api.github.com"
MAX_FILES = 500


class GitHubScraperError(Exception):
    """Raised for known, user-facing failure conditions."""


def _build_headers() -> dict:
    token = (os.getenv("GITHUB_TOKEN") or "").strip()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_github_url(url: str) -> tuple[str, str]:
    """
    Extract (owner, repo) from a GitHub URL.
    Accepts:
      https://github.com/owner/repo
      https://github.com/owner/repo/
      https://github.com/owner/repo/tree/main
      github.com/owner/repo
    """
    url = url.strip().rstrip("/")
    # Normalize — add scheme if missing
    if not url.startswith("http"):
        url = "https://" + url

    pattern = r"https?://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    if not match:
        raise GitHubScraperError(
            "Invalid GitHub URL. Expected format: https://github.com/owner/repo"
        )

    owner, repo = match.group(1), match.group(2)
    # Strip .git suffix if present
    repo = repo.replace(".git", "")
    return owner, repo


def _get_default_branch(owner: str, repo: str, headers: dict) -> str:
    """Fetch the repo's default branch name."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
    except requests.exceptions.ConnectionError:
        raise GitHubScraperError(
            "Cannot reach GitHub. Check your internet connection."
        )
    except requests.exceptions.Timeout:
        raise GitHubScraperError(
            "GitHub API request timed out. Try again."
        )

    if resp.status_code == 401:
        raise GitHubScraperError(
            "GitHub API returned 401 Unauthorized. "
            "Your GITHUB_TOKEN may be invalid or expired. "
            "Remove it from .env to use unauthenticated access (60 req/hr limit)."
        )
    if resp.status_code == 404:
        raise GitHubScraperError(
            f"Repository '{owner}/{repo}' not found. "
            "It may be private or the URL is incorrect."
        )
    if resp.status_code == 403:
        _raise_rate_limit_error(resp)
    resp.raise_for_status()

    return resp.json().get("default_branch", "main")


def _get_file_tree(owner: str, repo: str, branch: str, headers: dict) -> list[dict]:
    """
    Fetch the full recursive file tree in one API call.
    Returns raw tree items from GitHub.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except requests.exceptions.ConnectionError:
        raise GitHubScraperError(
            "Cannot reach GitHub. Check your internet connection."
        )
    except requests.exceptions.Timeout:
        raise GitHubScraperError(
            "GitHub API request timed out. Try again."
        )

    if resp.status_code == 403:
        _raise_rate_limit_error(resp)
    if resp.status_code == 404:
        raise GitHubScraperError(
            f"Could not find the file tree for '{owner}/{repo}'. "
            "The repository may be empty."
        )
    resp.raise_for_status()

    data = resp.json()

    # GitHub returns truncated=True for very large repos (>100k items)
    if data.get("truncated"):
        raise GitHubScraperError(
            f"Repository '{owner}/{repo}' is too large to process "
            "(file tree was truncated by GitHub). Try a smaller repository."
        )

    return [item for item in data.get("tree", []) if item["type"] == "blob"]


def _fetch_file_content(
    owner: str, repo: str, path: str, headers: dict
) -> Optional[str]:
    """
    Fetch and decode the raw content of a single file.
    Returns None if the file cannot be decoded (binary).
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    resp = requests.get(url, headers=headers, timeout=15)

    if resp.status_code == 403:
        _raise_rate_limit_error(resp)
    if resp.status_code != 200:
        return None  # Skip silently — individual file failures are non-fatal

    data = resp.json()
    encoding = data.get("encoding")

    if encoding != "base64":
        return None  # Binary or unsupported encoding

    try:
        raw = base64.b64decode(data["content"]).decode("utf-8")
        return raw
    except (UnicodeDecodeError, KeyError):
        return None  # Binary file — skip


def _raise_rate_limit_error(resp: requests.Response):
    reset_ts = resp.headers.get("X-RateLimit-Reset")
    if reset_ts:
        wait_min = max(0, int(reset_ts) - int(time.time())) // 60
        raise GitHubScraperError(
            f"GitHub API rate limit exceeded. Resets in ~{wait_min} minute(s). "
            "Set a GITHUB_TOKEN environment variable to raise the limit to 5,000 requests/hour."
        )
    raise GitHubScraperError(
        "GitHub API rate limit exceeded. "
        "Set a GITHUB_TOKEN environment variable to raise the limit to 5,000 requests/hour."
    )


def scrape_repo(url: str) -> dict:
    """
    Main entry point. Given a public GitHub repo URL, return:
    {
        "repo_name": str,
        "owner": str,
        "branch": str,
        "total_files_found": int,
        "files_ingested": int,
        "languages": list[str],
        "files": [
            {
                "filename": str,
                "path": str,
                "language": str,
                "content": str,
                "size": int,
            },
            ...
        ]
    }
    Raises GitHubScraperError for all known failure cases.
    """
    owner, repo = parse_github_url(url)
    headers = _build_headers()

    # Step 1: Get default branch
    branch = _get_default_branch(owner, repo, headers)

    # Step 2: Get full file tree
    all_blobs = _get_file_tree(owner, repo, branch, headers)
    total_found = len(all_blobs)

    # Step 3: Filter files
    accepted = [
        blob for blob in all_blobs
        if is_allowed(blob["path"], size=blob.get("size", 0))
    ]

    if not accepted:
        raise GitHubScraperError(
            f"No supported files found in '{owner}/{repo}'. "
            "The repository may contain only binary or unsupported file types."
        )

    if len(accepted) > MAX_FILES:
        raise GitHubScraperError(
            f"Repository '{owner}/{repo}' has {len(accepted)} supported files, "
            f"which exceeds the {MAX_FILES}-file limit. "
            "Try pointing at a specific subdirectory or a smaller repository."
        )

    # Step 4: Fetch content for each accepted file
    files = []
    languages_seen = set()

    for blob in accepted:
        path = blob["path"]
        size = blob.get("size", 0)
        language = detect_language(path)

        content = _fetch_file_content(owner, repo, path, headers)
        if content is None:
            continue  # Binary or undecodable — skip silently

        languages_seen.add(language)
        files.append({
            "filename": path.split("/")[-1],
            "path": path,
            "language": language,
            "content": content,
            "size": size,
        })

    if not files:
        raise GitHubScraperError(
            f"Could not read any files from '{owner}/{repo}'. "
            "All accepted files may be binary."
        )

    return {
        "repo_name": repo,
        "owner": owner,
        "branch": branch,
        "total_files_found": total_found,
        "files_ingested": len(files),
        "languages": sorted(languages_seen),
        "files": files,
    }
