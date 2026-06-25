"""Capture git provenance (short SHA + dirty flag) with graceful fallback."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitInfo:
    """Git provenance recorded with a baseline write."""

    commit: str | None
    dirty: bool | None


def _run_git(args: list[str], cwd: Path | None) -> str | None:
    """Run a git command, returning trimmed stdout or ``None`` on any failure.

    Args:
        args: Arguments following ``git``.
        cwd: Directory to run in, or ``None`` for the current directory.

    Returns:
        Trimmed stdout, or ``None`` if git is missing or the command failed.
    """
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_metadata(cwd: Path | None = None) -> GitInfo:
    """Return the short HEAD SHA and dirty flag, or nulls if git is unavailable.

    Args:
        cwd: Directory to inspect; defaults to the current working directory.

    Returns:
        ``GitInfo(commit, dirty)``. Both are ``None`` when ``cwd`` is not a git
        repository or git is not installed.
    """
    sha = _run_git(["rev-parse", "--short", "HEAD"], cwd)
    if sha is None:
        return GitInfo(commit=None, dirty=None)
    status = _run_git(["status", "--porcelain"], cwd)
    return GitInfo(commit=sha, dirty=bool(status))


def file_history(path: Path) -> list[tuple[str, str]] | None:
    """Return commits that touched ``path``, oldest first.

    Args:
        path: File whose history to inspect (git is run in its directory).

    Returns:
        ``[(short_sha, date), ...]`` chronological, ``[]`` if the file has no
        commits, or ``None`` if not a git repo / git is unavailable.
    """
    out = _run_git(
        ["log", "--format=%h%x09%ad", "--date=short", "--", path.name], path.parent
    )
    if out is None:
        return None
    if not out:
        return []
    rows = [line.split("\t", 1) for line in out.splitlines()]
    rows.reverse()
    return [(sha, date) for sha, date in rows]


def file_at_commit(commit: str, path: Path) -> str | None:
    """Return the content of ``path`` as of ``commit`` via ``git show``.

    Args:
        commit: Commit-ish (e.g. a short SHA).
        path: File to read; git is run in its directory.

    Returns:
        The file's text at that commit, or ``None`` if git is unavailable or
        the path did not exist there.
    """
    prefix = _run_git(["rev-parse", "--show-prefix"], path.parent)
    if prefix is None:
        return None
    return _run_git(["show", f"{commit}:{prefix}{path.name}"], path.parent)
