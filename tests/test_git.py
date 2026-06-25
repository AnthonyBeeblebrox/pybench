"""Tests for git provenance capture."""

import subprocess

from pybench import git
from pybench.git import file_at_commit, file_history, git_metadata


def _git(tmp_path, *args):
    subprocess.run(
        ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
    )


def _init_repo(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "Tester")
    _git(tmp_path, "config", "commit.gpgsign", "false")


def test_metadata_outside_repo(tmp_path):
    info = git_metadata(tmp_path)
    assert info.commit is None
    assert info.dirty is None


def test_metadata_clean_then_dirty(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "Tester")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "f.txt").write_text("hello")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "init")

    clean = git_metadata(tmp_path)
    assert clean.commit is not None
    assert clean.dirty is False

    (tmp_path / "g.txt").write_text("new")
    assert git_metadata(tmp_path).dirty is True


def test_git_not_installed(monkeypatch):
    def boom(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", boom)
    info = git_metadata()
    assert info.commit is None
    assert info.dirty is None
    assert git._run_git(["status"], None) is None


def test_file_history_outside_repo(tmp_path):
    assert file_history(tmp_path / "baselines.jsonl") is None


def test_file_history_untracked_file(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "other.txt").write_text("x")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "init")  # HEAD exists, but file untracked
    f = tmp_path / "baselines.jsonl"
    f.write_text("{}\n")
    assert file_history(f) == []


def test_file_history_lists_commits_chronologically(tmp_path):
    _init_repo(tmp_path)
    f = tmp_path / "baselines.jsonl"
    f.write_text("v1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "one")
    f.write_text("v2\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "two")

    hist = file_history(f)
    assert hist is not None and len(hist) == 2
    older, newer = hist
    # _run_git strips trailing whitespace; harmless for JSONL parsing.
    assert file_at_commit(older[0], f) == "v1"
    assert file_at_commit(newer[0], f) == "v2"


def test_file_at_commit_outside_repo(tmp_path):
    assert file_at_commit("deadbeef", tmp_path / "baselines.jsonl") is None
