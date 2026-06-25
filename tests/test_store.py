"""Tests for the JSONL baseline store."""

import json

from pybench.store import BaselineRecord, read_baselines, write_baselines


def _record():
    return BaselineRecord(
        bench="bench_x",
        timestamp="2026-06-22T10:00:00Z",
        git_commit="a3f4c1d",
        git_dirty=False,
        seeds=[1, 2, 3],
        scores={0: {"min:loss": [0.1, 0.2, 0.3]}, 10: {"f1": [0.4, 0.5, 0.6]}},
    )


def test_round_trip(tmp_path):
    path = tmp_path / ".pybench" / "baselines.jsonl"
    record = _record()
    write_baselines(path, [record])
    assert read_baselines(path) == {"bench_x": record}


def test_write_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "baselines.jsonl"
    write_baselines(path, [_record()])
    assert path.exists()


def test_read_missing_file_returns_empty(tmp_path):
    assert read_baselines(tmp_path / "nope.jsonl") == {}


def test_write_empty_records(tmp_path):
    path = tmp_path / "baselines.jsonl"
    write_baselines(path, [])
    assert path.read_text() == ""


def test_blank_lines_skipped(tmp_path):
    path = tmp_path / "baselines.jsonl"
    obj = {
        "bench": "a",
        "timestamp": "t",
        "git_commit": None,
        "git_dirty": None,
        "seeds": [1],
        "n": 1,
        "scores": {"0": {"score": [0.5]}},
    }
    path.write_text("\n" + json.dumps(obj) + "\n\n")
    out = read_baselines(path)
    assert set(out) == {"a"}
    assert out["a"].git_commit is None
    assert out["a"].scores == {0: {"score": [0.5]}}


def test_step_keys_serialized_as_strings(tmp_path):
    path = tmp_path / "baselines.jsonl"
    write_baselines(path, [_record()])
    obj = json.loads(path.read_text().splitlines()[0])
    assert set(obj["scores"]) == {"0", "10"}
    assert obj["n"] == 3
