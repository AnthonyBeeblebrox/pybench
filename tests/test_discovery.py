"""Tests for benchmark discovery."""

import pytest

from pybench.discovery import DiscoveryError, discover


def test_discovers_and_sorts(tmp_path):
    (tmp_path / "b.py").write_text(
        "def bench_b(seed):\n    return 1.0\ndef bench_a(seed):\n    return 2.0\n"
    )
    assert [b.name for b in discover(tmp_path)] == ["bench_a", "bench_b"]


def test_ignores_imported_bench(tmp_path):
    (tmp_path / "m.py").write_text(
        "from json import dumps as bench_dumps\ndef bench_real(seed):\n    return 1.0\n"
    )
    assert [b.name for b in discover(tmp_path)] == ["bench_real"]


def test_config_extracted(tmp_path):
    (tmp_path / "c.py").write_text(
        "def bench_c(seed, *, n_seeds=7, alpha=0.01):\n    return 1.0\n"
    )
    cfg = discover(tmp_path)[0].config
    assert cfg.n_seeds == 7
    assert cfg.alpha == 0.01


def test_names_filter(tmp_path):
    (tmp_path / "d.py").write_text(
        "def bench_x(seed):\n    return 1.0\ndef bench_y(seed):\n    return 1.0\n"
    )
    assert [b.name for b in discover(tmp_path, ["bench_y"])] == ["bench_y"]


def test_missing_name_raises(tmp_path):
    (tmp_path / "d.py").write_text("def bench_x(seed):\n    return 1.0\n")
    with pytest.raises(DiscoveryError, match="not found"):
        discover(tmp_path, ["bench_z"])


def test_duplicate_raises(tmp_path):
    (tmp_path / "a.py").write_text("def bench_dup(seed):\n    return 1.0\n")
    (tmp_path / "b.py").write_text("def bench_dup(seed):\n    return 1.0\n")
    with pytest.raises(DiscoveryError, match="more than once"):
        discover(tmp_path)


def test_missing_path_raises(tmp_path):
    with pytest.raises(DiscoveryError, match="does not exist"):
        discover(tmp_path / "nope")


def test_single_file(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("def bench_f(seed):\n    return 1.0\n")
    assert [b.name for b in discover(f)] == ["bench_f"]


def test_unloadable_file_raises(tmp_path, monkeypatch):
    f = tmp_path / "f.py"
    f.write_text("def bench_f(seed):\n    return 1.0\n")
    monkeypatch.setattr(
        "pybench.discovery.importlib.util.spec_from_file_location",
        lambda *a, **k: None,
    )
    with pytest.raises(DiscoveryError, match="cannot import"):
        discover(f)
