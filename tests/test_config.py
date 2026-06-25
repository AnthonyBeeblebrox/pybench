"""Tests for benchmark config extraction and CLI overrides."""

from pybench.config import BenchConfig, apply_overrides, extract_config


def test_defaults_when_no_kwargs():
    def bench_x(seed):
        return 1.0

    assert extract_config(bench_x) == BenchConfig()


def test_reads_declared_kwargs():
    def bench_x(seed, *, n_seeds=7, alpha=0.01, min_effect=0.02, workers=4):
        return 1.0

    assert extract_config(bench_x) == BenchConfig(
        n_seeds=7, alpha=0.01, min_effect=0.02, workers=4
    )


def test_partial_kwargs_keep_other_defaults():
    def bench_x(seed, *, alpha=0.2):
        return 1.0

    cfg = extract_config(bench_x)
    assert cfg.alpha == 0.2
    assert cfg.n_seeds == 30


def test_overrides_none_keep_config():
    cfg = BenchConfig(alpha=0.01, min_effect=0.02)
    assert apply_overrides(cfg) == cfg


def test_overrides_applied():
    cfg = BenchConfig(alpha=0.01, min_effect=0.02)
    out = apply_overrides(cfg, alpha=0.1, min_effect=0.5)
    assert out.alpha == 0.1
    assert out.min_effect == 0.5
