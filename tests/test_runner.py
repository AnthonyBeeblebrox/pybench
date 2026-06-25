"""Tests for the seed runner."""

from pathlib import Path

import numpy as np
import pytest

from pybench import runner
from pybench.config import BenchConfig
from pybench.discovery import Benchmark, discover
from pybench.runner import RunShapeError, run_benchmark, sample_seeds


def _bench(fn):
    return Benchmark("bench_x", fn, BenchConfig(), Path("unused.py"))


def test_sample_seeds_in_range():
    seeds = sample_seeds(5, np.random.default_rng(0))
    assert len(seeds) == 5
    assert all(isinstance(s, int) for s in seeds)
    assert all(0 <= s < 2**32 for s in seeds)


def test_run_scalar_collects_aligned():
    out = run_benchmark(_bench(lambda seed: float(seed)), [1, 2, 3])
    assert out == {0: {"score": [1.0, 2.0, 3.0]}}


def test_run_list_collects_per_step():
    def fn(seed):
        return [
            {"step": 1, "min:loss": 0.1 * seed},
            {"step": 2, "f1": 0.2 * seed},
        ]

    out = run_benchmark(_bench(fn), [1, 2])
    assert out[1]["min:loss"] == pytest.approx([0.1, 0.2])
    assert out[2]["f1"] == pytest.approx([0.2, 0.4])


def test_inconsistent_keys_raise():
    def fn(seed):
        return {"a": 1.0} if seed == 1 else {"b": 2.0}

    with pytest.raises(RunShapeError, match="inconsistent keys"):
        run_benchmark(_bench(fn), [1, 2])


def test_on_seed_called_per_seed():
    calls = []
    run_benchmark(
        _bench(lambda seed: float(seed)), [1, 2, 3], on_seed=lambda: calls.append(1)
    )
    assert len(calls) == 3


def test_worker_helpers(tmp_path):
    # _init_worker / _run_seed run in subprocesses (not coverage-traced there),
    # so exercise them directly in-process.
    f = tmp_path / "bench_w.py"
    f.write_text("def bench_w(seed):\n    return float(seed)\n")
    runner._init_worker(str(f), "bench_w")
    assert runner._run_seed(3) == {0: {"score": 3.0}}


def test_run_parallel_collects_aligned(tmp_path):
    f = tmp_path / "bench_p.py"
    f.write_text("def bench_p(seed, *, workers=2):\n    return float(seed) * 2\n")
    bench = discover(f)[0]
    assert bench.config.workers == 2
    assert run_benchmark(bench, [1, 2]) == {0: {"score": [2.0, 4.0]}}  # no callback
    calls = []
    out = run_benchmark(bench, [1, 2, 3, 4], on_seed=lambda: calls.append(1))
    assert out == {0: {"score": [2.0, 4.0, 6.0, 8.0]}}
    assert len(calls) == 4
