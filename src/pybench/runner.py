"""Run a benchmark over a set of seeds and collect per-seed scores."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pybench.discovery import Benchmark, import_file
from pybench.normalizer import Scores, normalize

SeedScores = dict[int, dict[str, list[float]]]
"""Per-seed raw scores: ``{step: {metric: [value for each seed]}}``."""

_SEED_MAX = 2**32


class RunShapeError(ValueError):
    """Raised when a benchmark returns different keys across seeds."""


def sample_seeds(n: int, rng: np.random.Generator) -> list[int]:
    """Sample ``n`` distinct-enough random integer seeds.

    Args:
        n: Number of seeds to draw.
        rng: Random generator.

    Returns:
        A list of ``n`` Python ints in ``[0, 2**32)``.
    """
    return [int(s) for s in rng.integers(0, _SEED_MAX, size=n)]


def _shape(scores: Scores) -> dict[int, frozenset[str]]:
    """Return the ``{step: {metrics}}`` shape of a normalized result."""
    return {step: frozenset(metrics) for step, metrics in scores.items()}


def _assemble(name: str, seeds: list[int], results: list[Scores]) -> SeedScores:
    """Collect per-seed normalized results into aligned ``SeedScores``.

    Args:
        name: Benchmark name, for error messages.
        seeds: Seeds in order (aligned with ``results``).
        results: Normalized scores for each seed, same order as ``seeds``.

    Returns:
        Per-seed scores ``{step: {metric: [value for each seed]}}``.

    Raises:
        RunShapeError: If a later seed yielded different step/metric keys than
            the first.
    """
    scores: SeedScores = {}
    first_shape: dict[int, frozenset[str]] = {}
    for i, (seed, result) in enumerate(zip(seeds, results, strict=True)):
        if i == 0:
            first_shape = _shape(result)
            scores = {s: {m: [] for m in metrics} for s, metrics in result.items()}
        elif _shape(result) != first_shape:
            msg = (
                f"{name} returned inconsistent keys across seeds: "
                f"seed {seed} gave {dict(_shape(result))}, "
                f"first seed gave {dict(first_shape)}"
            )
            raise RunShapeError(msg)
        for step, metrics in result.items():
            for metric, value in metrics.items():
                scores[step][metric].append(value)
    return scores


_worker_fn: Callable[[int], object] | None = None


def _init_worker(file: str, name: str) -> None:
    """Import the benchmark file once per worker process and cache the function."""
    global _worker_fn
    _worker_fn = getattr(import_file(Path(file)), name)


def _run_seed(seed: int) -> Scores:
    """Run the cached worker function on one seed and normalize the result."""
    assert _worker_fn is not None  # set by _init_worker
    return normalize(_worker_fn(seed))


def run_benchmark(
    bench: Benchmark,
    seeds: list[int],
    *,
    on_seed: Callable[[], None] | None = None,
) -> SeedScores:
    """Run ``bench`` on each seed and collect aligned per-seed scores.

    Runs serially when ``workers == 1``; otherwise fans the seeds out across a
    process pool (each worker re-imports the benchmark file by path).

    Args:
        bench: The benchmark to run.
        seeds: Seeds to run, in order; output lists align position-by-position.
        on_seed: Optional callback invoked once per completed seed (progress).

    Returns:
        Per-seed scores ``{step: {metric: [value for each seed]}}``.

    Raises:
        RunShapeError: If a later seed yields different step/metric keys than
            the first.
    """
    results: list[Scores] = [{} for _ in seeds]
    if bench.config.workers > 1:
        with ProcessPoolExecutor(
            max_workers=bench.config.workers,
            initializer=_init_worker,
            initargs=(str(bench.file), bench.name),
        ) as pool:
            futures = {pool.submit(_run_seed, seed): i for i, seed in enumerate(seeds)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                if on_seed is not None:
                    on_seed()
    else:
        for i, seed in enumerate(seeds):
            results[i] = normalize(bench.fn(seed))
            if on_seed is not None:
                on_seed()
    return _assemble(bench.name, seeds, results)
