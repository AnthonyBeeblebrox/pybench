"""Per-benchmark configuration: keyword defaults plus CLI overrides."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace

_PARAMS = ("n_seeds", "alpha", "min_effect", "workers")


@dataclass(frozen=True)
class BenchConfig:
    """Resolved configuration for one benchmark."""

    n_seeds: int = 30
    alpha: float = 0.05
    min_effect: float | None = None
    workers: int = 1


def extract_config(fn: Callable[..., object]) -> BenchConfig:
    """Read a benchmark's keyword-only config defaults from its signature.

    Args:
        fn: The ``bench_*`` function to inspect.

    Returns:
        A :class:`BenchConfig`; any of ``n_seeds``, ``alpha``, ``min_effect``,
        ``workers`` not declared on ``fn`` keep their package default.
    """
    sig = inspect.signature(fn)
    values: dict[str, object] = {}
    for name in _PARAMS:
        param = sig.parameters.get(name)
        if param is not None and param.default is not inspect.Parameter.empty:
            values[name] = param.default
    return replace(BenchConfig(), **values)


def apply_overrides(
    config: BenchConfig,
    *,
    alpha: float | None = None,
    min_effect: float | None = None,
) -> BenchConfig:
    """Return ``config`` with non-``None`` CLI overrides applied.

    Args:
        config: The benchmark's resolved configuration.
        alpha: CLI ``--alpha`` override, or ``None`` to keep the benchmark's.
        min_effect: CLI ``--min-effect`` override, or ``None`` to keep it.

    Returns:
        A new :class:`BenchConfig` with the overrides merged in.
    """
    return replace(
        config,
        alpha=config.alpha if alpha is None else alpha,
        min_effect=config.min_effect if min_effect is None else min_effect,
    )
