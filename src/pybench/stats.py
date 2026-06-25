"""Statistical comparison: paired t-test slots + sign-flip permutation meta-test.

For each ``(step, metric)`` slot a one-sided paired t-test (in *goodness* space,
i.e. after the ``min:`` sign flip) decides whether the current run regressed.
The benchmark verdict is the within-seed sign-flip permutation p-value of a
continuous severity statistic (see ``SPECIFICATIONS.md`` §3).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

_EPS = 1e-9
_ZERO_STD = 1e-12  # treat std below this (e.g. identical values) as degenerate
_BIG_T = 1e9  # finite stand-in for an infinite t (degenerate constant slot)
DEFAULT_N_PERM = 4096

SeedScores = dict[int, dict[str, list[float]]]
"""Per-seed raw scores: ``{step: {metric: [value for each seed]}}``."""

_FloatArray = npt.NDArray[np.float64]
_BoolArray = npt.NDArray[np.bool_]


def _t_ppf(q: float, df: int) -> float:
    ppf = stats.t.ppf(q, df)  # pyright: ignore[reportUnknownMemberType]
    return float(ppf)


def _t_cdf(values: _FloatArray, df: int) -> _FloatArray:
    cdf = stats.t.cdf(values, df)  # pyright: ignore[reportUnknownMemberType]
    return np.asarray(cdf, dtype=np.float64)


@dataclass(frozen=True)
class SlotResult:
    """Comparison outcome for one ``(step, metric)`` slot, in raw units."""

    step: int
    metric: str
    baseline_mean: float
    baseline_std: float
    current_mean: float
    current_std: float
    effect_size: float
    p_value: float
    flagged: bool
    denom_at_floor: bool
    """True when the baseline mean is so small that effect_size is unreliable."""


@dataclass(frozen=True)
class Comparison:
    """Full benchmark comparison result."""

    slots: list[SlotResult]
    n_flagged: int
    n_slots: int
    meta_p: float
    passed: bool


def _goodness(metric: str, values: _FloatArray) -> _FloatArray:
    """Convert raw metric values to goodness, negating ``min:`` metrics."""
    if metric.startswith("min:"):
        return -values
    return values


def _flag_mask(
    diffs: _FloatArray,
    denom: _FloatArray,
    *,
    min_effect: float | None,
    t_crit: float,
) -> _BoolArray:
    """Boolean mask of slots flagged as regressions for a difference matrix.

    Args:
        diffs: ``(n_seeds, n_slots)`` goodness differences (current - baseline).
        denom: ``(n_slots,)`` effect-size denominators.
        min_effect: Optional minimum relative effect gate.
        t_crit: Critical (negative) t value for the one-sided test at ``alpha``.

    Returns:
        A ``(n_slots,)`` boolean mask of flagged slots.
    """
    n = diffs.shape[0]
    mean = diffs.mean(axis=0)
    std = diffs.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = mean / (std / np.sqrt(n))
    flagged = t_stat < t_crit
    if min_effect is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            effect = mean / denom
        flagged &= effect < -min_effect
    return flagged


def _severity(
    diffs: _FloatArray,
    denom: _FloatArray,
    *,
    min_effect: float | None,
    t_crit: float,
) -> float:
    """Total depth of the slots into the one-sided rejection region.

    For each slot we measure ``max(0, t_crit - t_stat)`` — how far the paired
    t-statistic falls *below* the critical value ``t_crit`` (itself negative),
    i.e. how deep into the regression-rejection region it sits. Summing across
    slots gives a continuous severity that, unlike a flagged *count*, preserves
    effect magnitude; for a single slot it is monotone in ``-t_stat``, so the
    sign-flip permutation reduces to the one-sided paired t-test (§3.2).

    A degenerate constant slot (zero variance) has an infinite t; it is mapped
    to a finite ``∓_BIG_T`` by the sign of its mean so the arithmetic stays
    well-defined. The ``min_effect`` gate zeroes a slot's contribution unless
    its relative goodness drop exceeds the threshold.

    Args:
        diffs: ``(n_seeds, n_slots)`` goodness differences (current - baseline).
        denom: ``(n_slots,)`` effect-size denominators.
        min_effect: Optional minimum relative effect gate.
        t_crit: Critical (negative) t value for the one-sided test at ``alpha``.

    Returns:
        The summed severity across slots (``>= 0``).
    """
    n = diffs.shape[0]
    mean = diffs.mean(axis=0)
    std = diffs.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = mean / (std / np.sqrt(n))
    degenerate = std < _ZERO_STD
    t_stat = np.where(
        degenerate,
        np.where(mean < 0.0, -_BIG_T, np.where(mean > 0.0, _BIG_T, 0.0)),
        t_stat,
    )
    contrib = np.maximum(0.0, t_crit - t_stat)
    if min_effect is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            effect = mean / denom
        contrib = np.where(effect < -min_effect, contrib, 0.0)
    return float(contrib.sum())


def _sign_flip_meta_p(
    diffs: _FloatArray,
    denom: _FloatArray,
    observed: float,
    *,
    min_effect: float | None,
    t_crit: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Permutation p-value of the severity under within-seed sign flips.

    When the whole sign-flip space ``2**n`` is no larger than the Monte-Carlo
    budget, enumerate it exactly: the result is then exact and deterministic at
    no extra cost. Otherwise sample ``n_perm`` random sign vectors.
    """
    n = diffs.shape[0]
    if 2**n <= n_perm:
        # Exact: enumerate every sign pattern. The unflipped arrangement is in
        # the set, so ``ge >= 1`` and the p-value floors at the true ``1/2**n``.
        ge = 0
        for combo in itertools.product((-1.0, 1.0), repeat=n):
            signs = np.array(combo).reshape(n, 1)
            sev = _severity(signs * diffs, denom, min_effect=min_effect, t_crit=t_crit)
            if sev >= observed:
                ge += 1
        return ge / 2**n
    ge = 0
    for _ in range(n_perm):
        signs = rng.choice((-1.0, 1.0), size=(n, 1))
        sev = _severity(signs * diffs, denom, min_effect=min_effect, t_crit=t_crit)
        if sev >= observed:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def check_alpha_detectable(n_seeds: int, alpha: float) -> None:
    """Reject an ``alpha`` that no regression could ever satisfy.

    The within-seed sign-flip meta-test has only ``2**n_seeds`` arrangements, so
    the smallest achievable ``meta_p`` is ``1 / 2**n_seeds``. When
    ``alpha <= 1 / 2**n_seeds`` the verdict ``meta_p < alpha`` is unsatisfiable —
    even a maximally severe regression yields a PASS — so flag it loudly rather
    than report a vacuous green.

    Raises:
        ValueError: If ``alpha`` is unreachable at this seed count.
    """
    if 2**n_seeds <= 1.0 / alpha:
        floor = 1.0 / 2**n_seeds
        msg = (
            f"alpha={alpha:g} can never flag a regression with n_seeds={n_seeds}: "
            f"the sign-flip meta-test floors at 1/2**{n_seeds}={floor:g} >= alpha. "
            f"Use more seeds (need 2**n_seeds > 1/alpha = {1.0 / alpha:g}) "
            f"or a larger alpha."
        )
        raise ValueError(msg)


def compare(
    baseline: SeedScores,
    current: SeedScores,
    *,
    alpha: float = 0.05,
    min_effect: float | None = None,
    n_perm: int = DEFAULT_N_PERM,
    rng: np.random.Generator | None = None,
) -> Comparison:
    """Compare a paired current run against a baseline.

    Args:
        baseline: Stored per-seed baseline scores.
        current: Per-seed current scores, on the same seeds (paired).
        alpha: Per-slot and overall significance threshold.
        min_effect: Optional minimum relative goodness drop to flag a slot.
        n_perm: Number of sign-flip permutations for the meta-test.
        rng: Random generator; a fresh default one is used when ``None``.

    Returns:
        A :class:`Comparison` with per-slot detail and the overall verdict.

    Raises:
        ValueError: If baseline and current have mismatched seed counts, or if
            ``alpha`` is unreachable at this seed count (see
            :func:`check_alpha_detectable`).
    """
    rng = np.random.default_rng() if rng is None else rng
    slot_keys = [
        (step, metric) for step in sorted(baseline) for metric in sorted(baseline[step])
    ]
    base_raw = [np.asarray(baseline[s][m], dtype=np.float64) for s, m in slot_keys]
    cur_raw = [np.asarray(current[s][m], dtype=np.float64) for s, m in slot_keys]
    good_base = np.column_stack(
        [_goodness(m, a) for (_, m), a in zip(slot_keys, base_raw, strict=True)]
    )
    good_cur = np.column_stack(
        [_goodness(m, a) for (_, m), a in zip(slot_keys, cur_raw, strict=True)]
    )
    if good_base.shape != good_cur.shape:
        msg = "baseline and current must have the same number of seeds"
        raise ValueError(msg)

    diffs = good_cur - good_base
    n = diffs.shape[0]
    check_alpha_detectable(n, alpha)
    base_abs_mean = np.abs(good_base.mean(axis=0))
    denom = np.maximum(base_abs_mean, _EPS)
    at_floor = base_abs_mean < _EPS
    t_crit = _t_ppf(alpha, n - 1)

    mean = diffs.mean(axis=0)
    std = diffs.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = mean / (std / np.sqrt(n))
    p_values = _t_cdf(t_stat, n - 1)
    degenerate = std < _ZERO_STD
    p_values = np.where(degenerate, np.where(mean < 0, 0.0, 1.0), p_values)
    effect = mean / denom

    flagged = _flag_mask(diffs, denom, min_effect=min_effect, t_crit=t_crit)
    n_flagged = int(np.count_nonzero(flagged))
    observed_severity = _severity(diffs, denom, min_effect=min_effect, t_crit=t_crit)
    meta_p = _sign_flip_meta_p(
        diffs,
        denom,
        observed_severity,
        min_effect=min_effect,
        t_crit=t_crit,
        n_perm=n_perm,
        rng=rng,
    )

    slots = [
        SlotResult(
            step=step,
            metric=metric,
            baseline_mean=float(base_raw[i].mean()),
            baseline_std=float(base_raw[i].std(ddof=1)),
            current_mean=float(cur_raw[i].mean()),
            current_std=float(cur_raw[i].std(ddof=1)),
            effect_size=float(effect[i]),
            p_value=float(p_values[i]),
            flagged=bool(flagged[i]),
            denom_at_floor=bool(at_floor[i]),
        )
        for i, (step, metric) in enumerate(slot_keys)
    ]
    return Comparison(
        slots=slots,
        n_flagged=n_flagged,
        n_slots=len(slot_keys),
        meta_p=meta_p,
        passed=meta_p >= alpha,
    )
