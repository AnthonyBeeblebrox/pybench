"""Why pybench permutes a *severity* statistic, on the three CLI cases.

A rigorous companion to Part 1 of the docs. Part 1 runs the pybench CLI on three
cases — no regression, a global regression, and a localized regression — and
watches the verdict flip from PASS to FAIL. This script replays those same three
cases as Monte-Carlo experiments and pits pybench's verdict against the simpler
tests it could have used, to show *why* it makes the choice it does.

Crucially, it does **not** reimplement pybench's statistics: the pybench verdict
here calls the very functions the CLI runs, ``pybench.stats._severity`` and
``pybench.stats._sign_flip_meta_p`` (``SPECIFICATIONS.md`` §3). Only the competing
*alternative* tests — which are not part of pybench — are written out here.

Run with::

    uv run --package synthetic python examples/synthetic/main.py

The metric is a loss (lower-is-better), following pybench's ``min:`` convention:
goodness = -loss, and a regression is a *drop* in goodness. Four verdicts are
compared on each case:

- **global t-test** — one t-test pooling every (seed, step) difference;
- **per-step t-test + binomial** — a t-test per step, then a binomial test on how
  many steps came out significant;
- **sign-flip on count** — within-seed sign-flip permutation of the *flagged
  count* (pybench's discarded early design, which throws away effect magnitude);
- **sign-flip on severity** — pybench's actual verdict, via its own functions.

Case 1 (no regression). With *no real change* but correlated steps (a per-seed
offset shifts a whole curve), the global t-test (correlated slots pooled as
independent → standard error too small) and the per-step binomial (correlated
steps reject together → the count is over-dispersed) fire far above ``alpha``.
Both sign-flip tests stay at ``alpha``: a permutation test is exactly calibrated
whatever statistic it permutes.

Case 2 (global regression). Every checkpoint regresses a little. This is the easy
case the global t-test is built for, and all four tests catch it — pybench gives
up no power on a broad regression.

Case 3 (local regression). A single checkpoint regresses sharply while the rest
are unchanged. The global t-test dilutes that one spike across all the steps; the
per-step binomial sees one significant step, indistinguishable from its
``alpha``-rate false positives; and the sign-flip *on count* throws away the
spike's magnitude (one flag is one flag) — all three miss it. Only the sign-flip
*on severity*, which keeps the magnitude, catches it. This is precisely why §3.2
permutes the severity rather than a count.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats
from synthetic import sample_loss_curves

from pybench.stats import _severity, _sign_flip_meta_p

N_SEEDS = 10
ALPHA = 0.05
N_PERM = 800
REPS = 200
_EPS = 1e-9

# Shared curve shape; ``n_steps``/``seed_sigma`` and the injected regression vary
# per case.
_CURVE_KW = {"n_seeds": N_SEEDS, "amp": 1.0, "tau": 30.0, "floor": 0.10, "noise": 0.05}

# Case 1: correlated steps (seed_sigma > 0), no real change.
_CORR_STEPS = 100
_CORR_SIGMA = 0.05
# Case 2: a broad regression — every checkpoint's loss rises a little.
_GLOBAL_STEPS = 100
_GLOBAL_SHIFT = 0.03
# Case 3: many steps, one of which regresses sharply.
_LOCAL_STEPS = 200
_LOCAL_SPIKE = 0.33


def goodness_diff(baseline: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Per-seed goodness difference for a ``min:`` metric (loss is negated)."""
    return (-current) - (-baseline)  # == baseline - current


def _t_crit(n: int) -> float:
    """One-sided critical (negative) t value at ``ALPHA`` for ``n`` seeds."""
    return float(stats.t.ppf(ALPHA, df=n - 1))


def global_t_test(diffs: np.ndarray) -> bool:
    """Alternative 1: one t-test over every (seed, step) difference at once.

    Pools the ``n_seeds * n_steps`` slots as independent samples. Positive
    cross-step correlation then deflates the standard error (it over-fires under
    the null), and a regression in a single step is diluted across all the others
    (it misses a localized regression).
    """
    flat = diffs.reshape(-1)
    t_stat = flat.mean() / (flat.std(ddof=1) / np.sqrt(flat.size))
    return bool(t_stat < stats.t.ppf(ALPHA, df=flat.size - 1))


def per_step_binomial(diffs: np.ndarray) -> bool:
    """Alternative 2: a paired t per step, then a binomial test on the count.

    Assumes the per-step rejections are independent ``Bernoulli(alpha)`` under
    the null. Correlated steps reject together, over-dispersing the count (it
    over-fires under the null); and one regressed step is a single rejection,
    indistinguishable from the ``alpha``-rate false positives (it misses a
    localized regression).
    """
    n, k = diffs.shape
    mean = diffs.mean(axis=0)
    std = diffs.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = mean / (std / np.sqrt(n))
    rejected = int(np.count_nonzero(t_stat < _t_crit(n)))
    p = stats.binomtest(rejected, k, ALPHA, alternative="greater").pvalue
    return bool(p < ALPHA)


def _flagged_count(diffs: np.ndarray, *, t_crit: float) -> int:
    """Count slots whose paired t-test rejects — the discarded *count* statistic.

    Binarizes each slot to a flag *before* permuting, so it throws away effect
    magnitude: one sharply-regressed slot counts the same as a barely-significant
    one. (pybench keeps the magnitude instead; see :func:`pybench_severity`.)
    """
    n = diffs.shape[0]
    mean = diffs.mean(axis=0)
    std = diffs.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = mean / (std / np.sqrt(n))
    return int(np.count_nonzero((t_stat < t_crit) | ((std == 0) & (mean < 0))))


def sign_flip_count(diffs: np.ndarray, *, rng: np.random.Generator) -> bool:
    """Alternative 3: within-seed sign-flip permutation of the flagged *count*.

    Exactly calibrated like pybench's test (it is a permutation test), but by
    permuting a *count* it discards each slot's effect magnitude, so it cannot
    tell a single sharp regression from a single marginal one.
    """
    t_crit = _t_crit(diffs.shape[0])
    observed = _flagged_count(diffs, t_crit=t_crit)
    ge = 0
    for _ in range(N_PERM):
        signs = rng.choice((-1.0, 1.0), size=(diffs.shape[0], 1))
        if _flagged_count(signs * diffs, t_crit=t_crit) >= observed:
            ge += 1
    p = (ge + 1) / (N_PERM + 1)
    return p < ALPHA


def pybench_severity(
    diffs: np.ndarray, denom: np.ndarray, *, rng: np.random.Generator
) -> bool:
    """Run pybench's actual verdict, via ``pybench.stats`` (no reimplementation).

    Calls the same ``_severity`` and ``_sign_flip_meta_p`` the CLI runs: the
    within-seed sign-flip permutation of the continuous severity
    ``T = Σ max(0, t_crit - t_stat)`` (``SPECIFICATIONS.md`` §3.2). Keeping the
    magnitude is what lets it catch a regression hiding in a single slot.
    """
    t_crit = _t_crit(diffs.shape[0])
    observed = _severity(diffs, denom, min_effect=None, t_crit=t_crit)
    p = _sign_flip_meta_p(
        diffs, denom, observed, min_effect=None, t_crit=t_crit, n_perm=N_PERM, rng=rng
    )
    return p < ALPHA


# name → verdict for one paired sample (True == flags a regression). Each test
# takes the goodness differences, their effect-size denominators, and an rng.
Test = Callable[[np.ndarray, np.ndarray, np.random.Generator], bool]
_TESTS: dict[str, Test] = {
    "global t-test (seed×step)": lambda d, _denom, _rng: global_t_test(d),
    "per-step t + binomial": lambda d, _denom, _rng: per_step_binomial(d),
    "sign-flip on count": lambda d, _denom, rng: sign_flip_count(d, rng=rng),
    "sign-flip on severity (pybench)": lambda d, denom, rng: pybench_severity(
        d, denom, rng=rng
    ),
}


def detection_rate(
    *,
    n_steps: int,
    seed_sigma: float,
    shift: float,
    spike: float,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Fraction of ``reps`` each test flags a regression, for one case.

    Args:
        n_steps: Number of steps (columns) per curve.
        seed_sigma: Per-seed offset std; ``> 0`` correlates the steps in a seed.
        shift: Loss added to *every* step of the current run; a broad regression.
        spike: Loss added to the current run's *last* step; a localized regression.
        reps: Number of independent paired baseline/current draws.
        rng: Generator driving the curves and the sign-flip permutations.

    Returns:
        ``{test_name: rate}`` — a false-positive rate when ``shift == spike == 0``,
        otherwise the power.
    """
    hits = {name: 0 for name in _TESTS}
    kw = {**_CURVE_KW, "n_steps": n_steps, "seed_sigma": seed_sigma}
    for _ in range(reps):
        baseline = sample_loss_curves(rng, **kw)
        current = sample_loss_curves(rng, **kw)
        if shift:
            current = current + shift  # broad regression
        if spike:
            current[:, -1] += spike  # one regressed checkpoint
        diffs = goodness_diff(baseline, current)
        denom = np.maximum(np.abs((-baseline).mean(axis=0)), _EPS)
        for name, test in _TESTS.items():
            hits[name] += test(diffs, denom, rng)
    return {name: h / reps for name, h in hits.items()}


def _print_table(rates: dict[str, float], *, column: str) -> None:
    """Print a per-test rate table with the given right-hand column label."""
    print(f"  {'test':<33}{column:>20}")
    for name in _TESTS:
        print(f"  {name:<33}{rates[name]:>20.3f}")


def main() -> None:
    """Run the three cases, report each test's rate, and assert the contrast."""
    rng = np.random.default_rng(0)

    print("Case 1 — no regression, but the steps within a seed correlate")
    print(f"  false-positive rate (target = alpha = {ALPHA})\n")
    fpr = detection_rate(
        n_steps=_CORR_STEPS,
        seed_sigma=_CORR_SIGMA,
        shift=0.0,
        spike=0.0,
        reps=REPS,
        rng=rng,
    )
    _print_table(fpr, column="false-positive rate")

    print("\nCase 2 — a global regression: every checkpoint's loss rises a little")
    print("  detection rate / power (higher is better)\n")
    glob = detection_rate(
        n_steps=_GLOBAL_STEPS,
        seed_sigma=0.0,
        shift=_GLOBAL_SHIFT,
        spike=0.0,
        reps=REPS,
        rng=rng,
    )
    _print_table(glob, column="detection rate")

    print("\nCase 3 — a local regression: one checkpoint spikes, the rest unchanged")
    print("  detection rate / power (higher is better)\n")
    power = detection_rate(
        n_steps=_LOCAL_STEPS,
        seed_sigma=0.0,
        shift=0.0,
        spike=_LOCAL_SPIKE,
        reps=REPS,
        rng=rng,
    )
    _print_table(power, column="detection rate")

    # Case 1: the naive tests over-fire under correlation; both permutation tests
    # (whatever statistic) stay at alpha.
    assert fpr["global t-test (seed×step)"] > 3 * ALPHA, fpr
    assert fpr["per-step t + binomial"] > 2 * ALPHA, fpr
    assert fpr["sign-flip on count"] < 2 * ALPHA, fpr
    assert fpr["sign-flip on severity (pybench)"] < 2 * ALPHA, fpr
    # Case 2: a broad regression is the easy case — every test catches it.
    assert min(glob.values()) > 0.85, glob
    # Case 3: every test that discards magnitude misses the localized regression;
    # only the severity permutation keeps it and catches it.
    assert power["global t-test (seed×step)"] < 0.5, power
    assert power["per-step t + binomial"] < 3 * ALPHA, power
    assert power["sign-flip on count"] < 0.5, power
    assert power["sign-flip on severity (pybench)"] > 0.85, power
    print(
        "\nOK: the naive tests raise false alarms under correlation (case 1); all "
        "tests catch a broad regression (case 2); but the naive tests and the count "
        "permutation miss the localized regression (case 3) — only the severity "
        "permutation does all three."
    )


if __name__ == "__main__":
    main()
