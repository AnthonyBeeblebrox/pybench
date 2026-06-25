"""Tests for the paired slot test and sign-flip permutation meta-test."""

import itertools

import numpy as np
import pytest

from pybench import stats


def _brute_meta_p(diffs, denom, observed, *, t_crit, min_effect=None):
    """Independent exhaustive sign-flip p-value, for cross-checking the exact path."""
    n = diffs.shape[0]
    ge = sum(
        stats._severity(
            np.array(combo).reshape(n, 1) * diffs,
            denom,
            min_effect=min_effect,
            t_crit=t_crit,
        )
        >= observed
        for combo in itertools.product((-1.0, 1.0), repeat=n)
    )
    return ge / 2**n


def _curve_scores(rng, steps, metric, *, mean, std, n=20):
    """Build per-seed scores: each step gets ``n`` samples of N(mean, std)."""
    return {s: {metric: (mean + rng.normal(0, std, n)).tolist()} for s in steps}


def test_detects_consistent_regression():
    rng = np.random.default_rng(0)
    steps = list(range(1, 11))
    base = _curve_scores(rng, steps, "min:loss", mean=0.30, std=0.03)
    cur = _curve_scores(rng, steps, "min:loss", mean=0.36, std=0.03)
    result = stats.compare(base, cur, n_perm=1000, rng=np.random.default_rng(1))
    assert not result.passed
    assert result.meta_p < 0.05
    assert result.n_flagged > 0


def test_ignores_null():
    rng = np.random.default_rng(2)
    steps = list(range(1, 11))
    base = _curve_scores(rng, steps, "min:loss", mean=0.30, std=0.03)
    cur = _curve_scores(rng, steps, "min:loss", mean=0.30, std=0.03)
    result = stats.compare(base, cur, n_perm=1000, rng=np.random.default_rng(3))
    assert result.passed
    assert result.meta_p >= 0.05


def test_scalar_regression_detected():
    rng = np.random.default_rng(4)
    base = {0: {"score": (0.9 + rng.normal(0, 0.01, 30)).tolist()}}
    cur = {0: {"score": (0.8 + rng.normal(0, 0.01, 30)).tolist()}}
    result = stats.compare(base, cur, n_perm=1000, rng=np.random.default_rng(5))
    assert not result.passed
    assert result.n_slots == 1


def test_min_effect_gates_tiny_regression():
    rng = np.random.default_rng(6)
    steps = list(range(1, 6))
    base = _curve_scores(rng, steps, "min:loss", mean=0.30, std=0.001, n=30)
    cur = _curve_scores(rng, steps, "min:loss", mean=0.305, std=0.001, n=30)
    ungated = stats.compare(base, cur, n_perm=500, rng=np.random.default_rng(7))
    gated = stats.compare(
        base, cur, min_effect=0.05, n_perm=500, rng=np.random.default_rng(7)
    )
    assert ungated.n_flagged > 0
    assert gated.n_flagged == 0


def test_degenerate_no_change_not_flagged():
    base = {0: {"min:loss": [0.3] * 10}}
    cur = {0: {"min:loss": [0.3] * 10}}
    result = stats.compare(base, cur, n_perm=200, rng=np.random.default_rng(8))
    slot = result.slots[0]
    assert not slot.flagged
    assert slot.p_value == 1.0


def test_degenerate_strict_decrease_flagged():
    base = {0: {"min:loss": [0.3] * 10}}
    cur = {0: {"min:loss": [0.4] * 10}}
    result = stats.compare(base, cur, n_perm=200, rng=np.random.default_rng(9))
    slot = result.slots[0]
    assert slot.flagged
    assert slot.p_value == 0.0
    assert slot.effect_size < 0


def test_shape_mismatch_raises():
    base = {0: {"score": [0.1, 0.2, 0.3, 0.4, 0.5]}}
    cur = {0: {"score": [0.1, 0.2, 0.3, 0.4]}}
    with pytest.raises(ValueError, match="same number of seeds"):
        stats.compare(base, cur, n_perm=10, rng=np.random.default_rng(0))


def test_scalar_strong_regression_fails_at_small_n():
    # Regression guard: the old flagged-count meta-test had a meta-p floor near
    # alpha for K=1, so a huge scalar regression wrongly PASSED. The severity
    # statistic must fail it even at small n and regardless of effect magnitude.
    rng = np.random.default_rng(0)
    for n in (12, 20):
        base = {0: {"score": (0.9 + rng.normal(0, 0.01, n)).tolist()}}
        cur = {0: {"score": (0.9 - 5.0 + rng.normal(0, 0.01, n)).tolist()}}
        result = stats.compare(base, cur, n_perm=2000, rng=np.random.default_rng(1))
        assert not result.passed
        assert result.meta_p < 0.05


def test_compare_without_rng_runs():
    base = {0: {"score": [0.1, 0.2, 0.3, 0.4, 0.5]}}
    cur = {0: {"score": [0.1, 0.2, 0.3, 0.4, 0.5]}}
    result = stats.compare(base, cur, n_perm=10)
    assert isinstance(result, stats.Comparison)


def test_check_alpha_detectable_rejects_unreachable_alpha():
    # 1/2**4 = 0.0625 >= 0.05, so no regression could ever fail at alpha=0.05.
    with pytest.raises(ValueError, match="can never flag"):
        stats.check_alpha_detectable(4, 0.05)
    stats.check_alpha_detectable(5, 0.05)  # 1/2**5 = 0.031 < 0.05 -> feasible


def test_compare_rejects_unreachable_alpha():
    base = {0: {"score": [0.90, 0.91, 0.89, 0.92]}}  # 4 seeds
    cur = {0: {"score": [0.50, 0.51, 0.49, 0.52]}}
    with pytest.raises(ValueError, match="can never flag"):
        stats.compare(base, cur, alpha=0.05, n_perm=64, rng=np.random.default_rng(0))


def test_exact_meta_p_for_clean_regression():
    # A cleanly separated regression: the unflipped arrangement is the unique
    # most-extreme one, so the exact sign-flip p-value is exactly 1 / 2**n.
    rng = np.random.default_rng(0)
    base = {0: {"score": (0.9 + rng.normal(0, 0.001, 5)).tolist()}}
    cur = {0: {"score": (0.4 + rng.normal(0, 0.001, 5)).tolist()}}
    result = stats.compare(base, cur, n_perm=4096, rng=np.random.default_rng(1))
    assert result.meta_p == pytest.approx(1 / 2**5)


def test_exact_path_is_deterministic_and_matches_brute_force():
    # When 2**n <= n_perm the meta-p is enumerated exactly: independent of the
    # rng, and equal to an independent brute-force enumeration.
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(3, 9))  # 2**n in [8, 256] <= 4096 -> exact path
        diffs = rng.normal(0.0, 1.0, size=(n, 3))
        denom = np.maximum(np.abs(diffs.mean(axis=0)), stats._EPS)
        t_crit = stats._t_ppf(0.05, n - 1)
        observed = stats._severity(diffs, denom, min_effect=None, t_crit=t_crit)
        p1 = stats._sign_flip_meta_p(
            diffs,
            denom,
            observed,
            min_effect=None,
            t_crit=t_crit,
            n_perm=4096,
            rng=np.random.default_rng(1),
        )
        p2 = stats._sign_flip_meta_p(
            diffs,
            denom,
            observed,
            min_effect=None,
            t_crit=t_crit,
            n_perm=4096,
            rng=np.random.default_rng(999),
        )
        expected = _brute_meta_p(diffs, denom, observed, t_crit=t_crit)
        assert p1 == p2 == expected
