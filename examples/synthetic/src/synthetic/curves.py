"""A noisy exponential-decay loss-curve sampler.

Shared by the ``bench_synthetic`` benchmark and the standalone ``main.py``
illustration so both draw curves the same way.
"""

from __future__ import annotations

import numpy as np


def sample_loss_curves(
    rng: np.random.Generator,
    *,
    n_seeds: int,
    n_steps: int,
    amp: float,
    tau: float,
    floor: float,
    noise: float,
    seed_sigma: float = 0.0,
) -> np.ndarray:
    """Sample ``(n_seeds, n_steps)`` losses: ``amp*exp(-t/tau)+floor + N(0,noise)``.

    Args:
        rng: NumPy random generator driving the additive noise.
        n_seeds: Number of independent curves (rows) to sample.
        n_steps: Number of steps (columns) per curve.
        amp: Initial excess loss above ``floor`` at step 0.
        tau: Exponential decay constant of the mean curve.
        floor: Asymptotic loss the mean curve decays toward.
        noise: Standard deviation of the per-step Gaussian noise.
        seed_sigma: Standard deviation of a per-seed offset shared across all
            steps of a curve. ``0`` (default) keeps steps independent; a positive
            value makes a lucky/unlucky seed shift its whole curve, correlating
            the steps within a seed (as a real training run's steps are).

    Returns:
        A ``(n_seeds, n_steps)`` array of sampled losses.
    """
    steps = np.arange(n_steps)
    mean_curve = amp * np.exp(-steps / tau) + floor
    curves = mean_curve + rng.normal(0.0, noise, size=(n_seeds, n_steps))
    if seed_sigma:
        curves = curves + rng.normal(0.0, seed_sigma, size=(n_seeds, 1))
    return curves
