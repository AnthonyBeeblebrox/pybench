"""pybench benchmark: a synthetic exponential-decay loss curve.

A dependency-light stand-in for a real training run (numpy only). Each seed
samples one noisy loss curve (``synthetic.sample_loss_curves``) and reports
``min:loss`` at fixed step checkpoints, exercising pybench's ``list[dict]``
multi-step format and the ``min:`` lower-is-better convention.

Set the ``PYBENCH_SYNTHETIC_REGRESS`` environment variable to inject a regression
into the run, so the CLI walkthrough can show a baseline pass turn into a failure:

- ``global`` — every checkpoint's loss rises (a broad regression);
- ``local`` — a single checkpoint's loss spikes (a localized regression).

Set ``PYBENCH_SYNTHETIC_RESAMPLE`` to draw a *different* curve for each seed
(``seed`` is deterministically remapped to a new seed). The scores no longer match
the baseline bit-for-bit, yet without a real regression the test still passes —
showing the verdict is statistical, not a score-equality check. The remap is
deterministic, so the run stays reproducible.
"""

from __future__ import annotations

import os

import numpy as np
from synthetic import sample_loss_curves

_CHECKPOINTS = (1, 30, 100)
_AMP = 1.0
_TAU = 30.0
_FLOOR = 0.10
_NOISE = 0.05

_GLOBAL_SHIFT = 0.05  # added to every checkpoint when REGRESS=global
_LOCAL_SPIKE = 0.20  # added to the last checkpoint when REGRESS=local
_SPIKE_STEP = _CHECKPOINTS[-1]
_SEED_MAX = 2**32


def bench_synthetic(seed: int, *, n_seeds: int = 30) -> list[dict]:
    """Sample one noisy decaying loss curve and report it at checkpoints.

    Args:
        seed: Random seed controlling the curve's noise.
        n_seeds: Seeds pybench samples for the baseline.

    Returns:
        One record per checkpoint, e.g. ``[{"step": 1, "min:loss": ...}, ...]``.
    """
    del n_seeds  # consumed by pybench, not used inside the function
    if os.environ.get("PYBENCH_SYNTHETIC_RESAMPLE"):
        seed = int(np.random.default_rng(seed).integers(_SEED_MAX))  # different curve
    rng = np.random.default_rng(seed)
    curve = sample_loss_curves(
        rng,
        n_seeds=1,
        n_steps=max(_CHECKPOINTS) + 1,
        amp=_AMP,
        tau=_TAU,
        floor=_FLOOR,
        noise=_NOISE,
    )[0]
    losses = {s: float(curve[s]) for s in _CHECKPOINTS}
    regress = os.environ.get("PYBENCH_SYNTHETIC_REGRESS")
    if regress == "global":
        losses = {s: v + _GLOBAL_SHIFT for s, v in losses.items()}
    elif regress == "local":
        losses[_SPIKE_STEP] += _LOCAL_SPIKE
    return [{"step": s, "min:loss": losses[s]} for s in _CHECKPOINTS]
