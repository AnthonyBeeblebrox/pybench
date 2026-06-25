"""Doc plot: six curves from ``sample_loss_curves`` around the mean curve."""

import matplotlib.pyplot as plt
import numpy as np
from synthetic import sample_loss_curves

rng = np.random.default_rng(0)
curves = sample_loss_curves(
    rng, n_seeds=6, n_steps=100, amp=1.0, tau=30.0, floor=0.10, noise=0.05
)
steps = np.arange(100)
fig, ax = plt.subplots(figsize=(6, 3.5))
for row in curves:
    ax.plot(steps, row, lw=1, alpha=0.7)
ax.plot(steps, 1.0 * np.exp(-steps / 30.0) + 0.10, "k--", lw=2, label="mean")
ax.set(
    xlabel="step",
    ylabel="loss",
    title="sample_loss_curves: noisy exponential decay",
)
ax.legend()
fig.tight_layout()
