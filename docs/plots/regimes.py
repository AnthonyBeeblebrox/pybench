"""Doc plot: the two Part 2 data regimes (correlated null, localized spike)."""

import matplotlib.pyplot as plt
import numpy as np
from synthetic import sample_loss_curves

rng = np.random.default_rng(1)
kw = dict(amp=1.0, tau=30.0, floor=0.10, noise=0.05)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
for row in sample_loss_curves(rng, n_seeds=6, n_steps=80, seed_sigma=0.05, **kw):
    ax1.plot(row, lw=1, alpha=0.8)
ax1.set(title="No-regression case: correlated steps", xlabel="step", ylabel="loss")

base = sample_loss_curves(rng, n_seeds=1, n_steps=60, **kw)[0]
cur = sample_loss_curves(rng, n_seeds=1, n_steps=60, **kw)[0]
cur[-1] += 0.33
ax2.plot(base, lw=1.5, label="baseline")
ax2.plot(cur, lw=1.5, label="regressed")
ax2.set(title="Local case: one regressed step", xlabel="step", ylabel="loss")
ax2.legend()
fig.tight_layout()
