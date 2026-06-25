"""pybench benchmark: train a Flax NNX MLP on MNIST.

Exercises pybench's ``list[dict]`` multi-step format, multiple metrics per step,
and the ``min:`` lower-is-better convention on a real (noisy) training run.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from mnist.data import load_mnist
from mnist.model import MLP
from mnist.train import train_step


def bench_mnist_mlp(seed: int, *, n_seeds: int = 5, workers: int = 1) -> list[dict]:
    """Train an MLP on MNIST and report metrics at fixed step checkpoints.

    Args:
        seed: Random seed controlling parameter init and batch sampling.
        n_seeds: Seeds pybench samples (kept low — training is costly).
        workers: Parallel seed workers; kept at 1 because training holds a device.

    Returns:
        One record per checkpoint, e.g.
        ``[{"step": 200, "min:train_loss": ..., "accuracy": ...}, ...]``.
    """

    x_train, y_train, x_test, y_test = load_mnist()

    lr = 1e-3
    batch_size = 128
    # A toggle to simulate an improvement: a wider hidden layer lifts accuracy.
    hidden = 512 if os.environ.get("PYBENCH_MNIST_WIDE") else 128
    checkpoints = (200, 1000)
    n_classes = 10

    model = MLP(x_train.shape[1], hidden, n_classes, rngs=nnx.Rngs(seed))

    # A toggle to simulate a regression: Adam (the default) trains well; set
    # PYBENCH_MNIST_SGD to fall back to SGD, which barely moves at this rate.
    if os.environ.get("PYBENCH_MNIST_SGD"):
        optimizer = nnx.Optimizer(model, optax.sgd(lr), wrt=nnx.Param)
    else:
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    rng = np.random.default_rng(seed)
    x_test_j = jnp.asarray(x_test)
    y_test_j = jnp.asarray(y_test)

    curve: list[dict] = []
    for step in range(1, max(checkpoints) + 1):
        idx = rng.integers(0, len(x_train), size=batch_size)
        loss = train_step(
            model, optimizer, jnp.asarray(x_train[idx]), jnp.asarray(y_train[idx])
        )
        if step in checkpoints:
            preds = jnp.argmax(model(x_test_j), axis=1)
            accuracy = float((preds == y_test_j).mean())
            curve.append(
                {
                    "step": step,
                    "min:train_loss": float(loss),
                    "accuracy": accuracy,
                }
            )
    return curve
