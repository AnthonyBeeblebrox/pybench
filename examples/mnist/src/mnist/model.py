"""A small MLP defined with Flax NNX."""

from __future__ import annotations

import jax
from flax import nnx


class MLP(nnx.Module):
    """Two-layer MLP for MNIST classification."""

    def __init__(self, din: int, dhidden: int, dout: int, *, rngs: nnx.Rngs) -> None:
        """Initialize the layers.

        Args:
            din: Input feature dimension.
            dhidden: Hidden layer width.
            dout: Number of output classes.
            rngs: NNX random number generators for parameter init.
        """
        self.linear1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run the forward pass, returning class logits."""
        x = nnx.relu(self.linear1(x))
        return self.linear2(x)
