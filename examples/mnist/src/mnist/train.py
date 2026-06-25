from flax import nnx
import jax
import optax
from mnist.model import MLP


def loss_fn(model: MLP, xb: jax.Array, yb: jax.Array) -> jax.Array:
    logits = model(xb)
    return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()


@nnx.jit
def train_step(
    model: MLP, optimizer: nnx.Optimizer, xb: jax.Array, yb: jax.Array
) -> jax.Array:
    loss, grads = nnx.value_and_grad(loss_fn)(model, xb, yb)
    optimizer.update(model, grads)
    return loss
