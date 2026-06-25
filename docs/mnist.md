# MNIST example

The MNIST example trains a small Flax NNX MLP and reports a multi-step
accuracy / loss curve — a genuinely noisy metric, which is exactly where the
paired-seed design earns its keep.

```python
def bench_mnist_mlp(seed: int, *, n_seeds: int = 5, workers: int = 1) -> list[dict]:
    """Train an MLP on MNIST; report loss/accuracy at fixed step checkpoints."""
    ...
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
    ...
    return [
        {"step": 200,  "min:train_loss": 0.33, "accuracy": 0.92},
        {"step": 1000, "min:train_loss": 0.17, "accuracy": 0.96},
    ]
```

Default parameters are `n_seeds=5` and `workers=1` (a single device holds the
model, so parallel seed processes don't apply).

## A full lifecycle with Git

Pull the example's stack on demand, then establish a baseline with the
well-trained (Adam) model:

```bash
uv sync --package mnist                                       # JAX/Flax/datasets
uv run --package mnist pybench examples/mnist/benchmarks/
#   bench_mnist_mlp   .......... NEW   2 metrics × 2 steps   (baseline saved)
#   ──────────────────────────────────────────────────────────────
#   0 failed, 0 passed, 1 new  in 32s

git add .pybench/baselines.jsonl && git commit -m "baseline: mnist mlp"
```

### A regression is caught

A bad change regresses the model. Here we set `PYBENCH_MNIST_SGD`, so training
falls back to plain SGD. Re-running reuses the **same 5 seeds** — a paired
comparison:

```bash
PYBENCH_MNIST_SGD=1 uv run --package mnist pybench examples/mnist/benchmarks/ -v
#   bench_mnist_mlp   .......... FAIL   2 metrics × 2 steps   4/4 slots flagged
#   meta-p=0.031
#   metric     step   baseline    current      Δ        p
#   accuracy    200   0.92±0.00   0.23±0.04   -74.7%   0.000  ✗
#   min:train_loss   200   0.25±0.04   2.22±0.03   -778.4%   0.000  ✗
#   accuracy   1000   0.96±0.00   0.65±0.03   -32.1%   0.000  ✗
#   min:train_loss  1000   0.12±0.03   1.78±0.04   -1440.9%   0.000  ✗
#   ──────────────────────────────────────────────────────────────
#   1 failed, 0 passed, 0 new  in 32s         # → exit code 1
```

`pybench` exits non-zero, failing CI like a broken `pytest`. Since this
regression is a mistake, you simply fix the code (drop the SGD path; no
rebaseline) and the next run goes green against the unchanged baseline.

### An improvement is accepted

Now a *good* change: a wider hidden layer (`PYBENCH_MNIST_WIDE`) genuinely lifts
accuracy. pybench tests for regressions one-sidedly, so a run that only gets
better passes:

```bash
PYBENCH_MNIST_WIDE=1 uv run --package mnist pybench examples/mnist/benchmarks/
#   bench_mnist_mlp   .......... PASS   2 metrics × 2 steps   0/4 slots flagged
#   meta-p=1.000
#   ──────────────────────────────────────────────────────────────
#   0 failed, 1 passed, 0 new  in 32s         # → exit code 0
```

> **pybench only flags regressions — judging whether a run is genuinely better
> is up to you.**

Assuming the improvement is real, the baseline still holds the *old, lower*
numbers, so a later regression would only be measured against the weaker model.
Lock the gain in as the new bar with `update`, then inspect the trail with
`show --history`:

```bash
PYBENCH_MNIST_WIDE=1 uv run --package mnist pybench update examples/mnist/benchmarks/ --yes
git add .pybench/baselines.jsonl && git commit -m "rebaseline: wider MLP"

uv run --package mnist pybench show --history
#   bench_mnist_mlp
#     888483b  2026-06-25  accuracy@200: 0.92  min:train_loss@200: 0.33  accuracy@1000: 0.96  min:train_loss@1000: 0.17
#     468717e  2026-06-25  accuracy@200: 0.93  min:train_loss@200: 0.23  accuracy@1000: 0.97  min:train_loss@1000: 0.09
```

The accuracy bar ratchets up across the two baselines (0.92 → 0.93 at step 200,
0.96 → 0.97 at step 1000): any future regression is now measured against the
better model.
