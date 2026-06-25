# pybench

Discover benchmark functions, run them across many seeds, and **statistically
detect regressions** against a saved baseline.

`pybench` reruns each benchmark on the *same* stored seeds as its baseline, so
the comparison is **paired** (far more sensitive than a two-sample test), and
judges the whole benchmark with a within-seed sign-flip permutation test that
respects correlation across metrics and steps.


Docs: [pybench.readthedocs.io](https://pybench.readthedocs.io)

## Install

```bash
uv add git+https://github.com/AnthonyBeeblebrox/pybench    # or: pip install git+https://github.com/AnthonyBeeblebrox/pybench
```

## Quickstart

Write a `bench_*` function that takes a `seed` and returns a score (higher is
better; prefix lower-is-better metrics with `min:`):

```python
# benchmarks/bench_model.py
def bench_accuracy(seed: int) -> float:
    return train_and_score(seed)        # a float, or a dict, or a list[dict] of steps
```

```bash
pybench            # 1st time: samples seeds, saves a baseline, marks NEW
pybench            # later: reruns on the same seeds, marks PASS / FAIL (exit 1 on fail)
pybench update --yes   # re-baseline after an intended change
pybench show           # print current baseline stats  (--history for per-commit history)
```

`pybench` exits non-zero when any benchmark regresses, so it drops straight
into CI like `pytest`.

## Return formats

```python
def bench_a(seed): return 0.91                                   # scalar
def bench_b(seed): return {"accuracy": 0.91, "min:loss": 0.42}  # multiple metrics
def bench_c(seed):                                              # multi-step curve
    return [{"step": 1, "min:loss": 0.9}, {"step": 10, "min:loss": 0.3}]
```

## Configuration

Per-benchmark settings are keyword-only defaults — no config file:

```python
def bench_training(seed: int, *, n_seeds: int = 50, alpha: float = 0.01,
                   min_effect: float = 0.02, workers: int = 4) -> list[dict]:
    ...
```

| Parameter    | Default | Meaning                                              |
|--------------|---------|------------------------------------------------------|
| `n_seeds`    | `30`    | Seeds sampled for the baseline                       |
| `alpha`      | `0.05`  | Significance threshold                               |
| `min_effect` | `None`  | Minimum relative drop to flag (suppress trivia)      |
| `workers`    | `1`     | Parallel seed processes (keep `1` for GPU/serial)    |

## Commit your baseline

The baseline lives at `.pybench/baselines.jsonl` (one line per benchmark).
**Commit it to git — do not gitignore it.** History is delegated to git: commit
the file after each `pybench update`, and `pybench show --history` reconstructs
the baseline at every commit that touched it.
