# Getting started

## What is pybench?

`pybench` is a CLI tool that catches **performance regressions** in code whose
output is a random number rather than a fixed value: model accuracy, a loss
curve, a solver's score. Such metrics are *noisy*: they shift from run to run
because of random seeds, so a single before/after comparison can't tell a real
regression from luck.

pybench removes the noise the right way:

- **Discovery** Any function named `bench_*` that takes a `seed` and returns a
  score is a benchmark. No config file, no registration.
- **Paired by construction.** The seeds sampled for the first (baseline) run are
  *stored*; every later run reuses the *same* seeds. Comparing identical
  conditions cancels seed-to-seed variance, so far fewer seeds detect the same
  regression.
- **An honest verdict.** Each `(step, metric)` slot gets a one-sided paired
  t-test, and the whole benchmark is judged by a within-seed sign-flip
  permutation test. That test makes no independence assumption, so correlated
  metrics and steps don't inflate false alarms.
- **CI-native.** `pybench` exits non-zero when any benchmark regresses, so it
  drops straight into CI like `pytest`.

A benchmark returns one of three shapes:

```python
def bench_a(seed): return 0.91                                   # scalar
def bench_b(seed): return {"accuracy": 0.91, "min:loss": 0.42}  # multiple metrics
def bench_c(seed):                                              # multi-step curve
    return [{"step": 1, "accuracy": 0.5, "min:loss": 1.0}, {"step": 10, "accuracy": 0.91, "min:loss": 0.42}]
```

Scores follow a **higher-is-better** convention. For metrics where lower is
better (loss, error), prefix the key with `min:` and return the **raw** value —
pybench flips the sign internally so that "a decrease in goodness is a
regression."

## Quickstart

Install pybench:

```bash
uv add pybench        # or: pip install pybench
```

Write a `bench_*` function that takes a `seed` and returns a score:

```python
# benchmarks/bench_model.py
def bench_accuracy(seed: int) -> float:
    return train_and_score(seed)
```

Then drive it from the CLI:

```bash
pybench            # 1st time: samples seeds, saves a baseline, marks NEW
pybench            # later: reruns on the same seeds, marks PASS / FAIL
pybench update         # re-baseline after an intended change
pybench show           # print current baseline stats  (--history for per-commit)
```

The three ways to invoke pybench:

| Command | What it does | Writes to disk? |
|---------|--------------|-----------------|
| `pybench` | Discover, run, compare; exit 1 if any benchmark fails | Only the first time (baseline init) |
| `pybench update` | Re-run and overwrite the baseline (resamples fresh seeds) | Yes |
| `pybench show` | Print current baseline stats (`--history` for per-commit) | No |

Per-benchmark settings are keyword-only defaults — no config file:

```python
def bench_training(seed: int, *, n_seeds: int = 50, alpha: float = 0.01,
                   min_effect: float = 0.02, workers: int = 4) -> list[dict]:
    ...
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_seeds` | `30` | Seeds sampled for the baseline |
| `alpha` | `0.05` | Significance threshold |
| `min_effect` | `None` | Minimum relative drop to flag (suppress trivia) |
| `workers` | `1` | Parallel seed processes (keep `1` for GPU/serial) |

The baseline lives at `.pybench/baselines.jsonl` (one line per benchmark).
**Commit it to git — do not gitignore it.** History is delegated to git, and
`pybench show --history` reconstructs the baseline at every commit that touched
it.
