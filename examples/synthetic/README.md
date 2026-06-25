# synthetic — pybench example

A self-contained uv workspace package. It needs only numpy + scipy, so it runs
in seconds and makes a good first benchmark to try.

- `src/synthetic/` — library code (the loss-curve sampler, `sample_loss_curves`).
- `benchmarks/bench_synthetic.py` — the `bench_*` entry point pybench discovers;
  it imports the installed `synthetic` package. Set `PYBENCH_SYNTHETIC_REGRESS` to
  `global` or `local` to inject a regression for the CLI walkthrough.
- `main.py` — the rigorous companion (numpy + scipy) showing *why* pybench
  permutes a severity statistic (§3). It does **not** reimplement pybench's
  statistics: the pybench verdict calls pybench's own `_severity` and
  `_sign_flip_meta_p`; only the competing alternatives (a global t-test, a
  per-step t-test + binomial, and a sign-flip on a flagged *count*) are written
  out here. They either inflate their false-positive rate on correlated steps or
  miss a regression hiding in a single checkpoint, while pybench's test does
  neither.

## Part 1 — catch regressions with the CLI

```sh
uv run --package synthetic pybench examples/synthetic/benchmarks/      # NEW, then PASS
PYBENCH_SYNTHETIC_RESAMPLE=1      uv run --package synthetic pybench examples/synthetic/benchmarks/  # PASS (resampled, no regression)
PYBENCH_SYNTHETIC_REGRESS=global  uv run --package synthetic pybench examples/synthetic/benchmarks/  # FAIL
PYBENCH_SYNTHETIC_REGRESS=local   uv run --package synthetic pybench examples/synthetic/benchmarks/  # FAIL
```

The first run samples seeds and saves a baseline; later runs compare against it.

## Part 2 — the rigorous comparison

```sh
uv run --package synthetic python examples/synthetic/main.py
```

See the project `SPECIFICATIONS.md` §3 (the statistics) and §10 (examples).
