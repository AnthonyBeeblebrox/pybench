# mnist — pybench example

A self-contained uv workspace package. Its JAX/Flax/`datasets` stack stays out
of the core `pybench` dependencies and is installed only when you run it.

- `src/mnist/` — library code (data loading, model).
- `benchmarks/bench_model.py` — the `bench_*` entry point pybench discovers; it
  imports the installed `mnist` package.

## Run

```sh
uv run --package mnist pybench examples/mnist/benchmarks/
```

The first run trains across the sampled seeds and saves a baseline; later runs
compare against it. See the project `SPECIFICATIONS.md` §10.
