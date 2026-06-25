# pybench

While Pytest does deterministic tests, Pybench does statistical tests.

`pybench` discovers benchmark functions, runs them across many seeds, and
**statistically detects regressions** against a saved baseline.

It reruns each benchmark on the *same* stored seeds as its baseline, so the
comparison is **paired** (far more sensitive than a two-sample test), and judges
the whole benchmark with a within-seed sign-flip permutation test that respects
correlation across metrics and steps.

```{toctree}
:maxdepth: 2

getting_started
user_guide
how_it_works
cli
```
