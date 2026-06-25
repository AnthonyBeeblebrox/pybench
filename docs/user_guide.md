# User guide

pybench ships two example benchmarks you can run yourself.

Start with the **synthetic example** — a fast, dependency-light loss-curve
benchmark that runs in seconds. It is the quickest way to watch pybench save a
baseline and then report a PASS or a FAIL, and it doubles as a hands-on
demonstration of *why* pybench's statistics hold up where simpler tests don't.

Then see the **MNIST example** — a real neural-network training run, where
pybench catches a regression in a genuinely noisy metric and walks through the
full baseline → regression → re-baseline lifecycle.

```{toctree}
:maxdepth: 1

synthetic
mnist
```
