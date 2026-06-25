# How it works

This page explains, from the ground up, how pybench turns a pile of noisy numbers
into a single PASS/FAIL. There are only three ideas: **pairing**, a **per-slot
t-test**, and a **sign-flip permutation of a severity statistic**.

## The problem: the metric is noisy

A benchmark returns a *random* number — accuracy, a loss, a score — that shifts
from run to run because of the random seed. Run it once before a change and once
after, and you can't tell a real regression from luck. The usual fix is to run
many seeds and compare distributions, but a naive comparison needs a lot of
seeds to see through the noise.

## Idea 1 — pairing cancels the noise

pybench samples a set of seeds *once*, stores them in the baseline, and reuses
the **same seeds** on every later run. So for each seed `s` it has a baseline
value and a current value computed under *identical* conditions. It works with
the per-seed **difference**

```
d_s = goodness(current_s) − goodness(baseline_s)
```

Pairing subtracts away the seed-to-seed variance that both runs share, leaving
only the effect of the code change. That is why pybench detects a regression with
far fewer seeds than an unpaired two-sample test would need.

**Goodness, and the `min:` convention.** pybench's convention is *higher is
better*. For a metric where lower is better (a loss), you prefix its key with
`min:` and return the raw value; pybench negates it internally so that
`goodness = −loss`. A regression is then always a **drop in goodness**,
whichever direction the raw metric moves.

## Idea 2 — a per-slot paired t-test

A benchmark can report several metrics at several steps. Each `(step, metric)`
pair is a **slot**. For each slot, pybench has one difference `d_s` per seed and
runs a **one-sided paired t-test** of "mean difference < 0" (a regression):

```
t_stat = mean(d_s) / (std(d_s) / √n)
```

A very negative `t_stat` means a confident drop. The critical value
`t_crit = t.ppf(alpha, n−1)` (itself negative) is the threshold; a slot whose
`t_stat` falls below `t_crit` has individually regressed at level `alpha`.

That is enough to judge *one* slot. The hard part is combining many.

## Idea 3 — combine slots without assuming independence

The slots are **not independent**: one seed produces correlated values across
steps and across metrics (a lucky seed lifts the whole curve). Two tempting
shortcuts both break on exactly this correlation — and the
[synthetic example](synthetic.md#part-2-why-the-severity-permutation-rigorously)
measures them failing:

- a **global t-test** that pools every `(seed, step)` difference assumes they are
  all independent, so correlation shrinks its standard error and it raises false
  alarms;
- a **per-step t-test + binomial** on the *count* of significant steps assumes
  the counts are independent `Bernoulli(alpha)`, which correlation also violates.

pybench instead asks the data what "no change" looks like, by **permutation**.

### The severity statistic

First it reduces the whole benchmark to one number, the **severity** — how deep,
in total, the slots fall into the rejection region:

```
T = Σ_slots max(0, t_crit − t_stat)
```

A slot contributes only if it crossed `t_crit`, and contributes *more* the deeper
it went. This is the crucial choice: a **count** of flagged slots would treat one
catastrophic regression the same as one marginal blip. By keeping the magnitude,
severity stays sensitive to a single sharply-regressed slot — the very case the
synthetic example shows a *count*-based permutation misses.

### The sign-flip permutation

Under the null "the change had no effect," the baseline and current runs are
interchangeable for each seed, so `+d_s` and `−d_s` are equally likely. pybench
builds the null distribution of `T` by flipping the sign of **whole per-seed
difference vectors** — one coin flip per seed, never per slot, so the cross-slot
correlation is preserved:

```text
T = severity(D)                       # observed, D is (n_seeds, n_slots)
for _ in range(n_perm):               # n_perm ≈ 5000
    signs = random ±1, one per seed   # shape (n_seeds, 1)
    T_b = severity(signs * D)
meta_p = (#{ T_b ≥ T } + 1) / (n_perm + 1)
```

Because it compares `T` to *its own* permutation null, this is **exactly
calibrated for any correlation structure** — no independence assumption anywhere.
The benchmark **FAILS** when `meta_p < alpha`.

For a single slot (`K = 1`) the severity is monotone in `−t_stat`, so the
sign-flip reduces exactly to the one-sided paired t-test — the simplest case
falls out for free.

## Putting it together

```
sample/reuse seeds  →  run baseline & current on the same seeds
   →  per-seed goodness differences d_s per slot
   →  per-slot paired t-test  →  severity  T = Σ max(0, t_crit − t_stat)
   →  within-seed sign-flip permutation  →  meta_p
   →  FAIL if meta_p < alpha
```

Every step is plain arithmetic over the differences pybench already has, so the
statistics add negligible cost on top of running the benchmark itself. The full
rationale, including the alternatives pybench rejected, is in `SPECIFICATIONS.md`
§3.
