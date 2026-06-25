# CLI reference

`pybench` runs benchmarks by default (no subcommand needed), with `update` and
`show` subcommands for managing baselines.

## Default — discover & run benchmarks

```text
pybench [PATH] [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `PATH` | `str` | `./benchmarks` | Directory or file to scan for `bench_*` functions |
| `--baseline` | `str` | `.pybench/baselines.jsonl` | Path to the JSONL baseline store |
| `--bench` | `str` | — | Run only this benchmark (repeatable) |
| `--alpha` | `float` | — | Override `alpha` significance threshold |
| `--min-effect` | `float` | — | Override minimum relative drop to flag |
| `-v, --verbose` | flag | off | Expand per-metric per-step breakdown on failures |
| `--yes` | flag | off | Skip the dirty-tree confirmation prompt (for CI) |

Discovered benchmarks with an existing baseline are re-run on the same stored
seeds and compared statistically. Benchmarks without a baseline record are run,
saved, and marked NEW. Exit code is 1 if any benchmark FAILs (like pytest).

## Subcommands

```{eval-rst}
.. click:: pybench.cli:main
   :prog: pybench
   :nested: full
```
