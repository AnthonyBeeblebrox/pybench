"""Click-based CLI entry point."""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from pybench.config import apply_overrides
from pybench.discovery import DiscoveryError, discover
from pybench.git import file_at_commit, file_history, git_metadata
from pybench.reporter import (
    BenchOutcome,
    report,
    report_history,
    report_show,
    report_update,
)
from pybench.runner import RunShapeError, run_benchmark, sample_seeds
from pybench.stats import check_alpha_detectable, compare
from pybench.store import (
    BaselineRecord,
    parse_baselines,
    read_baselines,
    write_baselines,
)
from pybench.validator import (
    MetricKeyMismatchError,
    StepKeyMismatchError,
    validate_alignment,
)

DEFAULT_PATH = "./benchmarks"
DEFAULT_BASELINE = ".pybench/baselines.jsonl"

SeedScores = dict[int, dict[str, list[float]]]


def _counts(scores: SeedScores) -> tuple[int, int]:
    """Return ``(n_steps, n_metrics)`` for a (non-empty) scores mapping."""
    first = next(iter(scores.values()))
    return len(scores), len(first)


def _make_progress(console: Console) -> Progress:
    """A transient per-benchmark progress bar (count + ETA) that clears itself."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


def _confirm_write(
    console: Console,
    *,
    dirty: bool | None,
    yes: bool,
    confirm_when_clean: bool,
    n: int,
) -> bool:
    """Decide whether a baseline write may proceed, prompting if needed.

    On a dirty tree (§2.2) a warning is always shown; without ``--yes`` the
    user is asked to confirm. On a clean tree, ``confirm_when_clean`` (used by
    ``update``, which overwrites) triggers an accident-guard prompt. ``--yes``
    skips every prompt.

    Args:
        console: Rich console for the warning text.
        dirty: Git dirty flag (``None`` when there is no git repo).
        yes: Whether ``--yes`` was passed.
        confirm_when_clean: Prompt even on a clean tree (overwrite guard).
        n: Number of benchmarks about to be written (for the prompt text).

    Returns:
        ``True`` to proceed with the write, ``False`` to abort.
    """
    if dirty:
        console.print(
            "[yellow]⚠  Working tree is dirty. The baseline will be saved but "
            "cannot be\n   reliably reproduced from this commit. Commit your "
            "changes first.[/yellow]"
        )
        if yes:
            return True
        return click.confirm("   Continue anyway?", default=False)
    if confirm_when_clean and not yes:
        return click.confirm(f"Overwrite baseline for {n} benchmark(s)?", default=False)
    return True


class _DefaultGroup(click.Group):
    """Click group that falls back to the ``_default`` command when no
    subcommand matches, so that ``pybench <args>`` runs benchmarks directly."""

    def invoke(self, ctx: click.Context) -> None:
        if not ctx._protected_args and not ctx.args and ctx.invoked_subcommand is None:
            return ctx.invoke(self.commands["_default"])
        return super().invoke(ctx)

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        if args and args[0] in self.commands:
            return super().resolve_command(ctx, args)
        return "", self.commands["_default"], args


@click.group(cls=_DefaultGroup, invoke_without_command=True)
def main() -> None:
    """Statistical benchmark regression detection (pybench)."""


@main.command("_default", hidden=True)
@click.argument("path", required=False, default=DEFAULT_PATH)
@click.option(
    "--baseline", "baseline_path", default=DEFAULT_BASELINE, help="Path to JSONL store."
)
@click.option("--bench", "benches", multiple=True, help="Run only this benchmark.")
@click.option("--alpha", type=float, default=None, help="Override alpha for all.")
@click.option(
    "--min-effect", type=float, default=None, help="Override min_effect for all."
)
@click.option("-v", "--verbose", is_flag=True, help="Expand failing breakdowns.")
@click.option("--yes", is_flag=True, help="Skip the dirty-tree prompt (for CI).")
def _default(
    path: str,
    baseline_path: str,
    benches: tuple[str, ...],
    alpha: float | None,
    min_effect: float | None,
    verbose: bool,
    yes: bool,
) -> None:
    """Discover benchmarks, compare against the baseline, and report."""
    console = Console()
    rng = np.random.default_rng()
    try:
        discovered = discover(Path(path), list(benches) or None)
    except DiscoveryError as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    baseline_file = Path(baseline_path)
    baselines = read_baselines(baseline_file)
    git = git_metadata()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        for bench in discovered:
            cfg = apply_overrides(bench.config, alpha=alpha, min_effect=min_effect)
            if bench.name in baselines:
                n = len(baselines[bench.name].seeds)
            else:
                n = cfg.n_seeds
            check_alpha_detectable(n, cfg.alpha)
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    new_names = [b.name for b in discovered if b.name not in baselines]
    if new_names and not _confirm_write(
        console, dirty=git.dirty, yes=yes, confirm_when_clean=False, n=len(new_names)
    ):
        console.print("Aborted; no baseline written.")
        sys.exit(0)

    outcomes: list[BenchOutcome] = []
    new_records: dict[str, BaselineRecord] = {}
    any_fail = False
    start = time.perf_counter()
    try:
        with _make_progress(console) as progress:
            for bench in discovered:
                cfg = apply_overrides(bench.config, alpha=alpha, min_effect=min_effect)
                if bench.name in baselines:
                    record = baselines[bench.name]
                    task = progress.add_task(bench.name, total=len(record.seeds))
                    current = run_benchmark(
                        bench, record.seeds, on_seed=lambda t=task: progress.advance(t)
                    )
                    validate_alignment(bench.name, current, record.scores)
                    cmp = compare(
                        record.scores,
                        current,
                        alpha=cfg.alpha,
                        min_effect=cfg.min_effect,
                        rng=rng,
                    )
                    status = "PASS" if cmp.passed else "FAIL"
                    any_fail = any_fail or not cmp.passed
                    n_steps, n_metrics = _counts(record.scores)
                    outcomes.append(
                        BenchOutcome(bench.name, status, n_steps, n_metrics, cmp)
                    )
                else:
                    seeds = sample_seeds(cfg.n_seeds, rng)
                    task = progress.add_task(bench.name, total=len(seeds))
                    scores = run_benchmark(
                        bench, seeds, on_seed=lambda t=task: progress.advance(t)
                    )
                    new_records[bench.name] = BaselineRecord(
                        bench=bench.name,
                        timestamp=timestamp,
                        git_commit=git.commit,
                        git_dirty=git.dirty,
                        seeds=seeds,
                        scores=scores,
                    )
                    n_steps, n_metrics = _counts(scores)
                    outcomes.append(
                        BenchOutcome(bench.name, "NEW", n_steps, n_metrics, None)
                    )
    except (StepKeyMismatchError, MetricKeyMismatchError, RunShapeError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)
    elapsed = time.perf_counter() - start

    if new_records:
        write_baselines(baseline_file, {**baselines, **new_records}.values())

    report(console, outcomes, elapsed=elapsed, verbose=verbose)
    sys.exit(1 if any_fail else 0)


@main.command()
@click.argument("path", required=False, default=DEFAULT_PATH)
@click.option(
    "--baseline", "baseline_path", default=DEFAULT_BASELINE, help="Path to JSONL store."
)
@click.option("--bench", "benches", multiple=True, help="Update only this benchmark.")
@click.option(
    "--n-seeds", type=int, default=None, help="Resample this many fresh seeds."
)
@click.option("--yes", is_flag=True, help="Skip confirmation prompts (for CI).")
def update(
    path: str,
    baseline_path: str,
    benches: tuple[str, ...],
    n_seeds: int | None,
    yes: bool,
) -> None:
    """Re-run benchmarks and overwrite their baseline records."""
    console = Console()
    rng = np.random.default_rng()
    try:
        discovered = discover(Path(path), list(benches) or None)
    except DiscoveryError as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    baseline_file = Path(baseline_path)
    baselines = read_baselines(baseline_file)
    git = git_metadata()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        for bench in discovered:
            n = bench.config.n_seeds if n_seeds is None else n_seeds
            check_alpha_detectable(n, bench.config.alpha)
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    if not _confirm_write(
        console, dirty=git.dirty, yes=yes, confirm_when_clean=True, n=len(discovered)
    ):
        console.print("Aborted; baseline unchanged.")
        sys.exit(0)

    records = dict(baselines)
    summary: list[tuple[str, int]] = []
    try:
        with _make_progress(console) as progress:
            for bench in discovered:
                n = bench.config.n_seeds if n_seeds is None else n_seeds
                seeds = sample_seeds(n, rng)
                task = progress.add_task(bench.name, total=n)
                scores = run_benchmark(
                    bench, seeds, on_seed=lambda t=task: progress.advance(t)
                )
                records[bench.name] = BaselineRecord(
                    bench=bench.name,
                    timestamp=timestamp,
                    git_commit=git.commit,
                    git_dirty=git.dirty,
                    seeds=seeds,
                    scores=scores,
                )
                summary.append((bench.name, n))
    except RunShapeError as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    write_baselines(baseline_file, records.values())
    report_update(console, summary)


@main.command()
@click.option(
    "--baseline", "baseline_path", default=DEFAULT_BASELINE, help="Path to JSONL store."
)
@click.option("--bench", "benches", multiple=True, help="Show only this benchmark.")
@click.option("--history", is_flag=True, help="Show baseline history from git.")
def show(baseline_path: str, benches: tuple[str, ...], history: bool) -> None:
    """Print current baseline stats, or their history with --history."""
    console = Console()
    baseline_file = Path(baseline_path)
    names = set(benches) or None
    if history:
        _show_history(console, baseline_file, names)
        return
    records = read_baselines(baseline_file)
    if names is not None:
        records = {k: v for k, v in records.items() if k in names}
    report_show(console, records)


def _show_history(
    console: Console, baseline_file: Path, names: set[str] | None
) -> None:
    """Collect per-commit baseline records and render them grouped by bench."""
    commits = file_history(baseline_file)
    if commits is None:
        console.print(
            "[red]error:[/red] not a git repository (or git unavailable); "
            "cannot show history."
        )
        return
    collected: dict[str, list[tuple[str, str, BaselineRecord]]] = {}
    for sha, date in commits:
        content = file_at_commit(sha, baseline_file)
        if content is None:
            continue
        for name, record in parse_baselines(content).items():
            if names is not None and name not in names:
                continue
            collected.setdefault(name, []).append((sha, date, record))
    report_history(console, collected)
