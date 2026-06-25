"""Terminal output for a benchmark run (Rich, colored)."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from pybench.stats import Comparison
from pybench.store import BaselineRecord, SeedScores

_STATUS_COLOR = {"NEW": "yellow", "PASS": "green", "FAIL": "red"}
_RULE_WIDTH = 62


@dataclass(frozen=True)
class BenchOutcome:
    """One benchmark's result, ready to render."""

    name: str
    status: str  # "NEW" | "PASS" | "FAIL"
    n_steps: int
    n_metrics: int
    comparison: Comparison | None


def _detail(outcome: BenchOutcome) -> str:
    """Build the right-hand summary text for a benchmark line."""
    if outcome.comparison is None:
        return (
            f"{outcome.n_metrics} metrics × {outcome.n_steps} steps   (baseline saved)"
        )
    cmp = outcome.comparison
    if cmp.n_slots == 1:
        slot = cmp.slots[0]
        warn = "  [yellow]⚠ effect unreliable[/yellow]" if slot.denom_at_floor else ""
        return (
            f"{slot.metric}  {slot.baseline_mean:.3f}→{slot.current_mean:.3f}  "
            f"Δ{slot.effect_size * 100:+.1f}%  p={slot.p_value:.2f}{warn}"
        )
    return (
        f"{outcome.n_metrics} metrics × {outcome.n_steps} steps   "
        f"{cmp.n_flagged}/{cmp.n_slots} slots flagged  meta-p={cmp.meta_p:.3f}"
    )


def _print_verbose(console: Console, cmp: Comparison) -> None:
    """Print the per-slot breakdown table under a failing benchmark."""
    console.print("  metric     step   baseline    current      Δ        p")
    for slot in cmp.slots:
        mark = "  [red]✗[/red]" if slot.flagged else ""
        warn = "  [yellow]⚠[/yellow]" if slot.denom_at_floor else ""
        console.print(
            f"  {slot.metric:<9} {slot.step:>5}   "
            f"{slot.baseline_mean:.2f}±{slot.baseline_std:.2f}   "
            f"{slot.current_mean:.2f}±{slot.current_std:.2f}   "
            f"{slot.effect_size * 100:+.1f}%   {slot.p_value:.3f}{mark}{warn}"
        )


def report(
    console: Console,
    outcomes: list[BenchOutcome],
    *,
    elapsed: float,
    verbose: bool,
) -> None:
    """Render the full run report.

    Args:
        console: Rich console to write to.
        outcomes: One outcome per benchmark, in display order.
        elapsed: Wall-clock seconds for the whole run.
        verbose: Expand the per-slot table under each failing benchmark.
    """
    console.print(f"pybench — {len(outcomes)} benchmarks discovered\n")
    for outcome in outcomes:
        color = _STATUS_COLOR[outcome.status]
        console.print(
            f"{outcome.name:<18}{'.' * 10} [{color}]{outcome.status}[/{color}]   "
            f"{_detail(outcome)}"
        )
        if verbose and outcome.status == "FAIL" and outcome.comparison is not None:
            _print_verbose(console, outcome.comparison)
    failed = sum(o.status == "FAIL" for o in outcomes)
    passed = sum(o.status == "PASS" for o in outcomes)
    new = sum(o.status == "NEW" for o in outcomes)
    console.print("─" * _RULE_WIDTH)
    console.print(f"{failed} failed, {passed} passed, {new} new  in {elapsed:.1f}s")


def _slot_means(scores: SeedScores) -> str:
    """Format ``metric@step: mean`` for every slot, in sorted order."""
    parts: list[str] = []
    for step in sorted(scores):
        for metric in sorted(scores[step]):
            values = scores[step][metric]
            parts.append(f"{metric}@{step}: {sum(values) / len(values):.2f}")
    return "  ".join(parts)


def report_update(console: Console, updated: list[tuple[str, int]]) -> None:
    """Render the summary of a ``pybench update``.

    Args:
        console: Rich console to write to.
        updated: ``(name, n_seeds)`` for each rewritten benchmark.
    """
    console.print(f"pybench update — {len(updated)} benchmark(s) rewritten\n")
    for name, n_seeds in updated:
        console.print(f"{name:<18} [green]updated[/green]   {n_seeds} seeds")


def report_show(console: Console, records: dict[str, BaselineRecord]) -> None:
    """Render the current baseline stats for each benchmark.

    Args:
        console: Rich console to write to.
        records: Baseline records keyed by benchmark name.
    """
    if not records:
        console.print(
            "No baselines found. Run [bold]pybench[/bold] to create one, "
            "or [bold]pybench update --yes[/bold] to (re)baseline."
        )
        return
    for name in sorted(records):
        record = records[name]
        dirty = "  [yellow]⚠ dirty[/yellow]" if record.git_dirty else ""
        commit = record.git_commit or "—"
        n_seeds = len(record.seeds)
        console.print(
            f"{name}  {commit}  {record.timestamp[:10]}  {n_seeds} seeds{dirty}"
        )
        console.print(f"  {_slot_means(record.scores)}")


def report_history(
    console: Console, history: dict[str, list[tuple[str, str, BaselineRecord]]]
) -> None:
    """Render per-benchmark baseline history across commits.

    Args:
        console: Rich console to write to.
        history: ``{bench: [(short_sha, date, record), ...]}`` chronological.
    """
    if not history:
        console.print("No baseline history found.")
        return
    for name in sorted(history):
        console.print(name)
        for sha, date, record in history[name]:
            dirty = "  [yellow]⚠ dirty[/yellow]" if record.git_dirty else ""
            console.print(f"  {sha}  {date}  {_slot_means(record.scores)}{dirty}")
