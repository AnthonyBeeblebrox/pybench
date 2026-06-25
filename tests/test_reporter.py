"""Tests for terminal reporting."""

import io

from rich.console import Console

from pybench.reporter import (
    BenchOutcome,
    report,
    report_history,
    report_show,
    report_update,
)
from pybench.stats import Comparison, SlotResult
from pybench.store import BaselineRecord


def _console():
    buf = io.StringIO()
    return Console(file=buf, width=200, force_terminal=False), buf


def _slot(metric, flagged, *, denom_at_floor=False):
    return SlotResult(
        step=0,
        metric=metric,
        baseline_mean=0.90,
        baseline_std=0.01,
        current_mean=0.80,
        current_std=0.02,
        effect_size=-0.11,
        p_value=0.0 if flagged else 0.5,
        flagged=flagged,
        denom_at_floor=denom_at_floor,
    )


def test_renders_new_single_and_multi():
    console, buf = _console()
    single = Comparison([_slot("score", False)], 0, 1, 0.6, True)
    multi = Comparison([_slot("min:loss", True), _slot("f1", False)], 1, 2, 0.01, False)
    report(
        console,
        [
            BenchOutcome("bench_new", "NEW", 2, 3, None),
            BenchOutcome("bench_s", "PASS", 1, 1, single),
            BenchOutcome("bench_m", "FAIL", 1, 2, multi),
        ],
        elapsed=1.2,
        verbose=False,
    )
    text = buf.getvalue()
    assert "3 benchmarks discovered" in text
    assert "NEW" in text and "(baseline saved)" in text
    assert "PASS" in text and "score" in text
    assert "FAIL" in text and "meta-p=0.010" in text
    assert "1 failed, 1 passed, 1 new" in text
    assert "✗" not in text


def test_verbose_expands_failing_table():
    console, buf = _console()
    multi = Comparison(
        [_slot("min:loss", True), _slot("f1", False, denom_at_floor=True)],
        1,
        2,
        0.01,
        False,
    )
    report(
        console,
        [BenchOutcome("bench_m", "FAIL", 1, 2, multi)],
        elapsed=1.0,
        verbose=True,
    )
    text = buf.getvalue()
    assert "baseline" in text and "current" in text
    assert "✗" in text
    assert "⚠" in text


def test_single_line_shows_denom_warning():
    console, buf = _console()
    single = Comparison([_slot("score", False, denom_at_floor=True)], 0, 1, 0.6, True)
    report(
        console,
        [BenchOutcome("bench_s", "PASS", 1, 1, single)],
        elapsed=1.0,
        verbose=False,
    )
    text = buf.getvalue()
    assert "effect unreliable" in text


def _record(dirty):
    return BaselineRecord(
        bench="bench_x",
        timestamp="2026-06-22T10:00:00Z",
        git_commit="a3f4c1d",
        git_dirty=dirty,
        seeds=[1, 2],
        scores={0: {"min:loss": [0.30, 0.32]}, 10: {"f1": [0.90, 0.88]}},
    )


def test_report_update():
    console, buf = _console()
    report_update(console, [("bench_a", 30), ("bench_b", 3)])
    text = buf.getvalue()
    assert "2 benchmark(s) rewritten" in text
    assert "bench_a" in text and "30 seeds" in text


def test_report_show_empty():
    console, buf = _console()
    report_show(console, {})
    assert "No baselines found." in buf.getvalue()


def test_report_show_with_dirty_flag():
    console, buf = _console()
    report_show(console, {"bench_x": _record(True)})
    text = buf.getvalue()
    assert "bench_x  a3f4c1d  2026-06-22" in text
    assert "2 seeds" in text
    assert "min:loss@0: 0.31" in text and "f1@10: 0.89" in text
    assert "⚠ dirty" in text


def test_report_history_empty():
    console, buf = _console()
    report_history(console, {})
    assert "No baseline history found." in buf.getvalue()


def test_report_history_with_commits():
    console, buf = _console()
    history = {
        "bench_x": [
            ("a3f4c1d", "2026-06-20", _record(False)),
            ("c1d5f8a", "2026-06-22", _record(True)),
        ]
    }
    report_history(console, history)
    text = buf.getvalue()
    assert "bench_x" in text
    assert "a3f4c1d  2026-06-20" in text
    assert "c1d5f8a  2026-06-22" in text
    assert "⚠ dirty" in text
