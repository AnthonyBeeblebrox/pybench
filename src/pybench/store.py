"""Read and write the JSONL baseline store."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SeedScores = dict[int, dict[str, list[float]]]
"""Per-seed raw scores: ``{step: {metric: [value for each seed]}}``."""


@dataclass(frozen=True)
class BaselineRecord:
    """One benchmark's stored baseline."""

    bench: str
    timestamp: str
    git_commit: str | None
    git_dirty: bool | None
    seeds: list[int]
    scores: SeedScores


def _scores_to_json(scores: SeedScores) -> dict[str, dict[str, list[float]]]:
    """Serialize integer step keys to strings for JSON."""
    return {str(step): metrics for step, metrics in scores.items()}


def _scores_from_json(raw: dict[str, dict[str, list[float]]]) -> SeedScores:
    """Parse JSON scores, restoring integer step keys and float values."""
    return {
        int(step): {m: [float(v) for v in vals] for m, vals in metrics.items()}
        for step, metrics in raw.items()
    }


def parse_baselines(text: str) -> dict[str, BaselineRecord]:
    """Parse JSONL baseline content into records keyed by benchmark name.

    Args:
        text: Raw JSONL content (e.g. a file's text or ``git show`` output).

    Returns:
        Mapping of benchmark name to record; blank lines are skipped.
    """
    records: dict[str, BaselineRecord] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        record = BaselineRecord(
            bench=obj["bench"],
            timestamp=obj["timestamp"],
            git_commit=obj["git_commit"],
            git_dirty=obj["git_dirty"],
            seeds=[int(s) for s in obj["seeds"]],
            scores=_scores_from_json(obj["scores"]),
        )
        records[record.bench] = record
    return records


def read_baselines(path: Path) -> dict[str, BaselineRecord]:
    """Load all baseline records keyed by benchmark name.

    Args:
        path: Path to the JSONL store.

    Returns:
        Mapping of benchmark name to record; empty if the file is absent.
    """
    if not path.exists():
        return {}
    return parse_baselines(path.read_text())


def write_baselines(path: Path, records: Iterable[BaselineRecord]) -> None:
    """Rewrite the JSONL store with the given records, one line each.

    Args:
        path: Path to the JSONL store; parent directories are created.
        records: Records to write (full rewrite).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        obj = {
            "bench": record.bench,
            "timestamp": record.timestamp,
            "git_commit": record.git_commit,
            "git_dirty": record.git_dirty,
            "seeds": record.seeds,
            "n": len(record.seeds),
            "scores": _scores_to_json(record.scores),
        }
        lines.append(json.dumps(obj))
    path.write_text("".join(line + "\n" for line in lines))
