"""Coerce benchmark return values into a canonical scores mapping."""

from __future__ import annotations

from typing import Any, cast

Scores = dict[int, dict[str, float]]
"""Canonical normalized form: ``{step: {metric: value}}``."""


class NormalizationError(ValueError):
    """Raised when a benchmark return value has an unsupported shape."""


def _coerce_value(metric: str, value: object) -> float:
    """Coerce a single metric value to ``float`` or raise.

    Args:
        metric: Metric name, for error messages.
        value: The raw value returned by the benchmark.

    Returns:
        The value as a ``float``.

    Raises:
        NormalizationError: If the value is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"metric {metric!r} has non-numeric value {value!r}"
        raise NormalizationError(msg)
    return float(value)


def _coerce_metrics(record: dict[Any, Any], *, drop_step: bool) -> dict[str, float]:
    """Coerce a metric mapping, optionally dropping the reserved ``step`` key.

    Args:
        record: Mapping of metric name to value.
        drop_step: When true, ignore a ``step`` key (used for list records).

    Returns:
        Mapping of metric name to ``float`` value.

    Raises:
        NormalizationError: On a non-string key or an empty metric set.
    """
    metrics: dict[str, float] = {}
    for key, value in record.items():
        if not isinstance(key, str):
            msg = f"metric key must be a string, got {key!r}"
            raise NormalizationError(msg)
        if drop_step and key == "step":
            continue
        metrics[key] = _coerce_value(key, value)
    if not metrics:
        raise NormalizationError("benchmark returned no metrics")
    return metrics


def _normalize_steps(records: list[Any]) -> Scores:
    """Normalize a ``list[dict]`` with mandatory integer ``step`` keys.

    Args:
        records: The list returned by a multi-step benchmark.

    Returns:
        Canonical ``{step: {metric: value}}`` scores.

    Raises:
        NormalizationError: On an empty list, a non-dict record, a missing or
            non-integer ``step``, or a duplicate step.
    """
    if not records:
        raise NormalizationError("benchmark returned an empty list of steps")
    scores: Scores = {}
    for record in records:
        if not isinstance(record, dict):
            msg = f"each step record must be a dict, got {record!r}"
            raise NormalizationError(msg)
        step = cast(dict[Any, Any], record).get("step")
        if step is None:
            raise NormalizationError("step record is missing the 'step' key")
        if isinstance(step, bool) or not isinstance(step, int):
            msg = f"'step' must be an int, got {step!r}"
            raise NormalizationError(msg)
        if step in scores:
            msg = f"duplicate step key {step}"
            raise NormalizationError(msg)
        scores[step] = _coerce_metrics(cast(dict[Any, Any], record), drop_step=True)
    return scores


def normalize(result: object) -> Scores:
    """Coerce any accepted benchmark return value to canonical ``Scores``.

    Args:
        result: A ``float``, ``dict`` of metrics, or ``list`` of step dicts.

    Returns:
        ``{step: {metric: value}}``; scalars and bare dicts use step ``0``, and a
        bare scalar is stored under the metric name ``score``.

    Raises:
        NormalizationError: If the value has an unsupported shape or type.
    """
    if isinstance(result, bool):
        raise NormalizationError("benchmark returned a bool, expected a score")
    if isinstance(result, (int, float)):
        return {0: {"score": float(result)}}
    if isinstance(result, dict):
        return {0: _coerce_metrics(cast(dict[Any, Any], result), drop_step=False)}
    if isinstance(result, list):
        return _normalize_steps(cast(list[Any], result))
    msg = f"unsupported benchmark return type: {type(result).__name__}"
    raise NormalizationError(msg)
