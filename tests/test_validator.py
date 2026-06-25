"""Tests for step/metric alignment validation."""

import pytest

from pybench.validator import (
    MetricKeyMismatchError,
    StepKeyMismatchError,
    validate_alignment,
)


def test_aligned_ok():
    cur = {0: {"a": 1.0, "b": 2.0}}
    base = {0: {"a": 1.0, "b": 2.0}}
    validate_alignment("bench_x", cur, base)


def test_step_mismatch():
    cur = {0: {"a": 1.0}, 1: {"a": 1.0}}
    base = {0: {"a": 1.0}}
    with pytest.raises(StepKeyMismatchError) as excinfo:
        validate_alignment("bench_x", cur, base)
    assert excinfo.value.bench == "bench_x"
    assert "bench_x" in str(excinfo.value)


def test_metric_mismatch():
    cur = {0: {"a": 1.0}}
    base = {0: {"a": 1.0, "b": 2.0}}
    with pytest.raises(MetricKeyMismatchError) as excinfo:
        validate_alignment("bench_x", cur, base)
    assert excinfo.value.step == 0
    assert "bench_x" in str(excinfo.value)
