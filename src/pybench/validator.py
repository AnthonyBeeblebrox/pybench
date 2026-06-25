"""Alignment checks between a run's scores and the stored baseline."""

from __future__ import annotations

from collections.abc import Mapping

_RESET_HINT = "Re-run `pybench update` to reset."


class StepKeyMismatchError(ValueError):
    """Raised when the run and baseline step-key sets differ."""

    def __init__(self, bench: str, current: set[int], baseline: set[int]) -> None:
        """Record the mismatch and build a human-readable message.

        Args:
            bench: Benchmark name.
            current: Step keys produced by the current run.
            baseline: Step keys stored in the baseline.
        """
        self.bench = bench
        self.current = current
        self.baseline = baseline
        msg = (
            f"{bench} returned steps {sorted(current)} but baseline has steps "
            f"{sorted(baseline)}. {_RESET_HINT}"
        )
        super().__init__(msg)


class MetricKeyMismatchError(ValueError):
    """Raised when the run and baseline metric-key sets differ for a step."""

    def __init__(
        self, bench: str, step: int, current: set[str], baseline: set[str]
    ) -> None:
        """Record the mismatch and build a human-readable message.

        Args:
            bench: Benchmark name.
            step: The step at which the metric sets diverged.
            current: Metric keys produced by the current run at ``step``.
            baseline: Metric keys stored in the baseline at ``step``.
        """
        self.bench = bench
        self.step = step
        self.current = current
        self.baseline = baseline
        msg = (
            f"{bench} returned metrics {sorted(current)} but baseline has metrics "
            f"{sorted(baseline)} at step {step}. {_RESET_HINT}"
        )
        super().__init__(msg)


def validate_alignment(
    bench: str,
    current: Mapping[int, Mapping[str, object]],
    baseline: Mapping[int, Mapping[str, object]],
) -> None:
    """Assert the run and baseline share identical step and metric keys.

    Args:
        bench: Benchmark name, for error messages.
        current: Freshly normalized run scores (keyed by step then metric).
        baseline: Stored baseline scores (keyed by step then metric).

    Raises:
        StepKeyMismatchError: If the step-key sets differ.
        MetricKeyMismatchError: If any step's metric-key sets differ.
    """
    if set(current) != set(baseline):
        raise StepKeyMismatchError(bench, set(current), set(baseline))
    for step in current:
        if set(current[step]) != set(baseline[step]):
            raise MetricKeyMismatchError(
                bench, step, set(current[step]), set(baseline[step])
            )
