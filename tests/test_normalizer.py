"""Tests for benchmark return-value normalization."""

import pytest

from pybench.normalizer import NormalizationError, normalize


def test_scalar_float():
    assert normalize(0.91) == {0: {"score": 0.91}}


def test_scalar_int():
    assert normalize(5) == {0: {"score": 5.0}}


def test_bool_rejected():
    with pytest.raises(NormalizationError):
        normalize(True)


def test_single_dict():
    assert normalize({"accuracy": 0.9, "min:loss": 0.4}) == {
        0: {"accuracy": 0.9, "min:loss": 0.4}
    }


def test_dict_non_string_key():
    with pytest.raises(NormalizationError):
        normalize({1: 0.5})


def test_dict_empty():
    with pytest.raises(NormalizationError):
        normalize({})


def test_dict_non_numeric_value():
    with pytest.raises(NormalizationError):
        normalize({"a": "x"})


def test_dict_bool_value():
    with pytest.raises(NormalizationError):
        normalize({"a": True})


def test_list_of_steps():
    out = normalize([{"step": 1, "f1": 0.5}, {"step": 2, "f1": 0.6}])
    assert out == {1: {"f1": 0.5}, 2: {"f1": 0.6}}


def test_list_empty():
    with pytest.raises(NormalizationError):
        normalize([])


def test_list_element_not_dict():
    with pytest.raises(NormalizationError):
        normalize([1])


def test_list_missing_step():
    with pytest.raises(NormalizationError):
        normalize([{"f1": 0.5}])


def test_list_step_bool():
    with pytest.raises(NormalizationError):
        normalize([{"step": True, "f1": 0.5}])


def test_list_step_non_int():
    with pytest.raises(NormalizationError):
        normalize([{"step": 1.5, "f1": 0.5}])


def test_list_duplicate_step():
    with pytest.raises(NormalizationError):
        normalize([{"step": 1, "f1": 0.5}, {"step": 1, "f1": 0.6}])


def test_list_step_only_has_no_metrics():
    with pytest.raises(NormalizationError):
        normalize([{"step": 1}])


def test_unsupported_type():
    with pytest.raises(NormalizationError):
        normalize("hello")


def test_none_rejected():
    with pytest.raises(NormalizationError):
        normalize(None)
