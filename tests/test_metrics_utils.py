import pytest

from tools.metrics_utils import percentile


def test_percentile_basic():
    data = [1, 2, 3, 4]
    assert percentile(data, 0) == 1
    assert percentile(data, 100) == 4
    assert percentile(data, 50) == pytest.approx(2.5)
    assert percentile(data, 25) == pytest.approx(1.75)


def test_percentile_singleton():
    assert percentile([42], 0) == 42
    assert percentile([42], 100) == 42
    assert percentile([42], 50) == 42


def test_percentile_invalid():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1, 2, 3], -1)
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)


def test_percentile_interpolation():
    # Designed to exercise fractional interpolation path where f != c
    data = [10, 20, 30]
    # For 25th: k = 0.5 -> interpolation between 10 and 20
    assert percentile(data, 25) == pytest.approx(15)
    # For 75th: k = 1.5 -> interpolation between 20 and 30
    assert percentile(data, 75) == pytest.approx(25)


def test_percentile_exact_index():
    # For data length 5, 50th percentile -> k=(4*0.5)=2.0 exact index
    data = [5, 1, 9, 3, 7]
    assert percentile(data, 50) == sorted(data)[2]
