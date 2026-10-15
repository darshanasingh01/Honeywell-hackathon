import numpy as np
import pytest

from simulator.signal_quality import (
    check_timestamp_spacing,
    inspect_signal,
)


def test_clean_signal_passes():
    result = inspect_signal(np.sin(np.arange(100)), 4000)
    assert result["quality_ok"]
    assert result["finite_fraction"] == 1.0
    assert result["nyquist_hz"] == 2000


def test_non_finite_values_are_flagged():
    result = inspect_signal([1.0, np.nan, 2.0], 100)
    assert "non_finite_values" in result["warnings"]
    assert result["finite_fraction"] < 1.0


def test_clipping_is_flagged():
    result = inspect_signal([0.1, 0.2, 1.0, -1.0], 100,
                            clipping_limit=0.95)
    assert "possible_clipping" in result["warnings"]


def test_unexpected_sampling_rate_is_flagged():
    result = inspect_signal([0.1, 0.2, 0.3], 100,
                            expected_sample_hz=200)
    assert "unexpected_sample_rate" in result["warnings"]


def test_invalid_sampling_rate_is_rejected():
    with pytest.raises(ValueError):
        inspect_signal([1, 2, 3], 0)


def test_regular_timestamps_pass():
    t = np.arange(100) / 4000
    result = check_timestamp_spacing(t, 4000)
    assert result["quality_ok"]
    assert result["uniform_spacing"]


def test_irregular_timestamps_are_flagged():
    t = np.array([0.0, 0.01, 0.02, 0.04])
    result = check_timestamp_spacing(t, 100)
    assert "irregular_timestamp_spacing" in result["warnings"]


def test_duplicate_timestamps_are_flagged():
    result = check_timestamp_spacing([0.0, 0.01, 0.01, 0.03], 100)
    assert "timestamps_not_strictly_increasing" in result["warnings"]
