import numpy as np
import pytest

from simulator.ckp_model import generate_ckp


def test_constant_speed_produces_expected_60_2_pulse_count():
    sample_hz = 12000
    rpm = 1800
    duration_s = 1.0

    t = np.arange(0, duration_s, 1 / sample_hz)
    angle = 360.0 * rpm / 60.0 * t

    pulse = generate_ckp(angle, sample_hz)

    # 30 revolutions/sec * 58 physical teeth per revolution.
    expected = int(duration_s * rpm / 60.0 * 58)
    assert abs(int(pulse.sum()) - expected) <= 1


def test_missing_tooth_gap_is_longer_than_normal_interval():
    sample_hz = 12000
    rpm = 1800
    t = np.arange(0, 1.0, 1 / sample_hz)
    angle = 360.0 * rpm / 60.0 * t

    pulse = generate_ckp(angle, sample_hz)
    pulse_times = t[np.flatnonzero(pulse)]

    intervals = np.diff(pulse_times)
    nominal = 60.0 / (rpm * 60)

    # A 60-2 reference gap is approximately three nominal tooth pitches.
    assert np.max(intervals) > nominal * 2.5


def test_faster_rotation_reduces_tooth_interval():
    sample_hz = 12000
    t = np.arange(0, 1.0, 1 / sample_hz)

    slow_angle = 360.0 * 900 / 60.0 * t
    fast_angle = 360.0 * 1800 / 60.0 * t

    slow = generate_ckp(slow_angle, sample_hz)
    fast = generate_ckp(fast_angle, sample_hz)

    slow_times = t[np.flatnonzero(slow)]
    fast_times = t[np.flatnonzero(fast)]

    assert np.median(np.diff(fast_times)) < np.median(np.diff(slow_times))


def test_rejects_non_monotonic_crank_angle():
    with pytest.raises(ValueError):
        generate_ckp(np.array([0.0, 6.0, 5.0]), 4000)


def test_rejects_invalid_sampling_rate():
    with pytest.raises(ValueError):
        generate_ckp(np.array([0.0, 1.0, 2.0]), 0)
