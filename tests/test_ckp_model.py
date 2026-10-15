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


@pytest.mark.parametrize(
    "signal_hz",
    [0, -1, 1.5, float("nan"), float("inf"), float("-inf"), True],
)
def test_rejects_invalid_sampling_rates(signal_hz):
    with pytest.raises(ValueError):
        generate_ckp(np.array([0.0, 1.0, 2.0]), signal_hz)


@pytest.mark.parametrize(
    ("nominal_teeth", "missing_teeth"),
    [(3, 1), (60, 0), (60, 59), (60, 60), (60, -1)],
)
def test_rejects_invalid_tooth_wheel_configuration(
    nominal_teeth, missing_teeth
):
    with pytest.raises(ValueError):
        generate_ckp(
            np.array([0.0, 1.0, 2.0]),
            4000,
            nominal_teeth=nominal_teeth,
            missing_teeth=missing_teeth,
        )


def test_supports_custom_tooth_wheel():
    angle = np.arange(0.0, 720.0, 1.0)
    pulse = generate_ckp(
        angle, 4000, nominal_teeth=36, missing_teeth=1
    )

    assert pulse.dtype == np.uint8
    assert set(np.unique(pulse)).issubset({0, 1})
    assert pulse.sum() > 0
