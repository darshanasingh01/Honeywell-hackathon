import numpy as np
import pytest

from simulator.engine_simulator import CONDITIONS, simulate_engine


def test_all_conditions_generate_expected_labels():
    for condition in CONDITIONS:
        telemetry, waveform = simulate_engine(
            condition, duration_s=2, seed=7
        )
        assert set(telemetry["condition"]) == {condition}
        assert set(waveform["condition"]) == {condition}


def test_expected_sensor_columns_exist():
    telemetry, waveform = simulate_engine(
        "normal", duration_s=2, seed=7
    )
    assert {
        "timestamp_s", "engine_rpm", "coolant_temp_c",
        "throttle_pct", "engine_load", "stft_pct", "ltft_pct",
        "condition",
    }.issubset(telemetry.columns)

    assert {
        "timestamp_s", "microphone_audio", "knock_sensor",
        "crank_pulse", "condition",
    }.issubset(waveform.columns)


def test_sampling_rates_and_timestamps():
    telemetry, waveform = simulate_engine(
        "normal", duration_s=2, telemetry_hz=10,
        signal_hz=4000, seed=7
    )
    assert len(telemetry) == 20
    assert len(waveform) == 8000
    assert np.all(np.diff(telemetry["timestamp_s"]) > 0)
    assert np.all(np.diff(waveform["timestamp_s"]) > 0)


def test_seed_makes_simulation_reproducible():
    a, aw = simulate_engine("knock", duration_s=1, seed=123)
    b, bw = simulate_engine("knock", duration_s=1, seed=123)
    assert a.equals(b)
    assert aw.equals(bw)


def test_invalid_condition_is_rejected():
    with pytest.raises(ValueError):
        simulate_engine("broken_engine")


def test_invalid_sampling_rate_is_rejected():
    with pytest.raises(ValueError):
        simulate_engine("normal", telemetry_hz=0)
