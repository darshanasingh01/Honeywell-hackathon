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


def test_knock_has_elevated_knock_sensor_rms():
    _, normal = simulate_engine("normal", seed=42)
    _, knock = simulate_engine("knock", seed=42)

    normal_rms = np.sqrt(np.mean(normal["knock_sensor"].to_numpy() ** 2))
    knock_rms = np.sqrt(np.mean(knock["knock_sensor"].to_numpy() ** 2))

    assert knock_rms > normal_rms * 1.5


def test_overheating_has_rising_temperature():
    normal, overheating = (
        simulate_engine(condition, seed=42)[0]
        for condition in ("normal", "overheating")
    )

    assert overheating["coolant_temp_c"].iloc[-1] > (
        normal["coolant_temp_c"].iloc[-1] + 5
    )
    assert overheating["coolant_temp_c"].iloc[-1] > (
        overheating["coolant_temp_c"].iloc[0]
    )


def test_misfire_has_attenuated_audio_rms():
    _, normal = simulate_engine("normal", seed=42)
    _, misfire = simulate_engine("misfire", seed=42)

    normal_rms = np.sqrt(np.mean(normal["microphone_audio"].to_numpy() ** 2))
    misfire_rms = np.sqrt(np.mean(misfire["microphone_audio"].to_numpy() ** 2))

    assert misfire_rms < normal_rms


def test_valve_wear_has_elevated_vibration_rms():
    _, normal = simulate_engine("normal", seed=42)
    _, worn = simulate_engine("valve_train_wear", seed=42)

    normal_rms = np.sqrt(np.mean(normal["knock_sensor"].to_numpy() ** 2))
    worn_rms = np.sqrt(np.mean(worn["knock_sensor"].to_numpy() ** 2))

    assert worn_rms > normal_rms * 1.5


def test_accelerometer_axes_exist_and_are_finite():
    _, waveform = simulate_engine("valve_train_wear", seed=42)

    for axis in ("accel_x", "accel_y", "accel_z"):
        assert axis in waveform.columns
        assert np.isfinite(waveform[axis].to_numpy()).all()


def test_valve_wear_increases_accelerometer_x_rms():
    _, normal = simulate_engine("normal", seed=42)
    _, worn = simulate_engine("valve_train_wear", seed=42)

    normal_rms = np.sqrt(np.mean(normal["accel_x"].to_numpy() ** 2))
    worn_rms = np.sqrt(np.mean(worn["accel_x"].to_numpy() ** 2))

    assert worn_rms > normal_rms * 1.5


def test_knock_condition_has_retarded_simulated_ignition_timing():
    normal, knock = (
        simulate_engine(condition, seed=42)[0]
        for condition in ("normal", "knock")
    )

    assert "ignition_timing_deg" in normal.columns
    assert knock["ignition_timing_deg"].mean() < (
        normal["ignition_timing_deg"].mean() - 2
    )
