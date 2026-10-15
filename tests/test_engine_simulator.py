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
        signal_hz=25600, seed=7
    )
    assert len(telemetry) == 20
    assert len(waveform) == 51200
    assert np.all(np.diff(telemetry["timestamp_s"]) > 0)
    assert np.all(np.diff(waveform["timestamp_s"]) > 0)


def test_default_signal_rate_is_25600():
    _, waveform = simulate_engine("normal", duration_s=1, seed=7)
    assert len(waveform) == 25600


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


def test_knock_has_elevated_5k_10k_band_energy():
    # Loudness is deliberately not discriminative (seed-driven per-run gain).
    # The knock ring at ~7 kHz must show up in the 5-10 kHz knock band.
    from feature_extraction.signal_features import extract_band_features

    _, normal = simulate_engine("normal", seed=42)
    _, knock = simulate_engine("knock", seed=42)
    fs = 25600

    normal_r = extract_band_features(
        normal["knock_sensor"].to_numpy(), fs)["knock_5k_10k_ratio"]
    knock_r = extract_band_features(
        knock["knock_sensor"].to_numpy(), fs)["knock_5k_10k_ratio"]

    assert knock_r > normal_r * 1.3


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


def test_valve_wear_has_elevated_mid_band_energy():
    # Valve clicks sit at 3-5 kHz (mid band), NOT the 5-10 kHz knock band,
    # which is what separates wear from knock.
    from feature_extraction.signal_features import extract_band_features

    _, normal = simulate_engine("normal", seed=42)
    _, worn = simulate_engine("valve_train_wear", seed=42)
    fs = 25600

    normal_r = extract_band_features(
        normal["knock_sensor"].to_numpy(), fs)["mid_1k_5k_ratio"]
    worn_r = extract_band_features(
        worn["knock_sensor"].to_numpy(), fs)["mid_1k_5k_ratio"]

    assert worn_r > normal_r * 1.3
