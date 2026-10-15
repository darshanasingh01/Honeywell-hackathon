"""Synthetic engine sensor simulator for the Honeywell hackathon.

This generates illustrative synthetic data, not validated real-engine recordings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONDITIONS = ("normal", "misfire", "knock", "valve_train_wear", "overheating")


def simulate_engine(
    condition: str = "normal",
    duration_s: float = 10.0,
    telemetry_hz: int = 10,
    signal_hz: int = 4000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (telemetry, waveform) data for one labelled engine condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if duration_s <= 0 or telemetry_hz <= 0 or signal_hz <= 0:
        raise ValueError("duration and sampling rates must be positive")
    if signal_hz < 1000:
        raise ValueError("signal_hz must be at least 1000 for this prototype")

    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / telemetry_hz)
    tw = np.arange(0, duration_s, 1 / signal_hz)

    # Slowly varying operating point: these are illustrative assumptions.
    throttle = np.clip(35 + 8 * np.sin(2 * np.pi * 0.12 * t)
                       + rng.normal(0, 1.2, len(t)), 5, 90)
    load = np.clip(0.20 + 0.006 * throttle
                   + rng.normal(0, 0.025, len(t)), 0.05, 0.95)
    rpm = 1500 + 9 * throttle + 110 * np.sin(2 * np.pi * 0.08 * t)
    rpm += rng.normal(0, 12, len(t))

    coolant = 88 + 0.4 * np.sin(2 * np.pi * 0.025 * t)
    coolant += rng.normal(0, 0.12, len(t))
    fuel_stft = rng.normal(0, 1.5, len(t))
    fuel_ltft = 2.0 + rng.normal(0, 0.5, len(t))

    # Fault effects in the slower ECU-like telemetry.
    if condition == "misfire":
        fuel_stft += 5 + rng.normal(0, 1, len(t))
        rpm += 35 * np.sin(2 * np.pi * 1.3 * t)
    elif condition == "knock":
        fuel_stft += 2.5
    elif condition == "valve_train_wear":
        # The primary wear signature is modelled in the waveform below.
        pass
    elif condition == "overheating":
        coolant += 0.9 * t
        load = np.clip(load * 0.92, 0.05, 0.95)

    # Crank pulse timing follows instantaneous RPM by interpolating phase.
    rpm_wave = np.interp(tw, t, rpm)
    phase_cycles = np.cumsum(rpm_wave / 60.0 / signal_hz)
    crank_phase = np.mod(phase_cycles * 2 * np.pi, 2 * np.pi)
    crank_pulse = (np.sin(crank_phase) > 0.98).astype(np.int8)

    # Audio-like and knock/vibration-like synthetic waveform.
    engine_order_hz = rpm_wave / 60.0 * 2
    audio = 0.18 * np.sin(2 * np.pi * np.cumsum(engine_order_hz) / signal_hz)
    audio += 0.035 * rng.normal(size=len(tw))

    knock_signal = 0.025 * rng.normal(size=len(tw))
    if condition == "knock":
        # Repeated short high-frequency bursts; synthetic proxy, not measured data.
        burst = np.maximum(0, np.sin(2 * np.pi * 18 * tw)) ** 12
        knock_signal += 0.25 * burst * np.sin(2 * np.pi * 1100 * tw)
        audio += 0.18 * burst * np.sin(2 * np.pi * 1100 * tw)
    elif condition == "misfire":
        # Periodically attenuate the combustion-related audio component.
        event = np.sin(2 * np.pi * 3 * tw) > 0.92
        audio[event] *= 0.25
    elif condition == "valve_train_wear":
        wear = np.maximum(0, np.sin(2 * np.pi * 25 * tw)) ** 8
        knock_signal += 0.15 * wear
        audio += 0.06 * wear
    elif condition == "overheating":
        thermal_factor = 1 + 0.025 * tw
        audio *= thermal_factor

    telemetry = pd.DataFrame({
        "timestamp_s": t,
        "engine_rpm": rpm,
        "coolant_temp_c": coolant,
        "throttle_pct": throttle,
        "engine_load": load,
        "stft_pct": fuel_stft,
        "ltft_pct": fuel_ltft,
        "condition": condition,
    })
    waveform = pd.DataFrame({
        "timestamp_s": tw,
        "microphone_audio": audio,
        "knock_sensor": knock_signal,
        "crank_pulse": crank_pulse,
        "condition": condition,
    })
    return telemetry, waveform


def main() -> None:
    from pathlib import Path

    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        telemetry, waveform = simulate_engine(condition, seed=42)
        telemetry.to_csv(out / f"{condition}_telemetry.csv", index=False)
        waveform.to_csv(out / f"{condition}_waveform.csv", index=False)
        print(
            f"{condition}: {len(telemetry)} telemetry rows, "
            f"{len(waveform)} waveform rows"
        )


if __name__ == "__main__":
    main()
