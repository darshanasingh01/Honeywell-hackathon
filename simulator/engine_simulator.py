"""Reproducible synthetic engine simulator for the Honeywell hackathon.

All signals are illustrative simulations, not validated real-engine data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = (
    "normal",
    "misfire",
    "knock",
    "valve_train_wear",
    "overheating",
)


def simulate_engine(
    condition: str = "normal",
    duration_s: float = 10.0,
    telemetry_hz: int = 10,
    signal_hz: int = 4000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (telemetry, waveform) DataFrames for one condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if duration_s <= 0 or telemetry_hz <= 0 or signal_hz < 2000:
        raise ValueError("Invalid duration or sampling rate")

    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / telemetry_hz)
    tw = np.arange(0, duration_s, 1 / signal_hz)

    # Illustrative operating point and telemetry.
    throttle = np.clip(
        35 + 8 * np.sin(2 * np.pi * 0.12 * t)
        + rng.normal(0, 1.2, len(t)), 5, 90
    )
    load = np.clip(
        0.20 + 0.006 * throttle + rng.normal(0, 0.025, len(t)),
        0.05, 0.95
    )
    rpm = 1500 + 9 * throttle + 110 * np.sin(2 * np.pi * 0.08 * t)
    rpm += rng.normal(0, 12, len(t))

    coolant = 88 + 0.4 * np.sin(2 * np.pi * 0.025 * t)
    coolant += rng.normal(0, 0.12, len(t))
    stft = rng.normal(0, 1.5, len(t))
    ltft = 2.0 + rng.normal(0, 0.5, len(t))

    # Simplified ECU-like spark advance, in crank degrees before TDC.
    ignition = 12 + 0.012 * (rpm - 1500) - 8 * load
    ignition += rng.normal(0, 0.35, len(t))

    if condition == "misfire":
        stft += 5 + rng.normal(0, 1, len(t))
        rpm += 35 * np.sin(2 * np.pi * 1.3 * t)
        ignition -= 1.5
    elif condition == "knock":
        stft += 2.5
        ignition -= 4.0  # illustrative retarded-timing response
    elif condition == "overheating":
        coolant += 0.9 * t
        load = np.clip(load * 0.92, 0.05, 0.95)
        ignition -= 1.0

    # Interpolate RPM to waveform time and integrate crank angle.
    # CKP pulses use the configurable missing-tooth wheel model.
    # At finite sampling rates, some tooth events may be unresolved.
    from simulator.ckp_model import generate_ckp

    rpm_wave = np.interp(tw, t, rpm)
    crank_angle_deg = (
        np.cumsum(rpm_wave / 60.0 / signal_hz) * 360.0
    )
    phase = np.deg2rad(crank_angle_deg)
    crank_pulse = generate_ckp(crank_angle_deg, signal_hz)

    engine_order_hz = rpm_wave / 60.0 * 2
    combustion = 0.18 * np.sin(
        2 * np.pi * np.cumsum(engine_order_hz) / signal_hz
    )
    audio = combustion + 0.035 * rng.normal(size=len(tw))
    knock_signal = 0.025 * rng.normal(size=len(tw))

    # Three-axis accelerometer, in arbitrary synthetic amplitude units.
    accel_x = 0.06 * np.sin(phase) + 0.025 * rng.normal(size=len(tw))
    accel_y = 0.04 * np.sin(2 * phase + 0.3) + 0.025 * rng.normal(size=len(tw))
    accel_z = 0.05 * np.sin(phase + 0.7) + 0.025 * rng.normal(size=len(tw))

    if condition == "knock":
        burst = np.maximum(0, np.sin(2 * np.pi * 18 * tw)) ** 12
        impact = burst * np.sin(2 * np.pi * 1100 * tw)
        knock_signal += 0.25 * impact
        audio += 0.18 * impact
        accel_x += 0.10 * impact
        accel_y += 0.07 * impact
        accel_z += 0.08 * impact

    elif condition == "misfire":
        event = np.sin(2 * np.pi * 3 * tw) > 0.92
        audio[event] *= 0.25
        accel_x[event] *= 0.55
        accel_y[event] *= 0.55
        accel_z[event] *= 0.55

    elif condition == "valve_train_wear":
        impact = np.maximum(0, np.sin(2 * np.pi * 25 * tw)) ** 8
        knock_signal += 0.15 * impact
        audio += 0.06 * impact
        accel_x += 0.22 * impact
        accel_y += 0.16 * impact
        accel_z += 0.19 * impact

    elif condition == "overheating":
        audio *= 1 + 0.025 * tw

    telemetry = pd.DataFrame({
        "timestamp_s": t,
        "engine_rpm": rpm,
        "coolant_temp_c": coolant,
        "throttle_pct": throttle,
        "engine_load": load,
        "stft_pct": stft,
        "ltft_pct": ltft,
        "ignition_timing_deg": ignition,
        "condition": condition,
    })
    waveform = pd.DataFrame({
        "timestamp_s": tw,
        "microphone_audio": audio,
        "knock_sensor": knock_signal,
        "crank_pulse": crank_pulse,
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "condition": condition,
    })
    return telemetry, waveform


def main() -> None:
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
