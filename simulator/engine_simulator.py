"""Synthetic engine sensor simulator for the Honeywell hackathon.

This generates illustrative synthetic data, not validated real-engine recordings.

Physics model (4-cylinder, 4-stroke):
  - Combustion firing frequency = RPM/60 * cyl/2  (2 firings per crank revolution).
  - Crank pulse rises once per revolution (every 2 firings).
  - Camshaft (valve) rate = RPM/120 (half crank speed).
Fault signatures are injected into the 25.6 kHz waveform so that band / crank-gated
features are discriminative beyond raw loudness:
  - knock: damped ~7 kHz ring fired ~2 ms after a random subset of firings, random
    per-cycle amplitude (5-10 kHz band energy + crank-gated knock).
  - valve_train_wear: short 3-5 kHz click bursts at the cam rate (mid band, NOT the
    5-10 kHz knock band -- separates it from knock).
  - misfire: periodically dropped firing-rate combustion pulse + small crank slowdown
    + raised short-term fuel trim.
  - overheating: coolant_temp_c ramps upward over the run.
A wear-severity ramp worsens the fault amplitude over the run (for "predict wear"),
and a seed-driven per-run amplitude/noise variation keeps loudness alone from
separating the classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONDITIONS = ("normal", "misfire", "knock", "valve_train_wear", "overheating")

N_CYLINDERS = 4  # 4-stroke, 4-cylinder assumption

# Knock ring parameters.
KNOCK_RING_HZ = 7000.0
KNOCK_DELAY_MS = 2.0
KNOCK_RING_MS = 3.0
KNOCK_DECAY = 1400.0  # exponential decay rate (1/s)
KNOCK_PROB = 0.5      # fraction of firings that knock (sporadic, random amplitude)

# Valve-train click parameters (3-5 kHz band).
VALVE_CLICK_HZ = 4000.0
VALVE_CLICK_MS = 1.5
VALVE_DECAY = 3000.0


def _add_kernel(target: np.ndarray, start: int, kernel: np.ndarray) -> None:
    """Add a short kernel into target at start (clipped at array end)."""
    end = min(start + kernel.size, target.size)
    if start < 0 or start >= target.size or end <= start:
        return
    target[start:end] += kernel[: end - start]


def _firing_events(firing_freq_wave: np.ndarray, signal_hz: int) -> np.ndarray:
    """Sample indices where a combustion firing occurs."""
    phase = np.cumsum(firing_freq_wave / signal_hz)
    whole = np.floor(phase).astype(np.int64)
    events = np.nonzero(np.diff(whole) >= 1)[0] + 1
    return events


def simulate_engine(
    condition: str = "normal",
    duration_s: float = 10.0,
    telemetry_hz: int = 10,
    signal_hz: int = 25600,
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
    n = tw.size

    # Seed-driven per-run amplitude / noise variation so loudness alone cannot
    # separate the classes.
    amp_scale = float(rng.uniform(0.80, 1.20))
    noise_scale = float(rng.uniform(0.80, 1.20))

    # Wear-severity ramp: fault worsens over the run (0.4 -> 1.0).
    severity = 0.4 + 0.6 * (tw / duration_s)

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
        fuel_stft += 6 + rng.normal(0, 1, len(t))
        rpm += 35 * np.sin(2 * np.pi * 1.3 * t)  # periodic crank slowdown
    elif condition == "knock":
        fuel_stft += 2.5
    elif condition == "valve_train_wear":
        # The primary wear signature is modelled in the waveform below.
        pass
    elif condition == "overheating":
        coolant += 2.2 * t  # steep ramp: clearly positive slope after CAN quant
        load = np.clip(load * 0.92, 0.05, 0.95)

    # Instantaneous RPM / firing frequency across the waveform clock.
    rpm_wave = np.interp(tw, t, rpm)
    rev_per_s = rpm_wave / 60.0
    firing_freq = rev_per_s * (N_CYLINDERS / 2.0)   # 2 firings / rev for 4-stroke 4-cyl

    # --- Crank pulse: one rising edge per revolution ---
    crank_phase_cycles = np.cumsum(rev_per_s / signal_hz)
    crank_whole = np.floor(crank_phase_cycles).astype(np.int64)
    crank_edges = np.nonzero(np.diff(crank_whole) >= 1)[0] + 1
    crank_pulse = np.zeros(n, dtype=np.int8)
    pulse_width = max(1, int(round(0.001 * signal_hz)))  # ~1 ms high
    for e in crank_edges:
        crank_pulse[e: min(e + pulse_width, n)] = 1

    # --- Combustion firing events (2 per revolution) ---
    firing_events = _firing_events(firing_freq, signal_hz)

    # --- Base audio: combustion thumps + engine-order tone + noise ---
    engine_order_hz = rev_per_s * 2.0
    audio = 0.14 * np.sin(2 * np.pi * np.cumsum(engine_order_hz) / signal_hz)

    # Low-frequency combustion thump kernel (~300 Hz damped).
    thump_len = int(round(0.004 * signal_hz))
    tk = np.arange(thump_len) / signal_hz
    thump_kernel = np.exp(-600.0 * tk) * np.sin(2 * np.pi * 300.0 * tk)

    # Misfire: drop every Nth firing's combustion pulse.
    misfire_drop = set()
    if condition == "misfire":
        misfire_drop = set(firing_events[::7])  # drop ~1 in 7 firings

    combustion = np.zeros(n)
    for fe in firing_events:
        if fe in misfire_drop:
            continue
        amp = 0.20 * (0.85 + 0.30 * rng.random())
        _add_kernel(combustion, fe, amp * thump_kernel)
    audio += combustion

    # --- Knock sensor base (vibration) ---
    knock_signal = np.zeros(n)

    # --- Condition-specific high-frequency physics ---
    if condition == "knock":
        ring_len = int(round(KNOCK_RING_MS / 1000.0 * signal_hz))
        rk = np.arange(ring_len) / signal_hz
        base_ring = np.exp(-KNOCK_DECAY * rk) * np.sin(2 * np.pi * KNOCK_RING_HZ * rk)
        delay = int(round(KNOCK_DELAY_MS / 1000.0 * signal_hz))
        for fe in firing_events:
            if rng.random() > KNOCK_PROB:
                continue  # sporadic: only some firings knock
            sev = severity[min(fe, n - 1)]
            amp = (0.15 + 0.45 * rng.random()) * sev  # random per-cycle amplitude
            start = fe + delay
            _add_kernel(knock_signal, start, amp * base_ring)
            _add_kernel(audio, start, 0.4 * amp * base_ring)

    elif condition == "valve_train_wear":
        # Cam rate = RPM/120 -> one click per 2 revolutions (every 4th firing).
        cam_events = firing_events[::4]
        click_len = int(round(VALVE_CLICK_MS / 1000.0 * signal_hz))
        ck = np.arange(click_len) / signal_hz
        base_click = np.exp(-VALVE_DECAY * ck) * np.sin(2 * np.pi * VALVE_CLICK_HZ * ck)
        for ce in cam_events:
            sev = severity[min(ce, n - 1)]
            amp = (0.5 + 0.35 * rng.random()) * sev
            _add_kernel(knock_signal, ce, amp * base_click)
            _add_kernel(audio, ce, 0.35 * amp * base_click)

    elif condition == "overheating":
        thermal_factor = 1 + 0.025 * tw
        audio *= thermal_factor

    # Broadband noise (seed-driven level).
    audio += noise_scale * 0.035 * rng.normal(size=n)
    knock_signal += noise_scale * 0.025 * rng.normal(size=n)

    # Apply seed-driven per-run amplitude variation to the whole waveform.
    audio *= amp_scale
    knock_signal *= amp_scale

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
