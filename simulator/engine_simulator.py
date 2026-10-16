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

Three-axis accelerometer channels (``accel_x``, ``accel_y``, ``accel_z``) are also
emitted at 25.6 kHz. Each axis carries a distinct, fault-aware vibration mix so the
per-axis band structure (not raw loudness) is discriminative:
  - accel_x: broadband block noise + firing-rate thumps; for knock it also carries a
    scaled 7 kHz ring at fe+delay (lands in the 5-10 kHz band, like the knock sensor).
  - accel_y: block noise + lighter firing content; for valve_train_wear it carries a
    scaled 4 kHz click at each cam event (1-5 kHz mid band).
  - accel_z: block noise + firing thumps; for misfire it is amplitude-modulated at the
    crank rate and kicked at each dropped firing (crank-rate modulation / low band).
The per-run ``amp_scale`` is applied last to every channel, so loudness alone cannot
separate the classes.

A telemetry channel ``ignition_timing`` (deg BTDC, 10 Hz) is fault-aware: normal /
valve_train_wear sit near ~15 deg, knock retards toward ~8 deg with wear severity,
misfire scatters widely, overheating retards slightly to ~13 deg.
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
    # Separate RNG stream for the added sensors (accelerometer + ignition timing)
    # so the existing mic / knock / crank / telemetry draws stay bit-identical to
    # the pre-sensor baseline (these additions are additive, not a reshuffle).
    accel_rng = np.random.default_rng(seed + 10_000)
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

    # Ignition timing (deg BTDC, 10 Hz telemetry). Fault-aware: the ECU retards
    # timing to protect the engine when it detects knock, scatters under misfire,
    # and retards slightly when running hot. Severity is sampled on the telemetry
    # clock (0.4 -> 1.0 over the run) so the retard deepens as wear worsens.
    tel_severity = 0.4 + 0.6 * (t / duration_s)
    if condition == "knock":
        # Retard from ~15 deg toward ~8 deg with severity.
        ignition_timing = 15.0 - 7.0 * tel_severity + accel_rng.normal(0, 0.3, len(t))
    elif condition == "misfire":
        ignition_timing = 15.0 + accel_rng.normal(0, 2.0, len(t))  # wide scatter
    elif condition == "overheating":
        ignition_timing = 13.0 + accel_rng.normal(0, 0.4, len(t))  # slight retard
    else:  # normal, valve_train_wear
        ignition_timing = 15.0 + accel_rng.normal(0, 0.4, len(t))
    ignition_timing = np.clip(ignition_timing, 5.0, 25.0)

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

    # --- Vibration kernels (hoisted so the accel block below can reuse them) ---
    # 7 kHz damped knock ring (5-10 kHz band).
    ring_len = int(round(KNOCK_RING_MS / 1000.0 * signal_hz))
    rk = np.arange(ring_len) / signal_hz
    base_ring = np.exp(-KNOCK_DECAY * rk) * np.sin(2 * np.pi * KNOCK_RING_HZ * rk)
    knock_delay = int(round(KNOCK_DELAY_MS / 1000.0 * signal_hz))
    # 4 kHz damped valve-train click (1-5 kHz mid band).
    click_len = int(round(VALVE_CLICK_MS / 1000.0 * signal_hz))
    ck = np.arange(click_len) / signal_hz
    base_click = np.exp(-VALVE_DECAY * ck) * np.sin(2 * np.pi * VALVE_CLICK_HZ * ck)
    # Cam rate = RPM/120 -> one click per 2 revolutions (every 4th firing).
    cam_events = firing_events[::4]
    # Record the firings that actually knock so the accel_x ring matches them.
    knock_ring_events = []  # list of (start_index, amplitude)

    # --- Condition-specific high-frequency physics ---
    if condition == "knock":
        for fe in firing_events:
            if rng.random() > KNOCK_PROB:
                continue  # sporadic: only some firings knock
            sev = severity[min(fe, n - 1)]
            amp = (0.15 + 0.45 * rng.random()) * sev  # random per-cycle amplitude
            start = fe + knock_delay
            _add_kernel(knock_signal, start, amp * base_ring)
            _add_kernel(audio, start, 0.4 * amp * base_ring)
            knock_ring_events.append((start, amp))

    elif condition == "valve_train_wear":
        for ce in cam_events:
            sev = severity[min(ce, n - 1)]
            amp = (0.5 + 0.35 * rng.random()) * sev
            _add_kernel(knock_signal, ce, amp * base_click)
            _add_kernel(audio, ce, 0.35 * amp * base_click)

    elif condition == "overheating":
        thermal_factor = 1 + 0.025 * tw
        audio *= thermal_factor

    # --- Three-axis accelerometer (25.6 kHz) ---
    # Broadband "block" noise floor on every axis (seed-driven level), then
    # axis-specific firing content and the fault-aware signatures. amp_scale is
    # applied last (below) so raw loudness cannot separate the classes.
    accel_x = noise_scale * 0.030 * accel_rng.normal(size=n)
    accel_y = noise_scale * 0.030 * accel_rng.normal(size=n)
    accel_z = noise_scale * 0.030 * accel_rng.normal(size=n)

    for fe in firing_events:
        dropped = fe in misfire_drop
        # accel_x: firing-rate thumps (dropped on misfire).
        if not dropped:
            _add_kernel(accel_x, fe, 0.12 * (0.85 + 0.30 * accel_rng.random()) * thump_kernel)
        # accel_y: lighter firing content.
        _add_kernel(accel_y, fe, 0.06 * (0.85 + 0.30 * accel_rng.random()) * thump_kernel)
        # accel_z: firing-rate thumps (dropped on misfire).
        if not dropped:
            _add_kernel(accel_z, fe, 0.12 * (0.85 + 0.30 * accel_rng.random()) * thump_kernel)

    if condition == "knock":
        # Same 7 kHz ring at the same fe+delay starts the knock sensor saw, so
        # accel_x carries matching 5-10 kHz knock-band energy.
        for start, amp in knock_ring_events:
            _add_kernel(accel_x, start, 1.4 * amp * base_ring)

    elif condition == "valve_train_wear":
        # 4 kHz cam-rate clicks into accel_y (1-5 kHz mid band).
        for ce in cam_events:
            sev = severity[min(ce, n - 1)]
            amp = (0.5 + 0.35 * accel_rng.random()) * sev
            _add_kernel(accel_y, ce, 1.2 * amp * base_click)

    elif condition == "misfire":
        # Crank-rate amplitude modulation + a low-frequency kick at each dropped
        # firing: accel_z gains crank-rate envelope repetition and low-band energy.
        crank_am = 1.0 + 0.8 * np.sin(2 * np.pi * crank_phase_cycles)
        accel_z = accel_z * crank_am
        for fe in sorted(misfire_drop):
            _add_kernel(accel_z, fe, 0.5 * thump_kernel)

    # Broadband noise (seed-driven level).
    audio += noise_scale * 0.035 * rng.normal(size=n)
    knock_signal += noise_scale * 0.025 * rng.normal(size=n)

    # Apply seed-driven per-run amplitude variation to every channel so loudness
    # alone is not discriminative.
    audio *= amp_scale
    knock_signal *= amp_scale
    accel_x *= amp_scale
    accel_y *= amp_scale
    accel_z *= amp_scale

    telemetry = pd.DataFrame({
        "timestamp_s": t,
        "engine_rpm": rpm,
        "coolant_temp_c": coolant,
        "throttle_pct": throttle,
        "engine_load": load,
        "stft_pct": fuel_stft,
        "ltft_pct": fuel_ltft,
        "ignition_timing": ignition_timing,
        "condition": condition,
    })
    waveform = pd.DataFrame({
        "timestamp_s": tw,
        "microphone_audio": audio,
        "knock_sensor": knock_signal,
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
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
