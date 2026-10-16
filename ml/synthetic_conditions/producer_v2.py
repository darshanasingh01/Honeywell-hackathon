"""Isolated synthetic-only V2 producer. Never trains or edits upstream source."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import types

import numpy as np
import pandas as pd
import scipy

from .ckp_v2 import process_edges
from .ecu_v2 import decode_pid, window_features
from .schema_v2 import FEATURE_COLUMNS, MANIFEST, SCHEMA_VERSION, validate_frame, validate_row

SOURCE_COMMIT = "0ade8c2326d76bf104ba42613f1ea771558c6af8"
VERSION = "engineguard-synthetic-producer-v2.1"
FS = 25600
TELEMETRY_HZ = 10
TICK = 1e-5
WINDOW = 0.5
HOP = 0.25
CONDITIONS = tuple(MANIFEST["class_labels"])
DEFAULT_OUTPUT = Path("ml/datasets/synthetic_conditions_v2")
SOURCE_PATHS = ("simulator/engine_simulator.py", "feature_extraction/signal_features.py",
                "feature_extraction/can_codec.py")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def approved_sources():
    """Load immutable, explicitly approved blobs, with isolated module namespaces."""
    root = Path(__file__).resolve().parents[2]
    modules, hashes = [], {}
    for name in SOURCE_PATHS:
        result = subprocess.run(["git", "show", f"{SOURCE_COMMIT}:{name}"], cwd=root,
                                capture_output=True, check=True)
        hashes[name] = sha256(result.stdout)
        module = types.ModuleType("engineguard_v2_pinned_" + Path(name).stem)
        module.__file__ = f"{SOURCE_COMMIT}:{name}"
        exec(compile(result.stdout.decode("utf-8"), module.__file__, "exec"), module.__dict__)
        modules.append(module)
    return (*modules, hashes)


def map_pressure_kpa(load, throttle_pct, rpm, atmospheric_kpa=101.325):
    """Declared naturally aspirated spark-ignition surrogate, not calibrated physics.

    Higher throttle/load reduces intake vacuum. At a fixed opening, increasing
    speed increases pumping demand and slightly reduces manifold pressure.
    """
    load, throttle_pct, rpm = map(np.asarray, (load, throttle_pct, rpm))
    if (not all(np.isfinite(a).all() for a in (load, throttle_pct, rpm))
            or np.any((load < 0) | (load > 1))
            or np.any((throttle_pct < 0) | (throttle_pct > 100))
            or np.any((rpm < 300) | (rpm > 8000))):
        raise ValueError("invalid MAP operating point")
    filling = 0.18 + 0.65 * load + 0.17 * throttle_pct / 100
    pumping = 1 - 0.08 * (rpm - 1800) / 6200
    return np.clip(atmospheric_kpa * filling * pumping, 15, atmospheric_kpa)


def tooth_events(telemetry, duration_s):
    """Use the approved simulator's cumulative phase convention and RPM interpolation.

    Analytical pitch crossings are timestamped on a 100 kHz acquisition clock;
    no crank-angle values are interpreted as uniformly sampled time signals.
    """
    time = np.arange(0, duration_s, 1 / FS)
    rpm = np.interp(time, telemetry.timestamp_s, telemetry.engine_rpm)
    phase_pitches = np.cumsum(rpm / FS)  # RPM/60 * 60 pitches/revolution
    crossings = np.arange(np.ceil(phase_pitches[0]), np.floor(phase_pitches[-1]) + 1).astype(int)
    crossings = crossings[crossings % 60 < 58]
    times = np.round(np.interp(crossings, phase_pitches, time) / TICK) * TICK
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamp clock cannot resolve edges")
    return [(float(t), int(p % 60)) for t, p in zip(times, crossings)]


def decoded_telemetry(source):
    """Synchronous 10 Hz sample clock; real bytes, no fictitious CAN bus latency.

    Reuse approved byte encode/decode for the seven V1 PIDs. New ML-owned MAP
    and load adapter supplies the two explicitly named V2 measurements.
    """
    _, _, codec, _ = approved_sources()
    decoded = pd.DataFrame({"timestamp_s": source.timestamp_s.to_numpy()})
    payloads = {}
    for pid, (signal, _) in codec.PID_TO_SIGNAL.items():
        data = [codec.encode_pid(pid, float(v)) for v in source[signal]]
        payloads[f"0x{pid:02X}"] = [list(p) for p in data]
        decoded[signal] = [codec.decode_pid(pid, p) for p in data]
    load = [decode_pid(0x04, p) for p in payloads["0x04"]]
    decoded["engine_load_pct"] = [p["engine_load_pct"] for p in load]
    pressure = map_pressure_kpa(source.engine_load, source.throttle_pct, source.engine_rpm)
    payloads["0x0B"] = [[int(round(p))] for p in pressure]
    decoded["map_kpa"] = [decode_pid(0x0B, p)["map_kpa"] for p in payloads["0x0B"]]
    if not np.allclose(decoded.engine_load_pct, 100 * decoded.engine_load, atol=1e-12):
        raise ValueError("load fraction/percentage inconsistency")
    if np.max(np.abs(decoded.engine_rpm - source.engine_rpm)) > 0.125 + 1e-9:
        raise ValueError("RPM byte quantization inconsistency")
    if np.max(np.abs(decoded.map_kpa - pressure)) > 0.5 + 1e-9:
        raise ValueError("MAP byte quantization inconsistency")
    return decoded, payloads


def producer_paths():
    """Explicit exhaustive map, also saved in the generation manifest."""
    mapping = {}
    for name in FEATURE_COLUMNS:
        if name.startswith(("mic_", "knock_", "accel_")) and not name.startswith("knock_cycle_"):
            path = "approved DSP extract_fast_channel_features; approved simulated waveform"
        elif name.startswith("knock_cycle_"):
            path = "approved crank_gated_knock; tooth-0 revolution markers on waveform clock"
        elif name.startswith("crank_"):
            path = "complete revolution periods from indexed timestamped CKP"
        elif name.startswith("ckp_"):
            path = "ckp_v2.process_edges"
        elif name.startswith(("tel_map_kpa_", "tel_engine_load_pct_")):
            path = "ecu_v2.decode_pid + window_features"
        elif name.startswith("tel_"):
            path = "approved PID byte roundtrip + extract_telemetry_features"
        else:
            raise ValueError(f"no producer for {name}")
        mapping[name] = path
    return mapping


def generate_run(condition, seed, duration_s, namespace):
    simulator, dsp, _, _ = approved_sources()
    source, waveform = simulator.simulate_engine(condition, duration_s, TELEMETRY_HZ, FS, seed)
    decoded, payloads = decoded_telemetry(source)
    events = tooth_events(source, duration_s)
    # Derive one-revolution gating markers from CKP, never use every tooth as firing.
    pulse = np.zeros(len(waveform))
    for time, index in events:
        if index == 0:
            k = int(np.searchsorted(waveform.timestamp_s.to_numpy(), time))
            pulse[k:min(k + round(0.001 * FS), len(pulse))] = 1
    run_id = f"{namespace}_{condition}_s{seed}"
    rows, checks = [], []
    for window_id, a in enumerate(range(0, len(waveform) - round(WINDOW * FS) + 1, round(HOP * FS))):
        b = a + round(WINDOW * FS)
        start, end = a / FS, b / FS
        ckp = process_edges(events, start=start, end=end, timestamp_resolution_s=TICK)
        if ckp["quality_status"] != "good":
            raise ValueError(f"unexpected acquisition failure in {run_id}/{window_id}")
        tel = decoded[(decoded.timestamp_s >= start) & (decoded.timestamp_s < end)]
        if len(tel) < 2:
            raise ValueError("missing ECU samples")
        features = {}
        for signal, prefix in (("microphone_audio", "mic"), ("knock_sensor", "knock"),
                               ("accel_x", "accel_x"), ("accel_y", "accel_y"), ("accel_z", "accel_z")):
            features.update(dsp.extract_fast_channel_features(waveform[signal].to_numpy()[a:b], FS, prefix))
        features.update(dsp.extract_telemetry_features(tel))
        for family in ("map_kpa", "engine_load_pct"):
            ecu = window_features(list(zip(decoded.timestamp_s, decoded[family])),
                                  start=start, end=end, family=family)
            if ecu["quality_status"] != "good":
                raise ValueError("unexpected ECU acquisition failure")
            features.update(ecu["features"])
        features.update(ckp["features"])
        in_window = [(t, i) for t, i in events if start <= t < end]
        times = np.array([p[0] for p in in_window])
        periods = times[58:] - times[:-58]
        features.update(crank_rpm=float(np.mean(60 / periods)),
                        crank_rpm_std=float(np.std(60 / periods)),
                        crank_interval_cv=float(periods.std() / periods.mean()))
        gated = dsp.crank_gated_knock(waveform.knock_sensor.to_numpy()[a:b], pulse[a:b], FS)
        # The approved helper's no-gate zero fallback must never satisfy V2 silently.
        gates = dsp._rising_edges(pulse[a:b])
        if not np.any(gates + round(0.003 * FS) <= b - a):
            raise ValueError("no complete knock gating interval")
        features.update(gated)
        row = {"run_id": run_id, "window_id": window_id, "label": condition,
               "t_start": start, "t_end": end, "ckp_signal_quality_status": "good",
               "map_signal_quality_status": "good", "engine_load_signal_quality_status": "good",
               "data_kind": "synthetic_simulated_engine_conditions", "source_id": run_id,
               "source_commit": SOURCE_COMMIT, "processor_revision": VERSION,
               "timestamp_resolution_s": TICK}
        row.update({"available_" + g["name"]: True for g in MANIFEST["sensor_groups"]})
        row.update(features)
        validate_row(row)
        # CKP RPM is an interval estimate; check against shared-clock source bounds.
        truth = np.interp(np.array([start, end]), source.timestamp_s, source.engine_rpm)
        internal = source[(source.timestamp_s >= start) & (source.timestamp_s < end)].engine_rpm
        lower, upper = min(truth.min(), internal.min()), max(truth.max(), internal.max())
        ckp_min, ckp_max = features["ckp_tooth_rpm_min"], features["ckp_tooth_rpm_max"]
        if ckp_min < lower * 0.97 or ckp_max > upper * 1.03:
            raise ValueError("CKP and ECU trajectory inconsistent")
        checks.append({"window_id": window_id, "ecu_samples": len(tel),
                       "ckp_expected_gaps": features["ckp_expected_gap_count"],
                       "ckp_unexpected_gaps": features["ckp_unexpected_gap_count"],
                       "load_pct_consistent": True, "rpm_consistent": True})
        rows.append(row)
    return rows, {"run_id": run_id, "condition": condition, "seed": seed,
                  "windows": checks, "telemetry": decoded.to_dict(orient="list"),
                  "pid_payloads": payloads, "ckp_events": events}


def validate_dataset(frame):
    X, y, groups = validate_frame(frame)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("nominal dataset must contain all 167 finite features")
    if not (frame[["ckp_signal_quality_status", "map_signal_quality_status",
                   "engine_load_signal_quality_status"]] == "good").all().all():
        raise ValueError("acquisition failures cannot be nominal training rows")
    return {"status": "PASS", "synthetic_only": True, "rows": len(X),
            "feature_count": len(FEATURE_COLUMNS), "metadata_count": len(MANIFEST["metadata_columns"]),
            "run_count": int(groups.nunique()), "class_balance": y.value_counts().sort_index().to_dict(),
            "runs_per_class": frame.groupby("label").run_id.nunique().to_dict(),
            "all_features_finite": True, "all_rows_schema_valid": True,
            "unexpected_ckp_gaps": int(frame.ckp_unexpected_gap_count.sum()),
            "schema_version": SCHEMA_VERSION, "classifier_trained": False}


def generate_dataset(output, *, seeds, duration_s, namespace):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    if (not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds)
            or not np.isfinite(duration_s) or duration_s < WINDOW
            or not namespace or not all(c.isalnum() or c in "_-" for c in namespace)):
        raise ValueError("invalid seed/duration/namespace configuration")
    rows, evidence = [], []
    for condition in CONDITIONS:
        for seed in seeds:
            run_rows, audit = generate_run(condition, seed, duration_s, namespace)
            rows.extend(run_rows); evidence.append(audit)
    frame = pd.DataFrame(rows, columns=MANIFEST["metadata_columns"] + list(FEATURE_COLUMNS))
    report = validate_dataset(frame)
    # Validate fully in memory before creating any dataset files.
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "features.csv", index=False, float_format="%.17g")
    # Re-read the serialized contract, including header/order/types and all ranges.
    reread = pd.read_csv(output / "features.csv", float_precision="round_trip")
    validate_dataset(reread)
    producer_hashes = {p.name: sha256(p.read_bytes()) for p in
        [Path(__file__), *[Path(__file__).with_name(n) for n in
         ("schema_v2.py", "schema_v2.json", "ecu_v2.py", "ckp_v2.py")]]}
    manifest = {"producer_version": VERSION, "schema_version": SCHEMA_VERSION,
                "data_kind": "synthetic_simulated_engine_conditions", "real_engine_measurements": False,
                "source_commit": SOURCE_COMMIT, "approved_blob_sha256": approved_sources()[-1],
                "ml_source_sha256": producer_hashes, "feature_producer_paths": producer_paths(),
                "configuration": {"namespace": namespace, "seeds": list(seeds), "duration_s": duration_s,
                                  "waveform_hz": FS, "ecu_hz": TELEMETRY_HZ, "timestamp_tick_s": TICK,
                                  "nominal_teeth": 60, "missing_teeth": 2, "window_s": WINDOW, "hop_s": HOP,
                                  "map_model": "101.325*(0.18+0.65*load+0.17*throttle/100)*(1-0.08*(rpm-1800)/6200), clipped [15,101.325] kPa",
                                  "ecu_alignment": "synchronous samples, PID byte quantization, no bus latency"},
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__},
                "features_sha256": sha256((output / "features.csv").read_bytes()),
                "limitations": ["Synthetic-only illustrative engine conditions, not physical engine validation.",
                                "MAP is a declared naturally aspirated intake surrogate, not calibrated pressure physics.",
                                "CKP shares approved smoothed RPM; no torque-resolved combustion dynamics.",
                                "Acquisition failures are excluded, not mechanical fault labels.",
                                "Different simulator seeds do not establish independent physical engines."]}
    for name, obj in (("manifest.json", manifest), ("validation.json", report), ("run_evidence.json", evidence)):
        (output / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output root exists; choose a new directory")
    smoke = generate_dataset(args.output / "smoke", seeds=(3100,), duration_s=1.0, namespace="v2_smoke")
    print("Smoke:", json.dumps(smoke), flush=True)
    full = generate_dataset(args.output / "training", seeds=tuple(range(3200, 3206)),
                            duration_s=10.0, namespace="v2_train")
    print("Full:", json.dumps(full), flush=True)


if __name__ == "__main__":
    main()
