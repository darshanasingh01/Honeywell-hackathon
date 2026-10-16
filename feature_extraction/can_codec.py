"""SAE J1979 (OBD-II) telemetry <-> CAN frame codec.

Encodes the simulator's ECU telemetry into J1979 request/response CAN frames,
serializes them to/from candump ``.log`` text, and decodes them back to
telemetry. The decode is intentionally coarser than the source (PID
quantization) -- that quantization is the realism this models.

PID table (mode 01):
  0x04 engine_load        A*100/255  %   (stored 0-1, i.e. /100)
  0x05 coolant_temp_c     A-40       deg C
  0x06 stft_pct           (A-128)*100/128  %
  0x07 ltft_pct           (A-128)*100/128  %
  0x0C engine_rpm         (256*A+B)/4  rpm   (2 data bytes)
  0x0E ignition_timing    A/2-64     deg BTDC
  0x11 throttle_pct       A*100/255  %
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

REQUEST_ID = 0x7DF
RESPONSE_ID = 0x7E8
POLL_PIDS = [0x04, 0x05, 0x06, 0x07, 0x0C, 0x0E, 0x11]

# pid -> (telemetry column, n_data_bytes)
PID_TO_SIGNAL = {
    0x04: ("engine_load", 1),
    0x05: ("coolant_temp_c", 1),
    0x06: ("stft_pct", 1),
    0x07: ("ltft_pct", 1),
    0x0C: ("engine_rpm", 2),
    0x0E: ("ignition_timing", 1),
    0x11: ("throttle_pct", 1),
}
SIGNAL_TO_PID = {sig: pid for pid, (sig, _) in PID_TO_SIGNAL.items()}


def _clamp_byte(x):
    return int(min(255, max(0, round(x))))


def encode_pid(pid, value):
    """Encode an engineering value into the J1979 data byte tuple."""
    if pid == 0x04:            # engine_load, stored 0-1 -> percent
        a = _clamp_byte(value * 100.0 * 255.0 / 100.0)
        return (a,)
    if pid == 0x05:            # coolant: A - 40
        a = _clamp_byte(value + 40.0)
        return (a,)
    if pid in (0x06, 0x07):    # fuel trim: (A-128)*100/128
        a = _clamp_byte(value * 128.0 / 100.0 + 128.0)
        return (a,)
    if pid == 0x0C:            # rpm: (256A+B)/4
        raw = int(min(65535, max(0, round(value * 4.0))))
        a = (raw >> 8) & 0xFF
        b = raw & 0xFF
        return (a, b)
    if pid == 0x0E:            # ignition timing: A/2 - 64 -> A = (value+64)*2
        a = _clamp_byte((value + 64.0) * 2.0)
        return (a,)
    if pid == 0x11:            # throttle: A*100/255
        a = _clamp_byte(value * 255.0 / 100.0)
        return (a,)
    raise ValueError(f"Unsupported PID 0x{pid:02X}")


def decode_pid(pid, data):
    """Decode data bytes back to the engineering value."""
    data = list(data)
    a = data[0]
    if pid == 0x04:
        return a * 100.0 / 255.0 / 100.0   # back to 0-1
    if pid == 0x05:
        return float(a - 40)
    if pid in (0x06, 0x07):
        return (a - 128) * 100.0 / 128.0
    if pid == 0x0C:
        b = data[1]
        return (256 * a + b) / 4.0
    if pid == 0x0E:
        return a / 2.0 - 64.0
    if pid == 0x11:
        return a * 100.0 / 255.0
    raise ValueError(f"Unsupported PID 0x{pid:02X}")


def telemetry_to_can(telemetry, poll_pids=POLL_PIDS,
                     request_gap_ms=5.0, reply_delay_ms=2.0):
    """Build a CAN frame log (request + response per PID per telemetry row)."""
    rows = []
    gap = request_gap_ms / 1000.0
    reply = reply_delay_ms / 1000.0
    pid_idx = {pid: i for i, pid in enumerate(poll_pids)}

    for _, r in telemetry.iterrows():
        base = float(r["timestamp_s"])
        for pid in poll_pids:
            sig, nbytes = PID_TO_SIGNAL[pid]
            req_ts = base + pid_idx[pid] * gap
            # Request: [02 01 PID 00 00 00 00 00]
            req_data = [0x02, 0x01, pid, 0, 0, 0, 0, 0]
            rows.append({
                "timestamp_s": req_ts,
                "can_id": REQUEST_ID,
                "dlc": 8,
                "data": tuple(req_data),
                "pid": pid,
                "kind": "request",
            })
            # Response: [len, 41, PID, A, (B), 00...]
            payload = list(encode_pid(pid, float(r[sig])))
            resp_core = [0x41, pid] + payload
            length = len(resp_core)
            resp_data = [length] + resp_core
            resp_data += [0] * (8 - len(resp_data))
            rows.append({
                "timestamp_s": req_ts + reply,
                "can_id": RESPONSE_ID,
                "dlc": 8,
                "data": tuple(resp_data[:8]),
                "pid": pid,
                "kind": "response",
            })

    log = pd.DataFrame(rows).sort_values("timestamp_s").reset_index(drop=True)
    return log


def can_to_telemetry(can_log):
    """Decode 0x7E8 responses and merge all PIDs onto RPM reply timestamps."""
    log = can_log.sort_values("timestamp_s").reset_index(drop=True)
    responses = log[log["can_id"] == RESPONSE_ID]

    per_pid = {}
    for _, r in responses.iterrows():
        data = list(r["data"])
        if len(data) < 3 or data[1] != 0x41:
            continue
        pid = data[2]
        if pid not in PID_TO_SIGNAL:
            continue
        sig, nbytes = PID_TO_SIGNAL[pid]
        payload = data[3:3 + nbytes]
        value = decode_pid(pid, payload)
        per_pid.setdefault(pid, []).append((float(r["timestamp_s"]), value))

    if 0x0C not in per_pid:
        raise ValueError("CAN log missing RPM (PID 0x0C) responses")

    rpm_ts = sorted(t for t, _ in per_pid[0x0C])
    out = pd.DataFrame({"timestamp_s": rpm_ts})

    for pid, samples in per_pid.items():
        sig, _ = PID_TO_SIGNAL[pid]
        df = pd.DataFrame(samples, columns=["timestamp_s", sig]).sort_values("timestamp_s")
        out = pd.merge_asof(
            out.sort_values("timestamp_s"),
            df,
            on="timestamp_s",
            direction="nearest",
        )
    # Order columns: timestamp first then signals in canonical order.
    cols = ["timestamp_s"] + [PID_TO_SIGNAL[p][0] for p in POLL_PIDS
                              if PID_TO_SIGNAL[p][0] in out.columns]
    return out[cols].reset_index(drop=True)


def write_candump_log(can_log, path):
    """Write a candump-style log: ``(timestamp) can0 ID#DATAHEX``."""
    lines = []
    for _, r in can_log.sort_values("timestamp_s").iterrows():
        data = list(r["data"])
        hexstr = "".join(f"{b & 0xFF:02X}" for b in data)
        lines.append(f"({float(r['timestamp_s']):.6f}) can0 {int(r['can_id']):03X}#{hexstr}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


_LINE_RE = re.compile(r"\(([-\d.]+)\)\s+(\S+)\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)")


def read_candump_log(path):
    """Parse a candump log back into a CAN frame log DataFrame."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            ts = float(m.group(1))
            can_id = int(m.group(3), 16)
            hexstr = m.group(4)
            data = tuple(int(hexstr[i:i + 2], 16) for i in range(0, len(hexstr), 2))
            pid = None
            kind = None
            if can_id == REQUEST_ID and len(data) >= 3:
                pid = data[2]
                kind = "request"
            elif can_id == RESPONSE_ID and len(data) >= 3:
                pid = data[2]
                kind = "response"
            rows.append({
                "timestamp_s": ts,
                "can_id": can_id,
                "dlc": len(data),
                "data": data,
                "pid": pid,
                "kind": kind,
            })
    return pd.DataFrame(rows)


# Quantization-aware default tolerances (one quantum).
ROUNDTRIP_TOL = {
    "engine_load": 100.0 / 255.0 / 100.0 + 1e-9,   # ~0.0039 (0-1 scale)
    "coolant_temp_c": 1.0 + 1e-9,
    "stft_pct": 100.0 / 128.0 + 1e-9,              # ~0.78 %
    "ltft_pct": 100.0 / 128.0 + 1e-9,
    "engine_rpm": 0.25 + 1e-9,
    "ignition_timing": 0.5 + 1e-9,                 # A/2 quantum = 0.5 deg
    "throttle_pct": 100.0 / 255.0 + 1e-9,          # ~0.39 %
}


def roundtrip_check(telemetry, tol=None):
    """Encode -> decode telemetry and return max abs error per column."""
    if tol is None:
        tol = ROUNDTRIP_TOL
    log = telemetry_to_can(telemetry)
    decoded = can_to_telemetry(log)

    # Align decoded reply rows to source rows by nearest timestamp.
    merged = pd.merge_asof(
        decoded.sort_values("timestamp_s"),
        telemetry.sort_values("timestamp_s"),
        on="timestamp_s",
        direction="nearest",
        suffixes=("_dec", "_src"),
    )

    result = {}
    ok = True
    for sig in SIGNAL_TO_PID:
        dec_col = f"{sig}_dec" if f"{sig}_dec" in merged.columns else sig
        src_col = f"{sig}_src"
        if dec_col not in merged.columns or src_col not in merged.columns:
            continue
        err = float(np.max(np.abs(merged[dec_col].to_numpy()
                                  - merged[src_col].to_numpy())))
        result[sig] = err
        if err > tol.get(sig, np.inf):
            ok = False
    result["_within_tolerance"] = ok
    return result


def main(argv=None):
    """CLI: ``python -m feature_extraction.can_codec check`` runs a round-trip."""
    import sys
    from simulator.engine_simulator import simulate_engine

    args = argv if argv is not None else sys.argv[1:]
    if args and args[0] == "check":
        all_ok = True
        for cond in ("normal", "knock", "misfire", "valve_train_wear", "overheating"):
            telemetry, _ = simulate_engine(cond, duration_s=5.0, seed=0)
            res = roundtrip_check(telemetry)
            ok = res.pop("_within_tolerance")
            all_ok = all_ok and ok
            errs = ", ".join(f"{k}={v:.5g}" for k, v in res.items())
            print(f"{cond:18s} {'OK ' if ok else 'FAIL'} max_err: {errs}")
        print("ROUND-TRIP:", "PASS" if all_ok else "FAIL")
        sys.exit(0 if all_ok else 1)
    else:
        print("usage: python -m feature_extraction.can_codec check")


if __name__ == "__main__":
    main()
