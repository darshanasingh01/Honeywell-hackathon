import numpy as np
import pandas as pd
import pytest

from simulator.engine_simulator import simulate_engine
from feature_extraction.can_codec import (
    encode_pid,
    decode_pid,
    telemetry_to_can,
    can_to_telemetry,
    write_candump_log,
    read_candump_log,
    roundtrip_check,
    REQUEST_ID,
    RESPONSE_ID,
    POLL_PIDS,
)


def test_pid_formula_coolant():
    # 0x05: A - 40
    assert encode_pid(0x05, 90.0) == (130,)
    assert decode_pid(0x05, (130,)) == 90.0


def test_pid_formula_rpm_two_bytes():
    # 0x0C: (256A + B) / 4
    data = encode_pid(0x0C, 1500.0)
    assert len(data) == 2
    assert abs(decode_pid(0x0C, data) - 1500.0) <= 0.25


def test_pid_formula_throttle_and_load():
    a = encode_pid(0x11, 40.0)
    assert abs(decode_pid(0x11, a) - 40.0) <= 100.0 / 255.0 + 1e-6
    a = encode_pid(0x04, 0.5)  # stored 0-1
    assert abs(decode_pid(0x04, a) - 0.5) <= (100.0 / 255.0) / 100.0 + 1e-6


def test_fuel_trim_signed():
    a = encode_pid(0x06, -5.0)
    assert abs(decode_pid(0x06, a) - (-5.0)) <= 100.0 / 128.0 + 1e-6


def test_pid_formula_ignition_timing():
    # 0x0E: A/2 - 64 (quantum 0.5 deg BTDC)
    for deg in (8.0, 13.0, 15.0, 20.0):
        a = encode_pid(0x0E, deg)
        assert len(a) == 1
        assert abs(decode_pid(0x0E, a) - deg) <= 0.5 + 1e-9


def test_poll_pids_includes_ignition_timing():
    assert 0x0E in POLL_PIDS
    assert len(POLL_PIDS) == 7


def test_frame_structure():
    telemetry, _ = simulate_engine("normal", duration_s=1.0, seed=0)
    log = telemetry_to_can(telemetry)
    assert set(log["can_id"].unique()) == {REQUEST_ID, RESPONSE_ID}
    # one request + one response per PID per telemetry row
    assert len(log) == 2 * len(POLL_PIDS) * len(telemetry)
    # responses carry the 0x41 positive-response mode byte
    resp = log[log["can_id"] == RESPONSE_ID].iloc[0]["data"]
    assert resp[1] == 0x41


def test_candump_log_roundtrip(tmp_path):
    telemetry, _ = simulate_engine("knock", duration_s=1.0, seed=1)
    log = telemetry_to_can(telemetry)
    path = tmp_path / "run.log"
    write_candump_log(log, path)
    reread = read_candump_log(path)
    assert len(reread) == len(log)
    assert set(reread["can_id"].unique()) == {REQUEST_ID, RESPONSE_ID}


def test_can_to_telemetry_decodes(tmp_path):
    telemetry, _ = simulate_engine("overheating", duration_s=2.0, seed=2)
    log = telemetry_to_can(telemetry)
    decoded = can_to_telemetry(log)
    assert "engine_rpm" in decoded.columns
    assert "coolant_temp_c" in decoded.columns
    assert len(decoded) == len(telemetry)


def test_roundtrip_within_quantization_tolerance():
    for cond in ("normal", "knock", "misfire", "valve_train_wear", "overheating"):
        telemetry, _ = simulate_engine(cond, duration_s=3.0, seed=0)
        res = roundtrip_check(telemetry)
        assert res.pop("_within_tolerance"), f"{cond} exceeded tolerance: {res}"
