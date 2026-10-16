"""ML-owned PID adapter; no simulator or teammate codec imports."""
from numbers import Real

import numpy as np


def finite_number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError("expected finite numeric measurement")
    return float(value)


def decode_pid(pid, data):
    """Decode one Mode 01 payload (data bytes only), without inventing readings."""
    if type(pid) is not int or pid not in (0x04, 0x0B):
        raise ValueError("V2 adapter supports only PID 0x04 and 0x0B")
    if not isinstance(data, (list, tuple, bytes)) or len(data) != 1:
        raise ValueError("exactly one data byte is required")
    a = data[0]
    if type(a) is not int or not 0 <= a <= 255:
        raise ValueError("invalid PID byte")
    return ({"map_kpa": float(a)} if pid == 0x0B else
            {"engine_load": a / 255.0, "engine_load_pct": a * 100.0 / 255.0})


def window_features(samples, *, start, end, family, max_age_s=0.2):
    """Summarize timestamp/value pairs in [start,end); no interpolation or carry-forward.

    A gap, including either window boundary, exceeding max_age_s is stale.
    Inputs must be strictly ordered on the common run clock, including outside
    the window. None represents an unavailable source; [] an acquired empty window.
    """
    start, end, max_age_s = map(finite_number, (start, end, max_age_s))
    if end <= start or max_age_s <= 0 or family not in ("map_kpa", "engine_load_pct"):
        raise ValueError("invalid window, staleness limit, or family")
    prefix = "tel_" + family
    features = {prefix + "_" + stat: None for stat in ("mean", "std", "min", "max", "slope")}
    result = {"available": samples is not None, "quality_status": "unavailable",
              "sample_count": 0, "features": features}
    if samples is None:
        return result
    pairs = list(samples)
    times = np.array([finite_number(t) for t, _ in pairs])
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    selected = [(t, v) for t, v in pairs if start <= t < end]
    result["sample_count"] = len(selected)
    values = []
    for _, value in selected:
        value = finite_number(value)
        if not 0 <= value <= (255 if family == "map_kpa" else 100):
            raise ValueError("measurement outside PID range")
        values.append(value)
    result["quality_status"] = "insufficient_samples"
    if len(selected) < 2:
        return result
    t = np.array([p[0] for p in selected], dtype=float)
    if np.max(np.diff(np.r_[start, t, end])) > max_age_s + 1e-12:
        result["quality_status"] = "stale"
        return result
    v = np.asarray(values)
    centered = t - t.mean()
    stats = (v.mean(), v.std(), v.min(), v.max(), np.dot(centered, v - v.mean()) / np.dot(centered, centered))
    result["quality_status"] = "good"
    result["features"] = dict(zip(features, map(float, stats)))
    return result
