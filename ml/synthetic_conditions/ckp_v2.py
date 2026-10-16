"""Timestamped, indexed 60-2 edge processing on a shared absolute run clock."""
import numpy as np

from .ecu_v2 import finite_number

TIMING = ("ckp_tooth_period_s_mean", "ckp_tooth_period_s_std", "ckp_tooth_period_s_cv",
          "ckp_tooth_rpm_mean", "ckp_tooth_rpm_std", "ckp_tooth_rpm_min", "ckp_tooth_rpm_max",
          "ckp_tooth_rpm_cv", "ckp_tooth_rpm_p2p", "ckp_cycle_rpm_mean", "ckp_cycle_rpm_std",
          "ckp_cycle_rpm_min", "ckp_cycle_rpm_max", "ckp_cycle_rpm_cv",
          "ckp_angular_speed_mean_rad_s", "ckp_angular_speed_std_rad_s", "ckp_angular_speed_cv")
QUALITY = ("ckp_valid_tooth_fraction", "ckp_expected_gap_count", "ckp_unexpected_gap_count",
           "ckp_min_samples_per_tooth")


def process_edges(events, *, start, end, timestamp_resolution_s, min_samples_per_tooth=10):
    """Events are (time_s, tooth_index), teeth 0..57; indices 58,59 are absent.

    Tooth indices must come from the acquisition/decoder, never condition labels.
    Resolution is the acquisition timestamp tick, not the audio sampling period.
    Cycle means one crank revolution, not a 720-degree engine cycle. At least
    one complete revolution is required. Invalid/dropout data produces null timing.
    """
    start, end, resolution, minimum = map(finite_number,
        (start, end, timestamp_resolution_s, min_samples_per_tooth))
    if end <= start or resolution <= 0 or minimum < 4:
        raise ValueError("invalid clock/window/resolution threshold")
    features = dict.fromkeys(TIMING + QUALITY)
    result = {"available": events is not None, "quality_status": "unavailable", "features": features}
    if events is None:
        return result
    events = list(events)
    times = np.array([finite_number(t) for t, _ in events])
    indices = [i for _, i in events]
    if any(type(i) is not int or not 0 <= i < 58 for i in indices):
        raise ValueError("tooth_index must be an integer in 0..57")
    if np.any(np.diff(times) <= 0):
        raise ValueError("edge timestamps must be strictly increasing")
    selected = [(t, i) for t, i in events if start <= t < end]
    result["quality_status"] = "poor_quality"
    if len(selected) < 2:
        return result
    t = np.array([p[0] for p in selected]); idx = np.array([p[1] for p in selected])
    dt = np.diff(t)
    expected_gap = (idx[:-1] == 57) & (idx[1:] == 0)
    valid = (idx[1:] == idx[:-1] + 1) | expected_gap
    pitch = np.where(expected_gap, 3.0, 1.0)
    periods = dt / pitch
    rpm = 1.0 / periods  # 60 seconds/minute divided by 60 pitches/revolution
    quality = (float(valid.mean()), int(expected_gap.sum()), int((~valid).sum()),
               float(periods.min() / resolution))
    features.update(dict(zip(QUALITY, quality)))
    # Compare repeated physical teeth: complete revolution spans exactly 58 edges.
    cycle_rpm = np.array([60 / (t[j] - t[j - 58]) for j in range(58, len(t))
                          if idx[j] == idx[j - 58] and valid[j - 58:j].all()])
    if (not valid.all() or quality[-1] < minimum or not len(cycle_rpm)
            or np.any((rpm < 300) | (rpm > 8000))
            or t[0] - start > 3 / 300 or end - t[-1] > 3 / 300):
        return result
    speed = rpm * 2 * np.pi / 60
    cv = lambda a: float(a.std() / a.mean())
    values = (periods.mean(), periods.std(), cv(periods), rpm.mean(), rpm.std(), rpm.min(),
              rpm.max(), cv(rpm), np.ptp(rpm), cycle_rpm.mean(), cycle_rpm.std(), cycle_rpm.min(),
              cycle_rpm.max(), cv(cycle_rpm), speed.mean(), speed.std(), cv(speed))
    features.update(dict(zip(TIMING, map(float, values))))
    result["quality_status"] = "good"
    return result
