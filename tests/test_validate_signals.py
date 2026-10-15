from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulator.validate_signals import CONDITIONS, rms


def test_conditions_contains_all_five_classes():
    assert set(CONDITIONS) == {
        "normal",
        "misfire",
        "knock",
        "valve_train_wear",
        "overheating",
    }


def test_rms_calculates_root_mean_square():
    assert rms([3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


def test_rms_of_zero_signal_is_zero():
    assert rms([0.0, 0.0, 0.0]) == 0.0


def test_rms_handles_numpy_array():
    values = np.array([1.0, 1.0, 1.0])
    assert rms(values) == pytest.approx(1.0)


def test_rms_of_negative_values():
    assert rms([-3.0, -4.0]) == pytest.approx(np.sqrt(12.5))
