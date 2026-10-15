import numpy as np

from simulator.signal_analysis import (
    frequency_spectrum,
    time_frequency_spectrogram,
)


def test_fft_detects_known_frequency():
    sample_rate = 4000
    duration = 1.0
    expected_frequency = 200

    t = np.arange(int(sample_rate * duration)) / sample_rate
    signal = np.sin(2 * np.pi * expected_frequency * t)

    frequencies, amplitude = frequency_spectrum(signal, sample_rate)
    peak_frequency = frequencies[np.argmax(amplitude)]

    assert peak_frequency == expected_frequency


def test_spectrogram_has_valid_dimensions():
    sample_rate = 4000
    t = np.arange(sample_rate) / sample_rate
    signal = np.sin(2 * np.pi * 200 * t)

    times, frequencies, power = time_frequency_spectrogram(
        signal, sample_rate
    )

    assert len(times) > 0
    assert len(frequencies) > 0
    assert power.shape == (len(frequencies), len(times))
    assert frequencies.max() <= sample_rate / 2



import pytest


@pytest.mark.parametrize(
    "signal",
    [
        np.ones(8),
        np.ones((32, 2)),
        np.full(32, np.nan),
    ],
)
def test_fft_rejects_invalid_signals(signal):
    with pytest.raises(ValueError):
        frequency_spectrum(signal, 4000)


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_fft_rejects_invalid_sample_rates(sample_rate):
    with pytest.raises(ValueError):
        frequency_spectrum(np.ones(32), sample_rate)


def test_spectrogram_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        time_frequency_spectrogram(
            np.ones(1024),
            4000,
            overlap_fraction=1.0,
        )
