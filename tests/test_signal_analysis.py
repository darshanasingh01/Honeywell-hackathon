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
