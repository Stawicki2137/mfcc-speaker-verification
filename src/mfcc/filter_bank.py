from typing import Optional

import numpy as np

from .conversions import hz_to_mel, mel_to_hz
from .constants import EPSILON


def create_mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    num_filters: int = 26,
    low_freq_hz: float = 0.0,
    high_freq_hz: Optional[float] = None
) -> np.ndarray:
    if high_freq_hz is None:
        high_freq_hz = sample_rate / 2.0

    if low_freq_hz < 0:
        raise ValueError("low_freq_hz cannot be negative.")

    if high_freq_hz > sample_rate / 2.0:
        raise ValueError("high_freq_hz cannot be greater than Nyquist frequency.")

    if low_freq_hz >= high_freq_hz:
        raise ValueError("low_freq_hz must be smaller than high_freq_hz.")

    low_mel = hz_to_mel(low_freq_hz)
    high_mel = hz_to_mel(high_freq_hz)

    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)

    fft_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    max_bin = n_fft // 2
    fft_bins = np.clip(fft_bins, 0, max_bin)

    filter_bank = np.zeros((num_filters, max_bin + 1), dtype=np.float32)

    for filter_index in range(1, num_filters + 1):
        left_bin = fft_bins[filter_index - 1]
        center_bin = fft_bins[filter_index]
        right_bin = fft_bins[filter_index + 1]

        if center_bin == left_bin:
            center_bin += 1

        if right_bin == center_bin:
            right_bin += 1

        center_bin = min(center_bin, max_bin)
        right_bin = min(right_bin, max_bin)

        for bin_index in range(left_bin, center_bin):
            filter_bank[filter_index - 1, bin_index] = (
                (bin_index - left_bin) / (center_bin - left_bin)
            )

        for bin_index in range(center_bin, right_bin):
            filter_bank[filter_index - 1, bin_index] = (
                (right_bin - bin_index) / (right_bin - center_bin)
            )

    return filter_bank


def apply_mel_filter_bank(
    power_spectrum: np.ndarray,
    filter_bank: np.ndarray
) -> np.ndarray:
    mel_energies = np.dot(power_spectrum, filter_bank.T)
    mel_energies = np.where(mel_energies <= 0, EPSILON, mel_energies)

    return mel_energies


def compute_log_mel_energies(mel_energies: np.ndarray) -> np.ndarray:
    return np.log(mel_energies + EPSILON)
