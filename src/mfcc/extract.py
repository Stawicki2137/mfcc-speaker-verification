from typing import Optional

import numpy as np

from preprocessing import preprocess_signal
from .dct import compute_dct_type_2
from .filter_bank import apply_mel_filter_bank, compute_log_mel_energies, create_mel_filter_bank
from .spectrum import compute_power_spectrum, next_power_of_two


def extract_mfcc(
    signal: np.ndarray,
    sample_rate: int,
    num_coefficients: int = 13,
    num_filters: int = 26,
    frame_size_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    pre_emphasis_coefficient: float = 0.97,
    n_fft: Optional[int] = None,
    low_freq_hz: float = 0.0,
    high_freq_hz: Optional[float] = None,
    remove_c0: bool = True
) -> np.ndarray:
    _, windowed_frames = preprocess_signal(
        signal=signal,
        sample_rate=sample_rate,
        normalize=True,
        remove_silence_flag=False,
        pre_emphasis_coefficient=pre_emphasis_coefficient,
        frame_size_ms=frame_size_ms,
        frame_step_ms=frame_step_ms
    )

    frame_length = windowed_frames.shape[1]

    if n_fft is None:
        n_fft = next_power_of_two(frame_length)

    if n_fft < frame_length:
        raise ValueError("n_fft must be greater than or equal to frame length.")

    power_spectrum = compute_power_spectrum(
        frames=windowed_frames,
        n_fft=n_fft
    )

    mel_filter_bank = create_mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        num_filters=num_filters,
        low_freq_hz=low_freq_hz,
        high_freq_hz=high_freq_hz
    )

    mel_energies = apply_mel_filter_bank(
        power_spectrum=power_spectrum,
        filter_bank=mel_filter_bank
    )

    log_mel_energies = compute_log_mel_energies(mel_energies)

    coefficients_to_compute = num_coefficients + 1 if remove_c0 else num_coefficients

    cepstral_coefficients = compute_dct_type_2(
        log_mel_energies=log_mel_energies,
        num_coefficients=coefficients_to_compute
    )

    if remove_c0:
        mfcc = cepstral_coefficients[:, 1:num_coefficients + 1]
    else:
        mfcc = cepstral_coefficients[:, :num_coefficients]

    return mfcc.astype(np.float32)
