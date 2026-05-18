import numpy as np


def next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("Value must be positive.")

    return 1 << (value - 1).bit_length()


def compute_power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    if frames.ndim != 2:
        raise ValueError("Frames must be a 2D array.")

    complex_spectrum = np.fft.rfft(frames, n=n_fft)
    magnitude_spectrum = np.abs(complex_spectrum)
    power_spectrum = (1.0 / n_fft) * (magnitude_spectrum ** 2)

    return power_spectrum
