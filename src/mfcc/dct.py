import numpy as np

from .constants import EPSILON


def compute_dct_type_2(
    log_mel_energies: np.ndarray,
    num_coefficients: int
) -> np.ndarray:
    if log_mel_energies.ndim != 2:
        raise ValueError("log_mel_energies must be a 2D array.")

    num_filters = log_mel_energies.shape[1]
    filter_indices = np.arange(num_filters)
    coefficient_indices = np.arange(num_coefficients).reshape(-1, 1)

    dct_matrix = np.cos(
        np.pi / num_filters * coefficient_indices * (filter_indices + 0.5)
    )

    cepstral_coefficients = np.dot(log_mel_energies, dct_matrix.T)

    return cepstral_coefficients.astype(np.float32)
