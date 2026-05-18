from pathlib import Path
from typing import Tuple

import numpy as np

from audio_loader import load_wav
from mfcc import extract_mfcc


def aggregate_mfcc_features(mfcc_features: np.ndarray) -> np.ndarray:
    """
    Aggregates MFCC features from all frames into one feature vector.

    For each MFCC coefficient, we compute:
        - mean value across all frames
        - standard deviation across all frames

    If MFCC matrix has shape:
        (num_frames, num_coefficients)

    Then output vector has shape:
        (2 * num_coefficients,)

    Example:
        (1828, 13) -> (26,)

    Args:
        mfcc_features: MFCC matrix with shape (num_frames, num_coefficients).

    Returns:
        Aggregated feature vector.
    """
    if mfcc_features.ndim != 2:
        raise ValueError("MFCC features must be a 2D array.")

    if mfcc_features.shape[0] == 0:
        raise ValueError("MFCC features cannot have zero frames.")

    mean_features = np.mean(mfcc_features, axis=0)
    std_features = np.std(mfcc_features, axis=0)

    feature_vector = np.concatenate([mean_features, std_features])

    return feature_vector.astype(np.float32)


def extract_feature_vector_from_signal(
    signal: np.ndarray,
    sample_rate: int,
    num_coefficients: int = 13,
    num_filters: int = 26,
    frame_size_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    pre_emphasis_coefficient: float = 0.97,
    remove_c0: bool = True
) -> np.ndarray:
    """
    Extracts one feature vector from a raw audio signal.

    Pipeline:
        signal
        -> MFCC matrix
        -> mean + std aggregation
        -> one feature vector

    Args:
        signal: Audio signal.
        sample_rate: Sampling frequency in Hz.
        num_coefficients: Number of MFCC coefficients.
        num_filters: Number of Mel filters.
        frame_size_ms: Frame length in milliseconds.
        frame_step_ms: Step between frames in milliseconds.
        pre_emphasis_coefficient: Pre-emphasis coefficient.
        remove_c0: Whether to remove c0 coefficient.

    Returns:
        One feature vector for the whole recording.
    """
    mfcc_features = extract_mfcc(
        signal=signal,
        sample_rate=sample_rate,
        num_coefficients=num_coefficients,
        num_filters=num_filters,
        frame_size_ms=frame_size_ms,
        frame_step_ms=frame_step_ms,
        pre_emphasis_coefficient=pre_emphasis_coefficient,
        remove_c0=remove_c0
    )

    return aggregate_mfcc_features(mfcc_features)


def extract_feature_vector_from_file(
    wav_path: str | Path,
    num_coefficients: int = 13,
    num_filters: int = 26,
    frame_size_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    pre_emphasis_coefficient: float = 0.97,
    remove_c0: bool = True
) -> Tuple[int, np.ndarray]:
    """
    Loads WAV file and extracts one feature vector.

    Args:
        wav_path: Path to WAV file.
        num_coefficients: Number of MFCC coefficients.
        num_filters: Number of Mel filters.
        frame_size_ms: Frame length in milliseconds.
        frame_step_ms: Step between frames in milliseconds.
        pre_emphasis_coefficient: Pre-emphasis coefficient.
        remove_c0: Whether to remove c0 coefficient.

    Returns:
        sample_rate: Sampling frequency in Hz.
        feature_vector: Aggregated feature vector.
    """
    sample_rate, signal = load_wav(wav_path)

    feature_vector = extract_feature_vector_from_signal(
        signal=signal,
        sample_rate=sample_rate,
        num_coefficients=num_coefficients,
        num_filters=num_filters,
        frame_size_ms=frame_size_ms,
        frame_step_ms=frame_step_ms,
        pre_emphasis_coefficient=pre_emphasis_coefficient,
        remove_c0=remove_c0
    )

    return sample_rate, feature_vector


def get_feature_names(num_coefficients: int = 13) -> list[str]:
    """
    Returns names of aggregated MFCC features.

    Example:
        mean_mfcc_1, ..., mean_mfcc_13,
        std_mfcc_1, ..., std_mfcc_13
    """
    mean_names = [f"mean_mfcc_{i}" for i in range(1, num_coefficients + 1)]
    std_names = [f"std_mfcc_{i}" for i in range(1, num_coefficients + 1)]

    return mean_names + std_names