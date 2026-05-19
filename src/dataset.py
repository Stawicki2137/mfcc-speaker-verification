from pathlib import Path
from typing import Tuple

import numpy as np

from features import extract_feature_vector_from_file


TARGET_LABEL = 1
OTHER_LABEL = 0


def find_wav_files(directory: str | Path) -> list[Path]:
    """
    Finds all WAV files in a given directory.

    Args:
        directory: Path to directory.

    Returns:
        List of WAV file paths.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(directory.glob("*.wav"))


def load_dataset(
    target_dir: str | Path,
    others_dir: str | Path,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, list[Path]]:
    """
    Builds dataset from target and others directories.

    Each WAV file is converted into one feature vector.

    Args:
        target_dir: Directory with recordings of the target speaker.
        others_dir: Directory with recordings of other speakers.

    Returns:
        X: Feature matrix with shape (num_samples, num_features).
        y: Labels with shape (num_samples,).
           1 means target speaker, 0 means other speaker.
        file_paths: List of file paths in the same order as X and y.
    """
    target_files = find_wav_files(target_dir)
    other_files = find_wav_files(others_dir)

    if len(target_files) == 0:
        raise ValueError(f"No WAV files found in target directory: {target_dir}")

    if len(other_files) == 0:
        raise ValueError(f"No WAV files found in others directory: {others_dir}")

    X = []
    y = []
    file_paths = []

    for wav_path in target_files:
        _, feature_vector = extract_feature_vector_from_file(wav_path, **kwargs)

        X.append(feature_vector)
        y.append(TARGET_LABEL)
        file_paths.append(wav_path)

    for wav_path in other_files:
        _, feature_vector = extract_feature_vector_from_file(wav_path, **kwargs)

        X.append(feature_vector)
        y.append(OTHER_LABEL)
        file_paths.append(wav_path)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y, file_paths


def print_dataset_summary(
    X: np.ndarray,
    y: np.ndarray,
    file_paths: list[Path]
) -> None:
    """
    Prints basic information about the dataset.
    """
    print("Dataset summary")
    print("----------------")
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Target samples:", int(np.sum(y == TARGET_LABEL)))
    print("Other samples:", int(np.sum(y == OTHER_LABEL)))

    print("\nFiles:")
    for path, label in zip(file_paths, y):
        label_name = "target" if label == TARGET_LABEL else "other"
        print(f"{label_name:>6} | {path}")