from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile


def load_wav(path: str | Path) -> Tuple[int, np.ndarray]:
    """
    Loads a WAV file, converts it to mono and normalizes the signal to [-1, 1].

    Args:
        path: Path to the WAV file.

    Returns:
        sample_rate: Sampling frequency of the audio signal.
        signal: Mono audio signal normalized to [-1, 1] as np.float32.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sample_rate, signal = wavfile.read(path)

    signal = np.asarray(signal)

    # Convert stereo/multichannel audio to mono
    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    # Convert to float32 for further processing
    signal = signal.astype(np.float32)

    # Normalize amplitude to [-1, 1]
    max_abs_value = np.max(np.abs(signal))

    if max_abs_value > 0:
        signal = signal / max_abs_value

    return sample_rate, signal
