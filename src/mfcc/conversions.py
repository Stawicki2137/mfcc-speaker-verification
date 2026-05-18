from typing import Optional

import numpy as np


def hz_to_mel(frequency_hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency_hz / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
