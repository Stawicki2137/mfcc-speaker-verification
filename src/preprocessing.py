from typing import Tuple

import numpy as np


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalizes audio signal to the range [-1, 1].

    Args:
        signal: Input audio signal.

    Returns:
        Normalized signal as np.float32.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        return signal

    max_abs_value = np.max(np.abs(signal))

    if max_abs_value == 0:
        return signal

    return signal / max_abs_value


def remove_silence(
    signal: np.ndarray,
    threshold_ratio: float = 0.02,
    padding: int = 1000
) -> np.ndarray:
    """
    Removes silence from the beginning and the end of the signal.

    This is a simple energy/amplitude-based silence removal.
    It keeps samples whose absolute amplitude is above a threshold.

    Args:
        signal: Input audio signal.
        threshold_ratio: Silence threshold relative to max amplitude.
        padding: Number of samples kept before and after detected speech.

    Returns:
        Signal with leading and trailing silence removed.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        return signal

    max_abs_value = np.max(np.abs(signal))

    if max_abs_value == 0:
        return signal

    threshold = threshold_ratio * max_abs_value
    indices = np.where(np.abs(signal) > threshold)[0]

    if indices.size == 0:
        return signal

    start = max(indices[0] - padding, 0)
    end = min(indices[-1] + padding + 1, len(signal))

    return signal[start:end]


def pre_emphasis(signal: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    """
    Applies pre-emphasis filter to the signal.

    Formula:
        y[n] = x[n] - coefficient * x[n - 1]

    Args:
        signal: Input audio signal.
        coefficient: Pre-emphasis coefficient. Usually around 0.97.

    Returns:
        Pre-emphasized signal.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        return signal

    emphasized_signal = np.empty_like(signal)
    emphasized_signal[0] = signal[0]
    emphasized_signal[1:] = signal[1:] - coefficient * signal[:-1]

    return emphasized_signal


def frame_signal(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 25.0,
    frame_step_ms: float = 10.0
) -> np.ndarray:
    """
    Splits signal into overlapping frames.

    Args:
        signal: Input audio signal.
        sample_rate: Sampling frequency in Hz.
        frame_size_ms: Frame length in milliseconds.
        frame_step_ms: Step between consecutive frames in milliseconds.

    Returns:
        2D array of frames with shape: (num_frames, frame_length).
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        raise ValueError("Cannot frame an empty signal.")

    frame_length = int(round(sample_rate * frame_size_ms / 1000.0))
    frame_step = int(round(sample_rate * frame_step_ms / 1000.0))

    if frame_length <= 0:
        raise ValueError("Frame length must be greater than 0.")

    if frame_step <= 0:
        raise ValueError("Frame step must be greater than 0.")

    signal_length = len(signal)

    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((signal_length - frame_length) / frame_step))

    padded_length = (num_frames - 1) * frame_step + frame_length
    padding_length = padded_length - signal_length

    padded_signal = np.append(signal, np.zeros(padding_length, dtype=np.float32))

    indices = (
        np.tile(np.arange(frame_length), (num_frames, 1))
        + np.tile(np.arange(num_frames) * frame_step, (frame_length, 1)).T
    )

    frames = padded_signal[indices]

    return frames.astype(np.float32)


def apply_hamming_window(frames: np.ndarray) -> np.ndarray:
    """
    Applies Hamming window to each frame.

    Args:
        frames: 2D array of frames.

    Returns:
        Windowed frames.
    """
    if frames.ndim != 2:
        raise ValueError("Frames must be a 2D array.")

    frame_length = frames.shape[1]
    hamming_window = np.hamming(frame_length).astype(np.float32)

    return frames * hamming_window


def preprocess_signal(
    signal: np.ndarray,
    sample_rate: int,
    normalize: bool = True,
    remove_silence_flag: bool = False,
    pre_emphasis_coefficient: float = 0.97,
    frame_size_ms: float = 25.0,
    frame_step_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline before MFCC extraction.

    Pipeline:
        1. Normalization
        2. Optional silence removal
        3. Pre-emphasis
        4. Framing
        5. Hamming window

    Args:
        signal: Input audio signal.
        sample_rate: Sampling frequency in Hz.
        normalize: Whether to normalize the signal.
        remove_silence_flag: Whether to remove leading/trailing silence.
        pre_emphasis_coefficient: Pre-emphasis coefficient.
        frame_size_ms: Frame length in milliseconds.
        frame_step_ms: Step between frames in milliseconds.

    Returns:
        emphasized_signal: Signal after normalization, optional silence removal and pre-emphasis.
        windowed_frames: Frames after applying Hamming window.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if normalize:
        signal = normalize_signal(signal)

    if remove_silence_flag:
        signal = remove_silence(signal)

    emphasized_signal = pre_emphasis(
        signal,
        coefficient=pre_emphasis_coefficient
    )

    frames = frame_signal(
        emphasized_signal,
        sample_rate=sample_rate,
        frame_size_ms=frame_size_ms,
        frame_step_ms=frame_step_ms
    )

    windowed_frames = apply_hamming_window(frames)

    return emphasized_signal, windowed_frames


