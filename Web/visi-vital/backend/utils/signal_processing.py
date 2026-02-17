import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.7, high: float = 4.0, order: int = 3):
    if len(signal) < 10:
        return signal

    nyq = 0.5 * fs
    low_cut = max(low / nyq, 1e-4)
    high_cut = min(high / nyq, 0.9999)
    if low_cut >= high_cut:
        return signal

    b, a = butter(order, [low_cut, high_cut], btype="band")
    return filtfilt(b, a, signal)


def normalize_signal(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std
