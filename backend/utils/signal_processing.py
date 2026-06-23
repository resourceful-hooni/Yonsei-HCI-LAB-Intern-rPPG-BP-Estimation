"""Signal-processing utilities for rPPG (pure NumPy/SciPy — no cv2/mediapipe).

Kept dependency-light on purpose so the math can be unit-tested by
``backend/tests/test_signal_pipeline.py`` without importing the camera stack.

Public API:
  - normalize_signal(x)                  z-score with NaN/Inf guards
  - detrend_signal(x)                    linear detrend with guards
  - bandpass_filter(signal, fs, ...)     Butterworth band-pass (filtfilt)
  - pos_rppg(rgb, fs)                    POS pulse extraction (Wang 2017)
  - chrom_rppg(rgb)                      CHROM-like pulse extraction (fallback)
  - estimate_hr(ppg, fs, ...)            heart rate from PSD peak -> (bpm, ok)
  - signal_snr(ppg, fs, ...)            rPPG SNR in dB around the HR peak
  - effective_fps(timestamps)            mean sampling rate from frame stamps
  - resample_uniform(values, ts, fs)     interpolate onto a uniform time grid
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch

# Plausible human heart-rate band (Hz). 0.7-3.5 Hz == 42-210 bpm.
HR_LOW_HZ = 0.7
HR_HIGH_HZ = 3.5
_EPS = 1e-8


def _finite(x: np.ndarray) -> np.ndarray:
    """Coerce to float64 and replace NaN/Inf with 0 so downstream math is safe."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Z-score normalize. Returns zeros for a flat/degenerate signal."""
    x = _finite(x)
    std = np.std(x)
    if std < _EPS:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def detrend_signal(x: np.ndarray) -> np.ndarray:
    """Remove a linear trend (slow lighting drift) before pulse extraction."""
    x = _finite(x)
    if len(x) < 3:
        return x - np.mean(x) if len(x) else x
    try:
        return detrend(x, type="linear")
    except Exception:
        return x - np.mean(x)


def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.7, high: float = 4.0, order: int = 3):
    """Zero-phase Butterworth band-pass. Returns the input unchanged if it is
    too short or the band is degenerate for the given sampling rate."""
    signal = _finite(signal)
    if len(signal) < 10 or not np.isfinite(fs) or fs <= 0:
        return signal

    nyq = 0.5 * fs
    low_cut = max(low / nyq, 1e-4)
    high_cut = min(high / nyq, 0.9999)
    if low_cut >= high_cut:
        return signal

    b, a = butter(order, [low_cut, high_cut], btype="band")
    # filtfilt needs len > 3*max(len(a), len(b)); guard to avoid a hard error.
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return signal
    return filtfilt(b, a, signal)


def pos_rppg(rgb: np.ndarray, fs: float) -> np.ndarray:
    """POS (Plane-Orthogonal-to-Skin, Wang et al. 2017) pulse extraction.

    rgb: (T, 3) array of per-frame mean R,G,B for the ROI.
    Returns a 1-D pulse signal of length T. More robust to motion/illumination
    than the previous CHROM-like variant, which is retained as a fallback.
    """
    rgb = _finite(rgb)
    if rgb.ndim != 2 or rgb.shape[1] < 3:
        return np.zeros(rgb.shape[0] if rgb.ndim >= 1 else 0)
    T = rgb.shape[0]
    if T < 2:
        return np.zeros(T)

    win = int(np.ceil(1.6 * fs)) if np.isfinite(fs) and fs > 0 else T
    win = max(2, min(win, T))

    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)
    H = np.zeros(T, dtype=np.float64)

    for n in range(win - 1, T):
        m = n - win + 1
        block = rgb[m : n + 1, :3]              # (win, 3)
        mu = block.mean(axis=0)
        mu[np.abs(mu) < _EPS] = _EPS
        cn = (block / mu).T                     # (3, win) temporally normalized
        s = proj @ cn                           # (2, win)
        s1_std = np.std(s[1])
        alpha = (np.std(s[0]) / s1_std) if s1_std > _EPS else 0.0
        h = s[0] + alpha * s[1]
        h = h - np.mean(h)
        H[m : n + 1] += h

    return H


def chrom_rppg(rgb: np.ndarray) -> np.ndarray:
    """CHROM-like pulse extraction (the project's original method). Kept as a
    fallback / comparison baseline for the POS extractor above."""
    rgb = _finite(rgb)
    if rgb.ndim != 2 or rgb.shape[1] < 3:
        return np.zeros(rgb.shape[0] if rgb.ndim >= 1 else 0)
    r = normalize_signal(rgb[:, 0])
    g = normalize_signal(rgb[:, 1])
    b = normalize_signal(rgb[:, 2])
    x_comp = 3.0 * r - 2.0 * g
    y_comp = 1.5 * r + g - 1.5 * b
    y_std = np.std(y_comp)
    alpha = np.std(x_comp) / y_std if y_std > _EPS else 0.0
    return x_comp - alpha * y_comp


def _psd(ppg: np.ndarray, fs: float, nfft_min: int = 2048):
    nperseg = int(min(len(ppg), 256))
    if nperseg < 8:
        nperseg = len(ppg)
    # Zero-pad (nfft >= nperseg) for finer frequency resolution. At 30 fps / 10 s
    # the raw bin width is ~7 bpm; padding interpolates the peak to <1 bpm so the
    # HR estimate isn't quantized to coarse bins.
    nfft = max(nperseg, nfft_min)
    freqs, psd = welch(ppg, fs=fs, nperseg=nperseg, nfft=nfft)
    return freqs, psd


def estimate_hr(ppg: np.ndarray, fs: float, low: float = HR_LOW_HZ, high: float = HR_HIGH_HZ):
    """Estimate heart rate (bpm) from the dominant spectral peak in the HR band.

    More robust than time-domain peak counting on noisy rPPG. Returns
    ``(hr_bpm, ok)`` where ``ok`` is False when the signal is too short/flat or
    the peak lands outside a plausible physiological range (caller should then
    lower confidence instead of trusting a default).
    """
    ppg = _finite(ppg)
    if len(ppg) < 16 or np.std(ppg) < _EPS or not np.isfinite(fs) or fs <= 0:
        return 75.0, False

    freqs, psd = _psd(ppg, fs)
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band) or np.sum(psd[band]) <= 0:
        return 75.0, False

    peak_freq = float(freqs[band][int(np.argmax(psd[band]))])
    hr = peak_freq * 60.0
    ok = 40.0 <= hr <= 200.0
    return float(hr), bool(ok)


def signal_snr(ppg: np.ndarray, fs: float, low: float = HR_LOW_HZ, high: float = 4.0) -> float:
    """rPPG signal-to-noise ratio (dB): power within ±0.12 Hz of the HR peak and
    its first harmonic vs. the remaining in-band power (de Haan & van Leest style).
    Higher is cleaner. Returns 0.0 for unusable signals."""
    ppg = _finite(ppg)
    if len(ppg) < 16 or np.std(ppg) < _EPS or not np.isfinite(fs) or fs <= 0:
        return 0.0

    freqs, psd = _psd(ppg, fs)
    rng = (freqs >= low) & (freqs <= high)
    if not np.any(rng) or np.sum(psd[rng]) <= 0:
        return 0.0

    f = freqs[rng]
    p = psd[rng]
    peak_f = float(f[int(np.argmax(p))])
    sig_mask = (np.abs(f - peak_f) <= 0.12) | (np.abs(f - 2.0 * peak_f) <= 0.12)
    sig_power = float(np.sum(p[sig_mask]))
    noise_power = float(np.sum(p[~sig_mask]))
    if noise_power <= _EPS:
        return 10.0
    if sig_power <= _EPS:
        return 0.0
    return float(10.0 * np.log10(sig_power / noise_power))


def effective_fps(timestamps) -> float:
    """Mean sampling rate (Hz) from monotonic per-frame timestamps (seconds).
    Returns 0.0 if it can't be determined (caller falls back to nominal fps)."""
    t = _finite(timestamps)
    if t.ndim != 1 or len(t) < 2:
        return 0.0
    duration = float(t[-1] - t[0])
    if duration <= 0:
        return 0.0
    return float((len(t) - 1) / duration)


def resample_uniform(values: np.ndarray, timestamps, fs_target: float) -> np.ndarray:
    """Interpolate ``values`` (1-D or (T, C)) sampled at irregular ``timestamps``
    (seconds) onto a uniform grid at ``fs_target`` Hz. Used to correct for the
    browser's non-uniform frame delivery before spectral analysis."""
    values = _finite(values)
    t = _finite(timestamps)
    if len(t) != values.shape[0] or len(t) < 2 or not np.isfinite(fs_target) or fs_target <= 0:
        return values

    t = t - t[0]
    duration = float(t[-1])
    if duration <= 0:
        return values
    n = max(2, int(round(duration * fs_target)))
    grid = np.linspace(0.0, duration, n)

    if values.ndim == 1:
        return np.interp(grid, t, values)
    return np.column_stack([np.interp(grid, t, values[:, c]) for c in range(values.shape[1])])
