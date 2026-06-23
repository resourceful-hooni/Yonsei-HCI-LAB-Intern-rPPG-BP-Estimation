from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from utils.signal_processing import (
    bandpass_filter,
    chrom_rppg,
    detrend_signal,
    effective_fps,
    estimate_hr,
    normalize_signal,
    pos_rppg,
    resample_uniform,
    signal_snr,
)


FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]

# HRV frequency-domain analysis (LF/HF) needs minutes of beat data; a ~10 s
# capture cannot resolve the LF band (0.04-0.15 Hz). We therefore keep a neutral
# placeholder for backward-compatible callers and flag it as unavailable rather
# than computing a meaningless ratio from the band-passed pulse (the old bug).
_NEUTRAL_LF_HF = 1.2


class RPPGEstimator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _extract_forehead_rgb(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        points = []

        for idx in FOREHEAD_LANDMARKS:
            lm = landmarks[idx]
            x = int(np.clip(lm.x * w, 0, w - 1))
            y = int(np.clip(lm.y * h, 0, h - 1))
            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        roi_pixels = frame_rgb[mask == 255]
        if roi_pixels.size == 0:
            return None

        return roi_pixels.mean(axis=0)

    def extract_signal(self, frames: list[np.ndarray], frame_rate: float, timestamps=None):
        """Extract a band-passed rPPG pulse from face frames.

        Returns a dict: {ppg, fs, frame_ratio, method}.
          - ``fs`` is the *effective* sampling rate actually used (corrected from
            real frame timestamps when provided), not the nominal frame_rate.
          - ``method`` is "pos" (default) or "chrom" (fallback).
        """
        rgb_series = []
        kept_ts = []
        has_ts = timestamps is not None and len(timestamps) == len(frames)

        for i, frame in enumerate(frames):
            rgb_mean = self._extract_forehead_rgb(frame)
            if rgb_mean is not None:
                rgb_series.append(rgb_mean)
                if has_ts:
                    kept_ts.append(float(timestamps[i]))

        if len(rgb_series) < max(30, int(frame_rate * 3)):
            raise ValueError("유효한 얼굴 ROI 프레임이 충분하지 않습니다.")

        rgb = np.asarray(rgb_series, dtype=np.float64)
        fs = float(frame_rate)

        # A1: use real frame timing. The browser's setInterval is jittery, so the
        # nominal 30 fps is unreliable; resample onto a uniform grid at the
        # measured rate before any spectral analysis.
        if has_ts and len(kept_ts) == len(rgb):
            eff = effective_fps(kept_ts)
            if eff > 0:
                target = float(max(1.0, round(eff)))
                rgb = resample_uniform(rgb, kept_ts, target)
                fs = target

        # Detrend each channel (removes slow lighting drift) before extraction.
        rgb = np.column_stack([detrend_signal(rgb[:, c]) for c in range(3)])

        # POS is more robust to motion/illumination than the old CHROM variant;
        # fall back to CHROM if POS yields a degenerate (flat) signal.
        method = "pos"
        pulse = pos_rppg(rgb, fs)
        if np.std(pulse) < 1e-8:
            pulse = chrom_rppg(rgb)
            method = "chrom"

        ppg = bandpass_filter(pulse, fs=fs, low=0.7, high=4.0, order=3)
        ppg = normalize_signal(ppg)

        frame_ratio = float(np.clip(len(rgb_series) / max(len(frames), 1), 0.0, 1.0))
        return {"ppg": ppg, "fs": fs, "frame_ratio": frame_ratio, "method": method}

    def extract_features(self, ppg: np.ndarray, frame_rate: float, quality: float = 0.7):
        ppg = np.asarray(ppg, dtype=np.float64)
        fs = float(frame_rate)

        features = {
            "mean": float(np.mean(ppg)) if ppg.size else 0.0,
            "std": float(np.std(ppg)) if ppg.size else 0.0,
            "skewness": float(skew(ppg)) if len(ppg) > 3 else 0.0,
            "kurtosis": float(kurtosis(ppg)) if len(ppg) > 3 else 0.0,
            "quality": float(quality),
        }

        # Primary HR: dominant spectral peak in the HR band (robust to noise).
        hr_psd, hr_ok = estimate_hr(ppg, fs)

        # Cross-check with time-domain peaks; also the source for IBI/RMSSD.
        std = float(np.std(ppg))
        min_distance = max(1, int(fs * 0.34))  # ~176 bpm ceiling on peak spacing
        if std > 1e-8:
            peaks, _ = find_peaks(ppg, distance=min_distance, prominence=std * 0.3)
        else:
            peaks = np.array([], dtype=int)

        rmssd_available = False
        hr_peaks = None
        if len(peaks) > 1:
            ibi = np.diff(peaks) / fs
            features["ibi_mean"] = float(np.mean(ibi))
            features["ibi_std"] = float(np.std(ibi))
            features["rmssd"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) > 1 else 0.05
            rmssd_available = len(ibi) > 1
            mean_ibi = float(np.mean(ibi))
            hr_peaks = 60.0 / mean_ibi if mean_ibi > 1e-8 else None
        else:
            # Honest defaults: keep callers working but DO NOT pretend confidence.
            features["ibi_mean"] = 0.8
            features["ibi_std"] = 0.0
            features["rmssd"] = 0.05

        # Decide the reported heart rate and whether the two estimators agree.
        agreement = True
        if hr_ok:
            hr = hr_psd
            if hr_peaks is not None and abs(hr_peaks - hr_psd) > 12.0:
                agreement = False
        elif hr_peaks is not None and 40.0 <= hr_peaks <= 200.0:
            hr = hr_peaks  # PSD unsure but time-domain plausible
        else:
            hr = 75.0
            hr_ok = False

        snr_db = signal_snr(ppg, fs)
        snr_quality = float(np.clip((snr_db + 3.0) / 12.0, 0.0, 1.0))  # -3 dB->0, 9 dB->1
        signal_quality = float(np.clip(0.5 * float(quality) + 0.5 * snr_quality, 0.0, 1.0))
        if not agreement:
            signal_quality *= 0.7

        low_quality = (not hr_ok) or (snr_db < 1.0) or (len(peaks) < 2) or (not agreement)

        features["heart_rate"] = float(hr)
        features["hr_ok"] = bool(hr_ok)
        features["hr_agreement"] = bool(agreement)
        features["snr_db"] = float(snr_db)
        features["signal_quality"] = signal_quality
        features["low_quality"] = bool(low_quality)
        features["rmssd_available"] = bool(rmssd_available)

        # Backward-compat placeholders (see _NEUTRAL_LF_HF note above).
        features["lf_hf_ratio"] = _NEUTRAL_LF_HF
        features["hrv_available"] = False

        return features
