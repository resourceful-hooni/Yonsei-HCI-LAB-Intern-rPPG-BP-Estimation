from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks, welch

from utils.signal_processing import bandpass_filter, normalize_signal


FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]


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

    def extract_signal(self, frames: list[np.ndarray], frame_rate: float):
        rgb_series = []

        for frame in frames:
            rgb_mean = self._extract_forehead_rgb(frame)
            if rgb_mean is not None:
                rgb_series.append(rgb_mean)

        if len(rgb_series) < max(30, int(frame_rate * 3)):
            raise ValueError("유효한 얼굴 ROI 프레임이 충분하지 않습니다.")

        rgb = np.asarray(rgb_series, dtype=np.float64)
        r = normalize_signal(rgb[:, 0])
        g = normalize_signal(rgb[:, 1])
        b = normalize_signal(rgb[:, 2])

        x_comp = 3.0 * r - 2.0 * g
        y_comp = 1.5 * r + g - 1.5 * b

        y_std = np.std(y_comp)
        alpha = np.std(x_comp) / y_std if y_std > 1e-8 else 0.0
        ppg = x_comp - alpha * y_comp

        ppg_filtered = bandpass_filter(ppg, fs=frame_rate, low=0.7, high=4.0, order=3)
        ppg_filtered = normalize_signal(ppg_filtered)

        quality = float(np.clip(len(rgb_series) / max(len(frames), 1), 0.0, 1.0))
        return ppg_filtered, quality

    def extract_features(self, ppg: np.ndarray, frame_rate: float, quality: float = 0.7):
        ppg = np.asarray(ppg, dtype=np.float64)

        features = {
            "mean": float(np.mean(ppg)),
            "std": float(np.std(ppg)),
            "skewness": float(skew(ppg)) if len(ppg) > 3 else 0.0,
            "kurtosis": float(kurtosis(ppg)) if len(ppg) > 3 else 0.0,
            "quality": float(quality),
        }

        min_distance = max(1, int(frame_rate * 0.4))
        peaks, _ = find_peaks(ppg, distance=min_distance, prominence=np.std(ppg) * 0.2)

        if len(peaks) > 1:
            ibi = np.diff(peaks) / frame_rate
            features["ibi_mean"] = float(np.mean(ibi))
            features["ibi_std"] = float(np.std(ibi))
            features["rmssd"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) > 1 else 0.0
            features["heart_rate"] = float(60.0 / np.mean(ibi)) if np.mean(ibi) > 1e-8 else 75.0
        else:
            features["ibi_mean"] = 0.8
            features["ibi_std"] = 0.0
            features["rmssd"] = 0.05
            features["heart_rate"] = 75.0

        freqs, psd = welch(ppg, fs=frame_rate, nperseg=min(len(ppg), 256))
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)

        lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
        hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0

        features["lf_power"] = lf_power
        features["hf_power"] = hf_power
        features["lf_hf_ratio"] = float(lf_power / hf_power) if hf_power > 1e-8 else 1.2

        return features
