import os
import re
import importlib.util
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, resample


class ResearchBPModelBridge:
    """Bridge to load and run existing Yonsei research BP model (.h5)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_info_path: Optional[str] = None,
        target_len: int = 875,
    ):
        self.target_len = int(target_len)
        self.model = None
        self.model_path = None

        self.signal_mean = None
        self.signal_scale = None
        self.label_mean = None
        self.label_scale = None

        self._load_model(model_path)
        self._load_scaler_stats(scaler_info_path)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _candidate_model_paths(self, explicit_path: Optional[str]):
        if explicit_path:
            return [explicit_path]

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        lab_root = os.path.join(base_dir, "Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation")

        return [
            os.path.join(lab_root, "models", "resnet_rppg_adapted.h5"),
            os.path.join(lab_root, "models", "resnet_ppg_nonmixed.h5"),
            os.path.join(lab_root, "models", "transformer_bp_model.h5"),
            os.path.join(lab_root, "results", "ms_tcn_attention_best_weights.h5"),
            os.path.join(lab_root, "models", "multi_task_bp_model.h5"),
        ]

    def _load_model(self, explicit_model_path: Optional[str]):
        try:
            import tensorflow as tf
        except Exception:
            return

        def _load_ms_tcn_module(model_file: str):
            try:
                p = Path(model_file)
                if "ms_tcn" not in p.name.lower():
                    return None

                root = p
                while root.parent != root:
                    if root.name == "Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation":
                        break
                    root = root.parent

                module_path = root / "models" / "ms_tcn_attention_model.py"
                if not module_path.exists():
                    return None

                spec = importlib.util.spec_from_file_location("ms_tcn_attention_model", str(module_path))
                if spec is None or spec.loader is None:
                    return None

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                return module
            except Exception:
                return None

        def _load_ms_tcn_custom_objects(model_file: str):
            try:
                module = _load_ms_tcn_module(model_file)
                if module is None:
                    return None

                custom_objects = {}
                for name in ["SqueezeExcitation1D", "TCNBlock", "LinearAttention", "AttentionBlock"]:
                    if hasattr(module, name):
                        custom_objects[name] = getattr(module, name)
                return custom_objects if custom_objects else None
            except Exception:
                return None

        for candidate in self._candidate_model_paths(explicit_model_path):
            if not candidate:
                continue
            if not os.path.exists(candidate):
                continue

            try:
                custom_objects = _load_ms_tcn_custom_objects(candidate)
                if custom_objects is not None:
                    self.model = tf.keras.models.load_model(
                        candidate,
                        custom_objects=custom_objects,
                        compile=False,
                    )
                else:
                    self.model = tf.keras.models.load_model(candidate, compile=False)
                self.model_path = os.path.abspath(candidate)
                return
            except Exception:
                pass

            # Fallback for weights-only MS-TCN checkpoint (.h5 weights)
            try:
                p = Path(candidate)
                if "ms_tcn" in p.name.lower() and "weights" in p.name.lower():
                    module = _load_ms_tcn_module(candidate)
                    if module is not None and hasattr(module, "create_ms_tcn_attention_model"):
                        model = module.create_ms_tcn_attention_model(input_shape=(self.target_len, 1))
                        model.load_weights(candidate)
                        self.model = model
                        self.model_path = os.path.abspath(candidate)
                        return
            except Exception:
                pass

            continue

    def _load_scaler_stats(self, explicit_info_path: Optional[str]):
        candidate_paths = []
        if explicit_info_path:
            candidate_paths.append(explicit_info_path)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        candidate_paths.extend(
            [
                os.path.join(base_dir, "Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation", "data", "rppg_info.txt"),
                os.path.join(base_dir, "Yonsei-HCI-LAB-Intern-rPPG-BP-Estimation", "data", "scaler_info.txt"),
            ]
        )

        for info_path in candidate_paths:
            if not info_path or not os.path.exists(info_path):
                continue
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    text = f.read()

                signal_mean_match = re.search(r"Signal Statistics:\s*\n\s*Mean:\s*\[([^\]]+)\]", text, re.S)
                signal_scale_match = re.search(r"Signal Statistics:.*?Scale:\s*\[([^\]]+)\]", text, re.S)
                label_mean_match = re.search(r"Label Statistics:\s*\n\s*Mean:\s*\[([^\]]+)\]", text, re.S)
                label_scale_match = re.search(r"Label Statistics:.*?Scale:\s*\[([^\]]+)\]", text, re.S)

                if signal_mean_match and signal_scale_match:
                    self.signal_mean = np.fromstring(signal_mean_match.group(1), sep=" ")
                    self.signal_scale = np.fromstring(signal_scale_match.group(1), sep=" ")
                if label_mean_match and label_scale_match:
                    self.label_mean = np.fromstring(label_mean_match.group(1), sep=" ")
                    self.label_scale = np.fromstring(label_scale_match.group(1), sep=" ")
                return
            except Exception:
                continue

    def _proper_resample(self, signal: np.ndarray, fs: float) -> np.ndarray:
        if len(signal) <= 1:
            return signal.astype(np.float32)

        if self.target_len < len(signal):
            nyq = 0.5 * fs
            cutoff = (0.5 * self.target_len / len(signal)) / nyq if nyq > 0 else 0.3
            cutoff = float(np.clip(cutoff, 0.001, 0.999))
            try:
                b, a = butter(4, cutoff, btype="low")
                signal = filtfilt(b, a, signal)
            except Exception:
                pass

        return resample(signal, self.target_len).astype(np.float32)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        if self.signal_mean is not None and self.signal_scale is not None:
            mean = np.asarray(self.signal_mean, dtype=np.float32)
            scale = np.asarray(self.signal_scale, dtype=np.float32)
            if mean.size == 1:
                return (signal - float(mean[0])) / (float(scale[0]) + 1e-8)
            return (signal - mean) / (scale + 1e-8)

        mean = float(np.mean(signal))
        std = float(np.std(signal))
        if std < 1e-8:
            return signal - mean
        return (signal - mean) / std

    def _format_input(self, signal: np.ndarray) -> np.ndarray:
        input_shape = getattr(self.model, "input_shape", None)

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if input_shape is None:
            return signal.reshape(1, -1, 1)

        if len(input_shape) == 3:
            return signal.reshape(1, -1, 1)
        if len(input_shape) == 2:
            return signal.reshape(1, -1)
        return signal.reshape(1, -1, 1)

    def predict(self, ppg_signal: np.ndarray, frame_rate: float, quality: float = 0.75) -> Optional[Tuple[float, float, float]]:
        if not self.is_ready:
            return None

        signal = np.asarray(ppg_signal, dtype=np.float32).reshape(-1)
        signal = self._proper_resample(signal, frame_rate)
        signal = self._normalize(signal)

        input_data = self._format_input(signal)
        prediction = self.model.predict(input_data, verbose=0)

        if isinstance(prediction, list):
            sbp_model = float(prediction[0][0, 0])
            dbp_model = float(prediction[1][0, 0])
        else:
            pred = np.asarray(prediction)
            if pred.ndim == 2 and pred.shape[1] >= 2:
                sbp_model = float(pred[0, 0])
                dbp_model = float(pred[0, 1])
            else:
                return None

        if self.label_mean is not None and self.label_scale is not None and self.label_mean.size >= 2 and self.label_scale.size >= 2:
            sbp = sbp_model * float(self.label_scale[0]) + float(self.label_mean[0])
            dbp = dbp_model * float(self.label_scale[1]) + float(self.label_mean[1])
        else:
            sbp, dbp = sbp_model, dbp_model

        sbp = float(np.clip(sbp, 90, 180))
        dbp = float(np.clip(dbp, 60, 120))
        if dbp >= sbp:
            dbp = max(60.0, sbp - 20.0)

        confidence = float(np.clip(0.6 + quality * 0.35, 0.0, 1.0))
        return sbp, dbp, confidence
