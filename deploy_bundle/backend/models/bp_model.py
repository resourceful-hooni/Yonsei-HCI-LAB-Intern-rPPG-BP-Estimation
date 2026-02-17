from __future__ import annotations

import numpy as np

from models.research_bp_model import ResearchBPModelBridge


class BloodPressureEstimator:
    def __init__(self, model_path: str | None = None, scaler_info_path: str | None = None, target_len: int = 875):
        self.research_model = ResearchBPModelBridge(
            model_path=model_path,
            scaler_info_path=scaler_info_path,
            target_len=target_len,
        )

    def estimate(self, features: dict, ppg_signal=None, frame_rate: float = 30.0):
        quality = float(features.get("quality", 0.7))

        if ppg_signal is not None and self.research_model.is_ready:
            try:
                pred = self.research_model.predict(ppg_signal, frame_rate=frame_rate, quality=quality)
                if pred is not None:
                    systolic, diastolic, conf = pred
                    return {
                        "systolic": int(round(systolic)),
                        "diastolic": int(round(diastolic)),
                        "confidence": float(conf),
                        "source": "research_model",
                    }
            except Exception:
                pass

        hr = float(features.get("heart_rate", 75.0))
        rmssd = float(features.get("rmssd", 0.05))
        lf_hf = float(features.get("lf_hf_ratio", 1.2))

        lf_hf_adjustment = (lf_hf - 1.2) * 3.0
        systolic = 120 + (hr - 80) * 0.5 - rmssd * 80 + lf_hf_adjustment
        diastolic = systolic * 0.65 + (hr - 75) * 0.05

        systolic = float(np.clip(systolic, 90, 180))
        diastolic = float(np.clip(diastolic, 60, 120))

        if diastolic >= systolic:
            diastolic = max(60.0, systolic - 20.0)

        conf = float(np.clip(0.55 + quality * 0.4 - abs(hr - 75) / 150.0, 0.0, 1.0))

        return {
            "systolic": int(round(systolic)),
            "diastolic": int(round(diastolic)),
            "confidence": conf,
            "source": "empirical_fallback",
        }
