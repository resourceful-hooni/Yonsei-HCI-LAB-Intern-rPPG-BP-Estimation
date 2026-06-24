from __future__ import annotations

import numpy as np

from models.research_bp_model import ResearchBPModelBridge


class BloodPressureEstimator:
    def __init__(self, model_path: str | None = None, scaler_info_path: str | None = None,
                 target_len: int = 875, model_fs: float = 0.0):
        self.research_model = ResearchBPModelBridge(
            model_path=model_path,
            scaler_info_path=scaler_info_path,
            target_len=target_len,
            model_fs=model_fs,
        )

    def estimate(self, features: dict, ppg_signal=None, frame_rate: float = 30.0):
        # Prefer a signal-quality score (frame coverage + SNR) over the raw
        # frame-detection ratio so confidence reflects the actual pulse.
        signal_quality = float(features.get("signal_quality", features.get("quality", 0.7)))
        low_quality = bool(features.get("low_quality", False))
        hr_ok = bool(features.get("hr_ok", True))

        if ppg_signal is not None and self.research_model.is_ready:
            try:
                pred = self.research_model.predict(ppg_signal, frame_rate=frame_rate, quality=signal_quality)
                if pred is not None:
                    systolic, diastolic, conf = pred
                    if low_quality:
                        conf = min(conf, 0.5)
                    return {
                        "systolic": int(round(systolic)),
                        "diastolic": int(round(diastolic)),
                        "confidence": float(conf),
                        "source": "research_model",
                        "low_quality": low_quality,
                        "signal_quality": round(signal_quality, 3),
                    }
            except Exception:
                pass

        hr = float(features.get("heart_rate", 75.0))
        rmssd = float(features.get("rmssd", 0.05))

        # Empirical fallback. NOTE: this is a weakly-grounded heuristic, not a
        # validated BP model. The previous (lf_hf - 1.2) * 3 term was dead — the
        # band-passed pulse carries no power in the HRV bands — so it is removed.
        systolic = 120 + (hr - 80) * 0.5 - rmssd * 80
        diastolic = systolic * 0.65 + (hr - 75) * 0.05

        systolic = float(np.clip(systolic, 90, 180))
        diastolic = float(np.clip(diastolic, 60, 120))
        if diastolic >= systolic:
            diastolic = max(60.0, systolic - 20.0)

        conf = 0.5 + signal_quality * 0.4 - abs(hr - 75) / 200.0
        if not hr_ok:
            conf -= 0.2
        conf = float(np.clip(conf, 0.0, 1.0))
        if low_quality:
            conf = min(conf, 0.5)

        return {
            "systolic": int(round(systolic)),
            "diastolic": int(round(diastolic)),
            "confidence": conf,
            "source": "empirical_fallback",
            "low_quality": low_quality,
            "signal_quality": round(signal_quality, 3),
        }
