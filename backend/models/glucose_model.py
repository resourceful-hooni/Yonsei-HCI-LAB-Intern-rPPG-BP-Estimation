import numpy as np


class BloodGlucoseEstimator:
    def __init__(self, use_dummy: bool = True):
        self.use_dummy = use_dummy

    def estimate(self, features: dict):
        if self.use_dummy:
            hr = float(features.get("heart_rate", 75.0))
            quality = float(features.get("quality", 0.7))
            glucose = 95.0 + (hr - 75.0) * 0.15
            glucose = float(np.clip(glucose, 80, 120))
            conf = float(np.clip(0.45 + quality * 0.3, 0.0, 0.95))
            return {
                "glucose": int(round(glucose)),
                "confidence": conf,
                "source": "dummy",
            }

        lf_hf = float(features.get("lf_hf_ratio", 1.2))
        rmssd = float(features.get("rmssd", 0.05))
        hr = float(features.get("heart_rate", 75.0))

        glucose = 95.0
        glucose += 10 if lf_hf > 2.0 else -5 if lf_hf < 0.8 else 0
        glucose += 8 if rmssd < 0.03 else -5 if rmssd > 0.08 else 0
        glucose += 5 if hr > 90 else -3 if hr < 65 else 0
        glucose = float(np.clip(glucose, 60, 200))

        quality = float(features.get("quality", 0.7))
        conf = float(np.clip(0.5 + quality * 0.45 - abs(lf_hf - 1.2) / 5.0, 0.0, 1.0))

        return {
            "glucose": int(round(glucose)),
            "confidence": conf,
            "source": "empirical",
        }
