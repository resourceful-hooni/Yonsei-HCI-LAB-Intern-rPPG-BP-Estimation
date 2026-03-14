from __future__ import annotations

import numpy as np


def _calc_cv(values: list[float]):
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if abs(mean) < 1e-8:
        return 0.0
    return float(np.std(arr) / mean * 100.0)


def classify_status(bp_values: list[float], glucose_values: list[float]):
    bp_cv = _calc_cv(bp_values)
    gl_cv = _calc_cv(glucose_values)
    total_cv = max(bp_cv, gl_cv)

    bp_mean = float(np.mean(bp_values)) if bp_values else 120.0
    gl_mean = float(np.mean(glucose_values)) if glucose_values else 95.0

    in_bp_range = 90 <= bp_mean <= 140
    in_gl_range = 70 <= gl_mean <= 110
    in_range = in_bp_range and in_gl_range

    if total_cv < 10 and in_range:
        level = 1
        label = "안정적"
    elif total_cv < 15:
        level = 2
        label = "관심 필요"
    elif total_cv < 20:
        level = 3
        label = "집중 관리"
    else:
        level = 4
        label = "관리 강화"

    return {
        "status_level": level,
        "status_label": label,
        "bp_cv": round(bp_cv, 2),
        "glucose_cv": round(gl_cv, 2),
    }
