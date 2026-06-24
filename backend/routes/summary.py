from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from utils import i18n
from utils.security import rate_limit, require_api_key
from utils.status_classifier import classify_status
from utils.text_generator import generate_summary

summary_bp = Blueprint("summary", __name__, url_prefix="/api/summary")


def _date_key(dt_str: str):
    return dt_str.split(" ")[0] if dt_str else datetime.now().strftime("%Y-%m-%d")


def _time_label(dt_str: str):
    if not dt_str:
        return datetime.now().strftime("%m-%d %H:%M")
    try:
        iso = str(dt_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        pass
    candidates = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.strftime("%m-%d %H:%M")
        except Exception:
            continue
    return dt_str


def _code_from(ok_good: bool, ok_fair: bool) -> str:
    return "good" if ok_good else "fair" if ok_fair else "poor"


def _quality_checklist(quality: Optional[dict], confidence: float, bp_cv: float, gl_cv: float, lang: str):
    if quality:
        items = []
        for qid, score_key in (("lighting", "lighting_score"), ("movement", "movement_score"), ("alignment", "alignment_score")):
            score = quality.get(score_key)
            items.append({
                "id": qid,
                "label": i18n.quality_label(qid, lang),
                "status": i18n.quality_status_text(i18n.quality_status_code(score), lang),
                "score": round(float(score or 0.0), 3),
                "tip": i18n.quality_tip(qid, lang, measured=True),
            })
        return items

    mcv = max(bp_cv, gl_cv)
    codes = {
        "lighting": _code_from(confidence >= 0.8, confidence >= 0.65),
        "movement": _code_from(mcv < 12, mcv < 18),
        "alignment": _code_from(confidence >= 0.75, confidence >= 0.6),
    }
    return [
        {
            "id": qid,
            "label": i18n.quality_label(qid, lang),
            "status": i18n.quality_status_text(codes[qid], lang),
            "tip": i18n.quality_tip(qid, lang, measured=False),
        }
        for qid in ("lighting", "movement", "alignment")
    ]


def _daily_comment(bp_cv: float, gl_cv: float, bp_delta: float, gl_delta: float, lang: str):
    total_cv = max(bp_cv, gl_cv)
    if total_cv < 10 and abs(bp_delta) < 5 and abs(gl_delta) < 5:
        variant = "stable"
    elif total_cv < 16:
        variant = "small"
    else:
        variant = "large"
    return i18n.daily_comment(variant, lang)


@summary_bp.route("/daily", methods=["GET"])
@require_api_key
@rate_limit
def get_daily_summary():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")
    lang = i18n.normalize_lang(request.args.get("lang"))

    latest = current_app.db.get_latest_measurement(user_id)
    recent = current_app.db.get_recent_measurements(user_id, days=7)

    if not latest:
        return jsonify({"success": True, "data": None, "message": "No measurement data"})

    bp_values = [float(r["bp_systolic"]) for r in recent]
    glucose_values = [float(r["blood_sugar"]) for r in recent]

    status = classify_status(bp_values, glucose_values, lang)
    summary_text = generate_summary(status, lang)

    avg_sbp = float(np.mean(bp_values)) if bp_values else float(latest["bp_systolic"])
    avg_gl = float(np.mean(glucose_values)) if glucose_values else float(latest["blood_sugar"])
    current_conf = float(latest.get("confidence") or 0.0)

    delta_sbp = float(latest["bp_systolic"]) - avg_sbp
    delta_gl = float(latest["blood_sugar"]) - avg_gl

    confidence_trend = [
        {
            "date": _date_key(r["measurement_time"]),
            "value": round(float(r.get("confidence") or 0.0), 3),
        }
        for r in recent[-7:]
    ]

    bp_cv = status.get("bp_cv", 0.0)
    gl_cv = status.get("glucose_cv", 0.0)
    quality = current_app.db.get_measurement_quality(int(latest["measurement_id"])) if latest.get("measurement_id") else None

    return jsonify(
        {
            "success": True,
            "data": {
                "status": status,
                "summary_text": summary_text,
                "current_values": {
                    "bp_systolic": latest["bp_systolic"],
                    "bp_diastolic": latest["bp_diastolic"],
                    "blood_sugar": latest["blood_sugar"],
                    "confidence": latest.get("confidence"),
                },
                "labels": i18n.vital_labels(lang),
                "delta_badges": {
                    "bp_systolic": round(delta_sbp, 1),
                    "blood_sugar": round(delta_gl, 1),
                    "bp_avg_7d": round(avg_sbp, 1),
                    "glucose_avg_7d": round(avg_gl, 1),
                },
                "confidence_trend": confidence_trend,
                "quality_checklist": _quality_checklist(quality, current_conf, bp_cv, gl_cv, lang),
                "quality_meta": {
                    "type": "measured_vision_metrics" if quality else "rule_based_estimation",
                    "is_measured_directly": bool(quality),
                    "basis": {
                        "confidence": round(current_conf, 3),
                        "bp_cv": round(float(bp_cv or 0.0), 2),
                        "glucose_cv": round(float(gl_cv or 0.0), 2),
                        "lighting_score": None if not quality else quality.get("lighting_score"),
                        "movement_score": None if not quality else quality.get("movement_score"),
                        "alignment_score": None if not quality else quality.get("alignment_score"),
                        "face_detected_ratio": None if not quality else quality.get("face_detected_ratio"),
                        "method": None if not quality else quality.get("method"),
                    },
                },
                "daily_comment": _daily_comment(bp_cv, gl_cv, delta_sbp, delta_gl, lang),
                "reguide": i18n.reguide(lang),
            },
        }
    )


@summary_bp.route("/trends", methods=["GET"])
@require_api_key
@rate_limit
def get_trends():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")
    days = int(request.args.get("days", 7))

    recent = current_app.db.get_recent_measurements(user_id, days=days)
    if not recent:
        return jsonify({"success": True, "data": {"bp_trend": [], "glucose_trend": []}})

    bp = [float(r["bp_systolic"]) for r in recent]
    gl = [float(r["blood_sugar"]) for r in recent]

    bp_cv = float(np.std(bp) / np.mean(bp) * 100) if np.mean(bp) else 0.0
    gl_cv = float(np.std(gl) / np.mean(gl) * 100) if np.mean(gl) else 0.0

    bp_change = ((bp[-1] - bp[0]) / bp[0] * 100) if len(bp) > 1 and bp[0] else 0.0
    gl_change = ((gl[-1] - gl[0]) / gl[0] * 100) if len(gl) > 1 and gl[0] else 0.0

    bp_trend = [
        {
            "date": _time_label(r["measurement_time"]),
            "timestamp": r["measurement_time"],
            "value": r["bp_systolic"],
        }
        for r in recent
    ]
    glucose_trend = [
        {
            "date": _time_label(r["measurement_time"]),
            "timestamp": r["measurement_time"],
            "value": r["blood_sugar"],
        }
        for r in recent
    ]
    combined_trend = [
        {
            "timestamp": r["measurement_time"],
            "label": _time_label(r["measurement_time"]),
            "bp": r["bp_systolic"],
            "glucose": r["blood_sugar"],
        }
        for r in recent
    ]

    return jsonify(
        {
            "success": True,
            "data": {
                "bp_trend": bp_trend,
                "glucose_trend": glucose_trend,
                "combined_trend": combined_trend,
                "variability": {
                    "bp_cv": round(bp_cv, 2),
                    "glucose_cv": round(gl_cv, 2),
                },
                "trend_percentages": {
                    "bp": round(bp_change, 2),
                    "glucose": round(gl_change, 2),
                },
            },
        }
    )
