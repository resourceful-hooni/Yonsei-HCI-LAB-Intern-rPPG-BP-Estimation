from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from flask import Blueprint, current_app, jsonify, request

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


def _quality_status(score: Optional[float]):
    if score is None:
        return "보통"
    if score >= 0.75:
        return "좋음"
    if score >= 0.55:
        return "보통"
    return "개선 필요"


def _quality_checklist(quality: Optional[dict], confidence: float, bp_cv: float, gl_cv: float):
    if quality:
        return [
            {
                "id": "lighting",
                "label": "조명",
                "status": _quality_status(quality.get("lighting_score")),
                "score": round(float(quality.get("lighting_score") or 0.0), 3),
                "tip": "얼굴 좌우 밝기 차이를 줄이고 그림자가 생기지 않도록 조명을 맞춰주세요.",
            },
            {
                "id": "movement",
                "label": "움직임",
                "status": _quality_status(quality.get("movement_score")),
                "score": round(float(quality.get("movement_score") or 0.0), 3),
                "tip": "측정 중 고개·시선 이동을 줄이고 자연스럽게 정면을 유지해주세요.",
            },
            {
                "id": "alignment",
                "label": "얼굴정렬",
                "status": _quality_status(quality.get("alignment_score")),
                "score": round(float(quality.get("alignment_score") or 0.0), 3),
                "tip": "얼굴 중심이 프레임 중앙에 오도록 위치를 맞추고 턱선이 잘 보이게 해주세요.",
            },
        ]

    lighting = "좋음" if confidence >= 0.8 else "보통" if confidence >= 0.65 else "개선 필요"
    movement = "좋음" if max(bp_cv, gl_cv) < 12 else "보통" if max(bp_cv, gl_cv) < 18 else "개선 필요"
    alignment = "좋음" if confidence >= 0.75 else "보통" if confidence >= 0.6 else "개선 필요"
    return [
        {"id": "lighting", "label": "조명", "status": lighting, "tip": "얼굴 정면에 균일한 빛을 유지해주세요."},
        {"id": "movement", "label": "움직임", "status": movement, "tip": "측정 중 고개/상체 움직임을 줄여주세요."},
        {"id": "alignment", "label": "얼굴정렬", "status": alignment, "tip": "얼굴을 화면 중앙에 유지해주세요."},
    ]


def _daily_comment(bp_cv: float, gl_cv: float, bp_delta: float, gl_delta: float):
    total_cv = max(bp_cv, gl_cv)
    if total_cv < 10 and abs(bp_delta) < 5 and abs(gl_delta) < 5:
        return "오늘은 최근 흐름 대비 변동 폭이 비교적 안정적으로 보여요."
    if total_cv < 16:
        return "최근 기록에서 작은 변동이 보여요. 측정 환경을 일정하게 유지해보세요."
    return "최근 변동 폭이 다소 큰 편이에요. 조명과 자세를 일정하게 맞춰 측정해보세요."


@summary_bp.route("/daily", methods=["GET"])
@require_api_key
@rate_limit
def get_daily_summary():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")

    latest = current_app.db.get_latest_measurement(user_id)
    recent = current_app.db.get_recent_measurements(user_id, days=7)

    if not latest:
        return jsonify({"success": True, "data": None, "message": "No measurement data"})

    bp_values = [float(r["bp_systolic"]) for r in recent]
    glucose_values = [float(r["blood_sugar"]) for r in recent]

    status = classify_status(bp_values, glucose_values)
    summary_text = generate_summary(status)

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
                "labels": {
                    "bp": "혈압",
                    "glucose": "혈당",
                },
                "delta_badges": {
                    "bp_systolic": round(delta_sbp, 1),
                    "blood_sugar": round(delta_gl, 1),
                    "bp_avg_7d": round(avg_sbp, 1),
                    "glucose_avg_7d": round(avg_gl, 1),
                },
                "confidence_trend": confidence_trend,
                "quality_checklist": _quality_checklist(quality, current_conf, bp_cv, gl_cv),
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
                "daily_comment": _daily_comment(bp_cv, gl_cv, delta_sbp, delta_gl),
                "reguide": {
                    "title": "다시 측정 가이드",
                    "description": "조명/자세/정렬 팁을 확인하고 다시 측정해보세요.",
                    "path": "/measurement",
                },
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
