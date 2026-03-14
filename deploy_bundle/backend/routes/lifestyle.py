import numpy as np
from flask import Blueprint, current_app, jsonify, request
from scipy.stats import norm
from datetime import datetime

from utils.security import rate_limit, require_api_key

lifestyle_bp = Blueprint("lifestyle", __name__, url_prefix="/api/lifestyle")

BP_MEAN, BP_STD = 120, 15
GL_MEAN, GL_STD = 95, 12


def _to_percentile(value: float, mean: float, std: float):
    return float(np.clip(norm.cdf((value - mean) / std) * 100, 1, 99))


def _range_label(percentile: float):
    if percentile < 30:
        return "낮음"
    if percentile > 70:
        return "높음"
    return "일반적인 범위"


@lifestyle_bp.route("/comparison", methods=["GET"])
@require_api_key
@rate_limit
def get_comparison():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")

    latest = current_app.db.get_latest_measurement(user_id)
    if not latest:
        return jsonify({"success": True, "data": None, "message": "No measurement data"})

    bp_percentile = _to_percentile(float(latest["bp_systolic"]), BP_MEAN, BP_STD)
    gl_percentile = _to_percentile(float(latest["blood_sugar"]), GL_MEAN, GL_STD)

    return jsonify(
        {
            "success": True,
            "data": {
                "bp_percentile": round(bp_percentile, 1),
                "glucose_percentile": round(gl_percentile, 1),
                "bp_range": _range_label(bp_percentile),
                "glucose_range": _range_label(gl_percentile),
            },
        }
    )


@lifestyle_bp.route("/recommendations", methods=["GET"])
@require_api_key
@rate_limit
def get_recommendations():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")

    recent = current_app.db.get_recent_measurements(user_id, days=7)
    if not recent:
        return jsonify({"success": True, "data": []})

    bp = np.array([r["bp_systolic"] for r in recent], dtype=np.float64)
    gl = np.array([r["blood_sugar"] for r in recent], dtype=np.float64)

    bp_cv = np.std(bp) / np.mean(bp) * 100 if np.mean(bp) else 0
    gl_cv = np.std(gl) / np.mean(gl) * 100 if np.mean(gl) else 0

    recommendations = [
        {
            "id": "sleep",
            "title": "수면 시간 1시간 늘리기",
            "icon": "🌙",
            "description": "수면 습관은 혈당·혈압 관리에 영향을 줄 수 있는 생활 요소 중 하나로 알려져 있어요.",
            "detail": "취침/기상 시간을 일정하게 맞추면 일중 변동 관리에 참고가 될 수 있어요.",
            "reason_context": f"최근 7일 변동성(BP {bp_cv:.1f}%, 혈당 {gl_cv:.1f}%)을 기준으로 수면 루틴을 우선 추천해요.",
            "priority": 1 if bp_cv > 12 or gl_cv > 12 else 2,
        },
        {
            "id": "salt",
            "title": "염분 섭취 줄이기",
            "icon": "🧂",
            "description": "염분 섭취는 혈압 관리와 연관된 생활 요인으로 알려져 있어요.",
            "detail": "국물류 섭취 빈도를 조금 줄이면 혈압 변동 완화에 도움이 될 수 있어요.",
            "reason_context": f"최근 평균 수축기 혈압이 {np.mean(bp):.1f}로 나타나 염분 관리 실천을 함께 권장해요.",
            "priority": 1 if np.mean(bp) > 125 else 2,
        },
        {
            "id": "water",
            "title": "하루 하나 더 수분 마시기",
            "icon": "💧",
            "description": "수분 섭취는 일상적인 혈압 관리에 참고되는 요소 중 하나예요.",
            "detail": "하루 1회 물 섭취 알림을 설정해 루틴을 만드는 방법을 추천해요.",
            "reason_context": f"최근 혈당 평균 {np.mean(gl):.1f} 흐름을 참고해 수분 루틴을 함께 관리해보세요.",
            "priority": 2 if np.mean(gl) > 100 else 3,
        },
    ]

    recommendations = sorted(recommendations, key=lambda x: x["priority"])
    return jsonify({"success": True, "data": recommendations[:3]})


@lifestyle_bp.route("/habits/checkin", methods=["POST"])
@require_api_key
@rate_limit
def save_habit_checkin():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")
    payload = request.get_json(silent=True) or {}

    habit_id = payload.get("habit_id")
    completed = bool(payload.get("completed", False))
    check_date = payload.get("date") or datetime.now().strftime("%Y-%m-%d")

    if habit_id not in {"sleep", "water", "salt"}:
        return jsonify({"success": False, "error": "Invalid habit_id"}), 400

    current_app.db.save_habit_checkin(user_id, habit_id, check_date, completed)
    return jsonify({"success": True, "data": {"habit_id": habit_id, "completed": completed, "date": check_date}})


@lifestyle_bp.route("/habits/progress", methods=["GET"])
@require_api_key
@rate_limit
def get_habit_progress():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")
    days = int(request.args.get("days", 7))

    progress = current_app.db.get_habit_progress(user_id, days=days)
    today = datetime.now().strftime("%Y-%m-%d")
    checkins = current_app.db.get_habit_checkins(user_id, days=1)
    today_checks = {c["habit_id"]: bool(c["completed"]) for c in checkins if c.get("check_date") == today}

    return jsonify(
        {
            "success": True,
            "data": {
                "today_completion_rate": round(sum(1 for v in today_checks.values() if v) / 3 * 100, 1) if today_checks else 0.0,
                "today_checks": {
                    "sleep": today_checks.get("sleep", False),
                    "water": today_checks.get("water", False),
                    "salt": today_checks.get("salt", False),
                },
                "progress": progress,
            },
        }
    )


@lifestyle_bp.route("/before-after", methods=["GET"])
@require_api_key
@rate_limit
def get_before_after():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")
    days = int(request.args.get("days", 14))
    result = current_app.db.get_before_after_stats(user_id, days=days)
    return jsonify({"success": True, "data": result})


@lifestyle_bp.route("/notifications", methods=["GET"])
@require_api_key
@rate_limit
def get_notifications():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")
    settings = current_app.db.get_notification_settings(user_id)

    if not settings:
        defaults = [
            {"setting_type": "measurement", "setting_time": "09:00", "weekdays": ["Mon", "Tue", "Wed", "Thu", "Fri"], "enabled": True},
            {"setting_type": "water", "setting_time": "14:00", "weekdays": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "enabled": True},
            {"setting_type": "sleep", "setting_time": "23:00", "weekdays": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "enabled": False},
        ]
        for item in defaults:
            current_app.db.save_notification_setting(
                user_id,
                item["setting_type"],
                item["setting_time"],
                item["weekdays"],
                item["enabled"],
            )
        settings = current_app.db.get_notification_settings(user_id)

    return jsonify({"success": True, "data": settings})


@lifestyle_bp.route("/notifications", methods=["POST"])
@require_api_key
@rate_limit
def save_notifications():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")
    payload = request.get_json(silent=True) or {}

    setting_type = payload.get("setting_type")
    setting_time = payload.get("setting_time", "09:00")
    weekdays = payload.get("weekdays", [])
    enabled = bool(payload.get("enabled", True))

    if setting_type not in {"measurement", "water", "sleep"}:
        return jsonify({"success": False, "error": "Invalid setting_type"}), 400

    current_app.db.save_notification_setting(user_id, setting_type, setting_time, weekdays, enabled)
    return jsonify({"success": True, "data": {"setting_type": setting_type, "enabled": enabled}})
