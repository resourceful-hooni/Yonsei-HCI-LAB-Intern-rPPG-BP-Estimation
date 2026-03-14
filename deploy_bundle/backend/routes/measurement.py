import base64
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from models.bp_model import BloodPressureEstimator
from models.glucose_model import BloodGlucoseEstimator
from models.rppg_estimator import RPPGEstimator
from utils.security import rate_limit, require_api_key, validate_request_size

measurement_bp = Blueprint("measurement", __name__, url_prefix="/api/measurement")

SESSIONS = {}
rppg_estimator = RPPGEstimator()
bp_estimator = None
glucose_estimator = None


def _clamp01(value):
    if value is None:
        return None
    try:
        return float(max(0.0, min(1.0, float(value))))
    except Exception:
        return None


def _normalize_quality_metrics(metrics: Optional[dict]):
    metrics = metrics or {}
    return {
        "lighting_score": _clamp01(metrics.get("lighting_score")),
        "movement_score": _clamp01(metrics.get("movement_score")),
        "alignment_score": _clamp01(metrics.get("alignment_score")),
        "face_detected_ratio": _clamp01(metrics.get("face_detected_ratio")),
        "method": str(metrics.get("method") or "frontend_vision"),
    }


def _get_bp_estimator():
    global bp_estimator
    if bp_estimator is None:
        bp_estimator = BloodPressureEstimator(
            model_path=current_app.config.get("RESEARCH_BP_MODEL_PATH") or None,
            scaler_info_path=current_app.config.get("RESEARCH_SCALER_INFO_PATH") or None,
            target_len=int(current_app.config.get("RESEARCH_MODEL_TARGET_LEN", 875)),
        )
    return bp_estimator


def _get_glucose_estimator():
    global glucose_estimator
    if glucose_estimator is None:
        glucose_estimator = BloodGlucoseEstimator(
            use_dummy=bool(current_app.config.get("USE_DUMMY_GLUCOSE", True))
        )
    return glucose_estimator


def _decode_base64_image(frame_b64: str):
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]
    binary = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(binary, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


@measurement_bp.route("/start", methods=["POST"])
@require_api_key
@rate_limit
def start_measurement():
    user_id = current_app.config.get("DEMO_USER_ID", "demo-user")

    measurement_id = str(uuid.uuid4())
    SESSIONS[measurement_id] = {
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
    }
    return jsonify({"success": True, "measurement_id": measurement_id, "user_id": user_id})


@measurement_bp.route("/process", methods=["POST"])
@require_api_key
@rate_limit
@validate_request_size
def process_measurement():
    payload = request.get_json(silent=True) or {}
    measurement_id = payload.get("measurement_id")
    frames = payload.get("frames", [])
    frame_rate = float(payload.get("frame_rate", 30))
    quality_metrics = _normalize_quality_metrics(payload.get("quality_metrics"))

    if not measurement_id or measurement_id not in SESSIONS:
        return jsonify({"success": False, "error": "Invalid measurement_id"}), 400

    if not frames or not isinstance(frames, list):
        return jsonify({"success": False, "error": "frames array is required"}), 400

    try:
        decoded_frames = [img for img in (_decode_base64_image(f) for f in frames) if img is not None]
        if len(decoded_frames) < 30:
            return jsonify({"success": False, "error": "Not enough valid frames"}), 400

        ppg_signal, quality = rppg_estimator.extract_signal(decoded_frames, frame_rate)
        features = rppg_estimator.extract_features(ppg_signal, frame_rate, quality=quality)

        bp_result = _get_bp_estimator().estimate(features, ppg_signal=ppg_signal, frame_rate=frame_rate)
        gl_result = _get_glucose_estimator().estimate(features)

        confidence = round((bp_result["confidence"] + gl_result["confidence"]) / 2.0, 3)
        user_id = SESSIONS[measurement_id]["user_id"]

        saved_id = current_app.db.save_measurement(
            user_id=user_id,
            bp_sys=bp_result["systolic"],
            bp_dia=bp_result["diastolic"],
            glucose=gl_result["glucose"],
            confidence=confidence,
        )

        if any(
            quality_metrics.get(k) is not None
            for k in ["lighting_score", "movement_score", "alignment_score", "face_detected_ratio"]
        ):
            current_app.db.save_measurement_quality(
                measurement_id=saved_id,
                lighting_score=quality_metrics.get("lighting_score"),
                movement_score=quality_metrics.get("movement_score"),
                alignment_score=quality_metrics.get("alignment_score"),
                face_detected_ratio=quality_metrics.get("face_detected_ratio"),
                method=quality_metrics.get("method") or "frontend_vision",
            )

        return jsonify(
            {
                "success": True,
                "data": {
                    "measurement_id": saved_id,
                    "bp_systolic": bp_result["systolic"],
                    "bp_diastolic": bp_result["diastolic"],
                    "blood_sugar": gl_result["glucose"],
                    "confidence": confidence,
                    "bp_source": bp_result.get("source", "unknown"),
                    "glucose_source": gl_result.get("source", "unknown"),
                    "quality_metrics": quality_metrics,
                },
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        SESSIONS.pop(measurement_id, None)


@measurement_bp.route("/result/<int:result_id>", methods=["GET"])
@require_api_key
@rate_limit
def get_measurement_result(result_id: int):
    result = current_app.db.get_measurement_by_id(result_id)
    if not result:
        return jsonify({"success": False, "error": "Result not found"}), 404
    return jsonify({"success": True, "data": result})
