import numpy as np
import pytest

FRAME_RATE = 30.0


def _pulse_frames(face_image, seconds=5, bpm=72):
    import cv2

    h, w = face_image.shape[:2]
    base = cv2.resize(face_image, (320, int(320 * h / w))).astype(np.float64)
    rng = np.random.default_rng(0)
    frames = []
    for i in range(int(seconds * FRAME_RATE)):
        t = i / FRAME_RATE
        frame = base.copy()
        # 녹색 채널에 맥박 성분 + 약한 노이즈 (BGR 순서)
        frame[..., 1] *= 1.0 + 0.02 * np.sin(2 * np.pi * (bpm / 60.0) * t)
        frame += rng.normal(0, 1.0, frame.shape)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def test_extract_features_recovers_heart_rate():
    from models.rppg_estimator import RPPGEstimator

    t = np.arange(0, 10, 1 / FRAME_RATE)
    ppg = np.sin(2 * np.pi * 1.2 * t)

    features = RPPGEstimator().extract_features(ppg, FRAME_RATE, quality=0.9)

    assert features["heart_rate"] == pytest.approx(72, abs=2)
    assert all(np.isfinite(v) for v in features.values())


def test_no_face_raises():
    from models.rppg_estimator import RPPGEstimator

    blank = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(120)]
    with pytest.raises(ValueError):
        RPPGEstimator().extract_signal(blank, FRAME_RATE)


def test_face_landmarks_detected_on_every_frame(face_image):
    from models.rppg_estimator import RPPGEstimator

    frames = _pulse_frames(face_image)
    ppg, quality = RPPGEstimator().extract_signal(frames, FRAME_RATE)

    assert quality == pytest.approx(1.0)
    assert len(ppg) == len(frames)
    assert np.all(np.isfinite(ppg))


def test_research_scaler_parsing(tmp_path):
    from models.research_bp_model import ResearchBPModelBridge

    info = tmp_path / "scaler_info.txt"
    info.write_text(
        "Signal Statistics:\n  Mean: [0.5]\n  Scale: [2.0]\n"
        "Label Statistics:\n  Mean: [120.0 80.0]\n  Scale: [15.0 10.0]\n",
        encoding="utf-8",
    )
    bridge = ResearchBPModelBridge(model_path=str(tmp_path / "missing.h5"), scaler_info_path=str(info))

    assert not bridge.is_ready
    assert bridge.signal_mean.tolist() == [0.5]
    assert bridge.label_scale.tolist() == [15.0, 10.0]


def _encode_frames(frames):
    import base64

    import cv2

    out = []
    for f in frames:
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        out.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode())
    return out


def test_full_measurement_flow(client, headers, face_image):
    start = client.post("/api/measurement/start", headers=headers)
    assert start.status_code == 200
    measurement_id = start.get_json()["measurement_id"]

    res = client.post(
        "/api/measurement/process",
        headers=headers,
        json={
            "measurement_id": measurement_id,
            "frames": _encode_frames(_pulse_frames(face_image)),
            "frame_rate": FRAME_RATE,
            "quality_metrics": {"lighting_score": 0.8, "face_detected_ratio": 1.0},
        },
    )
    body = res.get_json()
    assert res.status_code == 200, body
    data = body["data"]
    assert 90 <= data["bp_systolic"] <= 180
    assert 60 <= data["bp_diastolic"] <= 120
    assert data["bp_diastolic"] < data["bp_systolic"]

    result = client.get(f"/api/measurement/result/{data['measurement_id']}", headers=headers)
    assert result.status_code == 200

    for path in ["/api/summary/daily", "/api/summary/trends"]:
        r = client.get(path, headers=headers)
        assert r.status_code == 200, (path, r.get_json())
        assert r.get_json()["success"] is True
