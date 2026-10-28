import pytest


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json() == {"success": True, "status": "ok"}


def test_api_key_required(client):
    assert client.post("/api/measurement/start").status_code == 401


def test_cors_header_for_frontend_origin(client):
    res = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert res.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"


def test_unknown_route_returns_json_404(client):
    res = client.get("/api/does-not-exist")
    assert res.status_code == 404
    assert res.get_json()["success"] is False


def test_process_rejects_unknown_session(client, headers):
    res = client.post("/api/measurement/process", headers=headers, json={"measurement_id": "nope", "frames": []})
    assert res.status_code == 400


@pytest.mark.parametrize(
    "path",
    [
        "/api/lifestyle/comparison",
        "/api/lifestyle/recommendations",
        "/api/lifestyle/habits/progress",
        "/api/lifestyle/before-after",
        "/api/lifestyle/notifications",
        "/api/summary/daily",
        "/api/summary/trends",
    ],
)
def test_get_endpoints(client, headers, path):
    res = client.get(path, headers=headers)
    assert res.status_code == 200, res.get_json()
    assert res.get_json()["success"] is True


def test_habit_checkin_and_notifications(client, headers):
    res = client.post("/api/lifestyle/habits/checkin", headers=headers, json={"habit_id": "water", "completed": True})
    assert res.status_code == 200
    assert client.post("/api/lifestyle/habits/checkin", headers=headers, json={"habit_id": "bad"}).status_code == 400

    res = client.post(
        "/api/lifestyle/notifications",
        headers=headers,
        json={"setting_type": "measurement", "setting_time": "08:30", "weekdays": [1, 3, 5], "enabled": True},
    )
    assert res.status_code == 200
