import os
import sys
import urllib.request
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

API_KEY = "test-api-key"
FACE_IMAGE_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    # Config는 import 시점에 환경변수를 읽으므로 import 전에 설정
    os.environ["API_KEY"] = API_KEY
    os.environ["DATABASE_PATH"] = str(tmp_path_factory.mktemp("db") / "test.db")
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"

    from app import create_app

    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def headers():
    return {"X-API-Key": API_KEY}


@pytest.fixture(scope="session")
def face_image(tmp_path_factory):
    import cv2

    path = tmp_path_factory.mktemp("img") / "portrait.jpg"
    try:
        urllib.request.urlretrieve(FACE_IMAGE_URL, path)
    except Exception as exc:
        pytest.skip(f"test face image unavailable: {exc}")
    image = cv2.imread(str(path))
    assert image is not None
    return image
