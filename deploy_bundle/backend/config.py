import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DEMO_USER_ID = os.getenv("DEMO_USER_ID", "demo-user")
    USE_DUMMY_GLUCOSE = os.getenv("USE_DUMMY_GLUCOSE", "True").lower() == "true"

    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
    API_KEY = os.getenv("API_KEY", "your-frontend-api-key")
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    DATABASE_PATH = os.getenv("DATABASE_PATH", "../data/visi_vital.db")
    CORS_ORIGINS = [origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000"
    ).split(",")]

    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "80")) * 1024 * 1024

    RESEARCH_BP_MODEL_PATH = os.getenv("RESEARCH_BP_MODEL_PATH", "")
    RESEARCH_SCALER_INFO_PATH = os.getenv("RESEARCH_SCALER_INFO_PATH", "")
    RESEARCH_MODEL_TARGET_LEN = int(os.getenv("RESEARCH_MODEL_TARGET_LEN", "875"))
