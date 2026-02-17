from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import wraps

from flask import current_app, jsonify, request

REQUEST_LOGS = defaultdict(deque)


def require_api_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        expected = current_app.config.get("API_KEY")
        if expected and api_key != expected:
            return jsonify({"success": False, "error": "Unauthorized"}), 401
        return func(*args, **kwargs)

    return wrapper


def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        limit = int(current_app.config.get("RATE_LIMIT_PER_MINUTE", 100))
        now = datetime.now()
        ip = request.remote_addr or "unknown"

        entries = REQUEST_LOGS[ip]
        while entries and entries[0] < now - timedelta(minutes=1):
            entries.popleft()

        if len(entries) >= limit:
            return jsonify({"success": False, "error": "Too Many Requests"}), 429

        entries.append(now)
        return func(*args, **kwargs)

    return wrapper


def validate_request_size(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_len = int(current_app.config.get("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))
        content_length = request.content_length or 0
        if content_length > max_len:
            return jsonify({"success": False, "error": "Payload Too Large"}), 413
        return func(*args, **kwargs)

    return wrapper
