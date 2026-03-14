from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime, timedelta


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self, schema_path: str):
        with self._get_connection() as conn:
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())

    def ensure_user(self, user_id: str, username: str | None = None):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (user_id, username) VALUES (?, ?)",
                (user_id, username),
            )
            conn.commit()

    def save_measurement(self, user_id: str, bp_sys: int, bp_dia: int, glucose: int, confidence: float):
        self.ensure_user(user_id)
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO measurements (user_id, bp_systolic, bp_diastolic, blood_sugar, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, int(bp_sys), int(bp_dia), int(glucose), float(confidence)),
            )
            conn.commit()
            return cursor.lastrowid

    def get_latest_measurement(self, user_id: str):
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM measurements
                WHERE user_id = ?
                ORDER BY measurement_time DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_recent_measurements(self, user_id: str, days: int = 7):
        start_time = datetime.now() - timedelta(days=days)
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM measurements
                WHERE user_id = ? AND measurement_time >= ?
                ORDER BY measurement_time ASC
                """,
                (user_id, start_time.strftime("%Y-%m-%d %H:%M:%S")),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_average_measurements(self, user_id: str, days: int = 7):
        start_time = datetime.now() - timedelta(days=days)
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT AVG(bp_systolic) as avg_systolic,
                       AVG(bp_diastolic) as avg_diastolic,
                       AVG(blood_sugar) as avg_glucose,
                       AVG(confidence) as avg_confidence
                FROM measurements
                WHERE user_id = ? AND measurement_time >= ?
                """,
                (user_id, start_time.strftime("%Y-%m-%d %H:%M:%S")),
            ).fetchone()
            return dict(row) if row else None

    def get_measurement_by_id(self, measurement_id: int):
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM measurements WHERE measurement_id = ?",
                (measurement_id,),
            ).fetchone()
            return dict(row) if row else None

    def save_measurement_quality(
        self,
        measurement_id: int,
        lighting_score: float | None,
        movement_score: float | None,
        alignment_score: float | None,
        face_detected_ratio: float | None,
        method: str = "frontend_vision",
    ):
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO measurement_quality (
                    measurement_id, lighting_score, movement_score, alignment_score, face_detected_ratio, method
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(measurement_id)
                DO UPDATE SET
                    lighting_score = excluded.lighting_score,
                    movement_score = excluded.movement_score,
                    alignment_score = excluded.alignment_score,
                    face_detected_ratio = excluded.face_detected_ratio,
                    method = excluded.method,
                    created_at = CURRENT_TIMESTAMP
                """,
                (
                    int(measurement_id),
                    None if lighting_score is None else float(lighting_score),
                    None if movement_score is None else float(movement_score),
                    None if alignment_score is None else float(alignment_score),
                    None if face_detected_ratio is None else float(face_detected_ratio),
                    method,
                ),
            )
            conn.commit()

    def get_measurement_quality(self, measurement_id: int):
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT measurement_id, lighting_score, movement_score, alignment_score, face_detected_ratio, method
                FROM measurement_quality
                WHERE measurement_id = ?
                LIMIT 1
                """,
                (int(measurement_id),),
            ).fetchone()
            return dict(row) if row else None

    def save_habit_checkin(self, user_id: str, habit_id: str, check_date: str, completed: bool):
        self.ensure_user(user_id)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO habit_checkins (user_id, habit_id, check_date, completed)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, habit_id, check_date)
                DO UPDATE SET completed = excluded.completed, created_at = CURRENT_TIMESTAMP
                """,
                (user_id, habit_id, check_date, 1 if completed else 0),
            )
            conn.commit()

    def get_habit_checkins(self, user_id: str, days: int = 7):
        start_date = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT habit_id, check_date, completed
                FROM habit_checkins
                WHERE user_id = ? AND check_date >= ?
                ORDER BY check_date ASC
                """,
                (user_id, start_date),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_habit_progress(self, user_id: str, days: int = 7):
        checkins = self.get_habit_checkins(user_id, days=days)
        by_habit = {
            "sleep": {"completed_days": 0, "target_days": days},
            "water": {"completed_days": 0, "target_days": days},
            "salt": {"completed_days": 0, "target_days": days},
        }

        for row in checkins:
            habit_id = row["habit_id"]
            if habit_id in by_habit and int(row["completed"]) == 1:
                by_habit[habit_id]["completed_days"] += 1

        for value in by_habit.values():
            target = max(1, value["target_days"])
            value["progress"] = round(value["completed_days"] / target * 100, 1)

        return by_habit

    def save_notification_setting(self, user_id: str, setting_type: str, setting_time: str, weekdays: list[str], enabled: bool):
        self.ensure_user(user_id)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO notification_settings (user_id, setting_type, setting_time, weekdays, enabled)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, setting_type)
                DO UPDATE SET
                    setting_time = excluded.setting_time,
                    weekdays = excluded.weekdays,
                    enabled = excluded.enabled,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, setting_type, setting_time, json.dumps(weekdays, ensure_ascii=False), 1 if enabled else 0),
            )
            conn.commit()

    def get_notification_settings(self, user_id: str):
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT setting_type, setting_time, weekdays, enabled
                FROM notification_settings
                WHERE user_id = ?
                ORDER BY setting_type ASC
                """,
                (user_id,),
            ).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            try:
                item["weekdays"] = json.loads(item.get("weekdays") or "[]")
            except Exception:
                item["weekdays"] = []
            item["enabled"] = bool(item.get("enabled"))
            result.append(item)
        return result

    def get_before_after_stats(self, user_id: str, days: int = 14):
        recent = self.get_recent_measurements(user_id, days=days)
        if len(recent) < 4:
            return None

        half = len(recent) // 2
        before = recent[:half]
        after = recent[half:]

        def _avg(rows, key):
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        result = {
            "before": {
                "bp_systolic": _avg(before, "bp_systolic"),
                "blood_sugar": _avg(before, "blood_sugar"),
            },
            "after": {
                "bp_systolic": _avg(after, "bp_systolic"),
                "blood_sugar": _avg(after, "blood_sugar"),
            },
        }
        result["delta"] = {
            "bp_systolic": round(result["after"]["bp_systolic"] - result["before"]["bp_systolic"], 2),
            "blood_sugar": round(result["after"]["blood_sugar"] - result["before"]["blood_sugar"], 2),
        }
        return result
