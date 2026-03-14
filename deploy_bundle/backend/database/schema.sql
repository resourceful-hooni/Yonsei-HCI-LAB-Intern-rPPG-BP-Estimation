CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    bp_systolic INTEGER NOT NULL,
    bp_diastolic INTEGER NOT NULL,
    blood_sugar INTEGER NOT NULL,
    confidence REAL,
    measurement_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS status_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    status_level INTEGER NOT NULL,
    recorded_date DATE NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS measurement_quality (
    quality_id INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id INTEGER NOT NULL UNIQUE,
    lighting_score REAL,
    movement_score REAL,
    alignment_score REAL,
    face_detected_ratio REAL,
    method TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (measurement_id) REFERENCES measurements(measurement_id)
);

CREATE TABLE IF NOT EXISTS habit_checkins (
    checkin_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    habit_id TEXT NOT NULL,
    check_date DATE NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    UNIQUE(user_id, habit_id, check_date)
);

CREATE TABLE IF NOT EXISTS notification_settings (
    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    setting_type TEXT NOT NULL,
    setting_time TEXT NOT NULL,
    weekdays TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    UNIQUE(user_id, setting_type)
);

CREATE INDEX IF NOT EXISTS idx_measurements_user_time ON measurements(user_id, measurement_time DESC);
CREATE INDEX IF NOT EXISTS idx_habit_checkins_user_date ON habit_checkins(user_id, check_date DESC);
CREATE INDEX IF NOT EXISTS idx_quality_measurement_id ON measurement_quality(measurement_id);
