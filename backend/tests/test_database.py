import sqlite3

import pytest

from app.database import Database

LEGACY_SCHEMA = """
CREATE TABLE devices (
    device_id TEXT PRIMARY KEY, api_key_hash TEXT NOT NULL, created_at TEXT NOT NULL, last_seen_at TEXT
);
CREATE TABLE users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, username TEXT NOT NULL REFERENCES users(username), expires_at TEXT);
CREATE TABLE failed_logins (username TEXT NOT NULL, attempted_at TEXT NOT NULL);
CREATE TABLE readings (
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    measured_at TEXT NOT NULL, current_rms REAL NOT NULL, received_at TEXT NOT NULL,
    PRIMARY KEY (device_id, measured_at)
) WITHOUT ROWID;
CREATE TABLE predictions (
    device_id TEXT NOT NULL, target_time TEXT NOT NULL, made_at TEXT NOT NULL,
    predicted_power_w REAL NOT NULL, peak_threshold_w REAL NOT NULL, is_peak INTEGER NOT NULL,
    PRIMARY KEY (device_id, target_time)
) WITHOUT ROWID;
INSERT INTO devices VALUES ('medidor-01', 'hash-viejo', '2026-11-20T00:00:00Z', NULL);
INSERT INTO readings VALUES ('medidor-01', '2026-11-20T10:00:00Z', 2.5, '2026-11-20T10:05:00Z');
INSERT INTO predictions VALUES ('medidor-01', '2026-11-20T11:00:00Z', '2026-11-20T10:06:00Z', 300.0, 1364.3, 0);
"""


def table_columns(database_path, table_name: str) -> list[str]:
    with sqlite3.connect(database_path) as connection:
        return [row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")]


def table_names(database_path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


@pytest.fixture
def migrated_legacy_database(tmp_path):
    database_path = tmp_path / "vieja.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(LEGACY_SCHEMA)
    Database(database_path).initialize_schema()
    return database_path


def test_legacy_authentication_schema_is_removed_keeping_devices(migrated_legacy_database):
    assert "api_key_hash" not in table_columns(migrated_legacy_database, "devices")
    assert not {"users", "sessions", "failed_logins"} & table_names(migrated_legacy_database)
    with sqlite3.connect(migrated_legacy_database) as connection:
        assert connection.execute("SELECT device_id FROM devices").fetchall() == [("medidor-01",)]


def test_legacy_readings_in_amperes_are_converted_to_kilowatts(migrated_legacy_database):
    assert "current_rms" not in table_columns(migrated_legacy_database, "readings")
    with sqlite3.connect(migrated_legacy_database) as connection:
        (power_kw,) = connection.execute("SELECT power_kw FROM readings").fetchone()
    assert power_kw == pytest.approx(0.3)  # 2.5 A × 120 V = 300 W


def test_legacy_predictions_in_watts_are_discarded(migrated_legacy_database):
    columns = table_columns(migrated_legacy_database, "predictions")
    assert "predicted_power_kw" in columns and "predicted_power_w" not in columns
    with sqlite3.connect(migrated_legacy_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone() == (0,)


def test_initializing_twice_is_harmless(tmp_path):
    database = Database(tmp_path / "nueva.db")

    database.initialize_schema()
    database.initialize_schema()

    assert {"devices", "readings", "predictions"} <= table_names(tmp_path / "nueva.db")
    assert "power_kw" in table_columns(tmp_path / "nueva.db", "readings")
