from datetime import datetime

from app.database import Database, from_storage_timestamp, to_storage_timestamp


class DevicesRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def ensure_device_exists(self, device_id: str, now: datetime) -> None:
        """Da de alta el dispositivo la primera vez que envía datos; si ya existe, no hace nada."""
        with self.database.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO devices (device_id, created_at) VALUES (?, ?)",
                (device_id, to_storage_timestamp(now)),
            )

    def update_last_seen(self, device_id: str, moment: datetime) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE devices SET last_seen_at = ? WHERE device_id = ?",
                (to_storage_timestamp(moment), device_id),
            )

    def list_device_ids(self) -> list[str]:
        with self.database.connect() as connection:
            rows = connection.execute("SELECT device_id FROM devices ORDER BY device_id").fetchall()
        return [row["device_id"] for row in rows]

    def list_last_seen(self) -> list[tuple[str, datetime | None]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT device_id, last_seen_at FROM devices ORDER BY device_id"
            ).fetchall()
        return [
            (row["device_id"], from_storage_timestamp(row["last_seen_at"]) if row["last_seen_at"] else None)
            for row in rows
        ]
