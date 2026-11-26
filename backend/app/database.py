import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS devices (
        device_id     TEXT PRIMARY KEY,
        created_at    TEXT NOT NULL,
        last_seen_at  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS readings (
        device_id    TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
        measured_at  TEXT NOT NULL,
        power_kw     REAL NOT NULL,
        received_at  TEXT NOT NULL,
        PRIMARY KEY (device_id, measured_at)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        device_id           TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
        target_time         TEXT NOT NULL,
        made_at             TEXT NOT NULL,
        predicted_power_kw  REAL NOT NULL,
        peak_threshold_kw   REAL NOT NULL,
        is_peak             INTEGER NOT NULL,
        PRIMARY KEY (device_id, target_time)
    ) WITHOUT ROWID
    """,
)

OBSOLETE_AUTHENTICATION_TABLES = ("sessions", "failed_logins", "users")  # en orden para respetar llaves foráneas
LEGACY_NOMINAL_VOLTAGE = 120.0  # voltaje con el que versiones anteriores convertían amperios a potencia

STORAGE_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def to_storage_timestamp(moment: datetime) -> str:
    """Convierte a texto UTC de formato fijo, que se ordena correctamente como texto en SQLite."""
    if moment.tzinfo is None:
        raise ValueError("Se requiere un datetime con zona horaria")
    return moment.astimezone(UTC).strftime(STORAGE_TIMESTAMP_FORMAT)


def from_storage_timestamp(stored_value: str) -> datetime:
    return datetime.strptime(stored_value, STORAGE_TIMESTAMP_FORMAT).replace(tzinfo=UTC)


class Database:
    """Abre una conexión SQLite por operación; es barato y evita compartir conexiones entre hilos."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize_schema(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            _migrate_legacy_schema(connection)
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _migrate_legacy_schema(connection: sqlite3.Connection) -> None:
    """Actualiza bases creadas por versiones anteriores. Cada paso solo actúa si encuentra el esquema viejo."""
    _remove_obsolete_authentication_schema(connection)
    _convert_readings_from_amperes_to_kilowatts(connection)
    _drop_predictions_stored_in_watts(connection)


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})")}


def _remove_obsolete_authentication_schema(connection: sqlite3.Connection) -> None:
    """Versiones anteriores tenían login y API key por dispositivo."""
    if "api_key_hash" in _table_columns(connection, "devices"):
        connection.execute("ALTER TABLE devices DROP COLUMN api_key_hash")
    for table_name in OBSOLETE_AUTHENTICATION_TABLES:
        connection.execute(f"DROP TABLE IF EXISTS {table_name}")


def _convert_readings_from_amperes_to_kilowatts(connection: sqlite3.Connection) -> None:
    """Versiones anteriores guardaban corriente RMS (A); se convierte con el voltaje que usaban entonces."""
    if "current_rms" not in _table_columns(connection, "readings"):
        return
    connection.execute("ALTER TABLE readings RENAME COLUMN current_rms TO power_kw")
    converted = connection.execute(
        "UPDATE readings SET power_kw = power_kw * ? / 1000", (LEGACY_NOMINAL_VOLTAGE,)
    ).rowcount
    if converted:
        logger.warning("Se convirtieron %d lecturas de amperios a kW suponiendo %s V", converted, LEGACY_NOMINAL_VOLTAGE)


def _drop_predictions_stored_in_watts(connection: sqlite3.Connection) -> None:
    """Las predicciones son datos derivados: se descartan y el servicio las vuelve a generar en kW."""
    if "predicted_power_w" in _table_columns(connection, "predictions"):
        connection.execute("DROP TABLE predictions")
