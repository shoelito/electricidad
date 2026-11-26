"""Entrena el modelo de picos y lo guarda en la ruta configurada (ELECTRICIDAD_MODEL_PATH).

Uso (desde backend/):
  python -m scripts.train_model --device-id medidor-01        # con datos reales de la BD
  python -m scripts.train_model --synthetic-days 28           # con datos sintéticos (para probar)
Luego, en el dashboard: "Recargar modelo". Desde el dashboard también se puede usar "Entrenar modelo".
"""

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta

from app.config import get_settings
from app.database import Database
from app.ml.features import resample_power_series
from app.ml.synthetic_data import generate_synthetic_readings
from app.ml.training import InsufficientDataError, save_model_artifact, train_peak_model
from app.repositories.readings_repository import ReadingsRepository
from app.services.training_service import train_and_save_from_database


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena el modelo de predicción de picos")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--device-id", help="Entrenar con las lecturas de este dispositivo")
    source.add_argument("--synthetic-days", type=int, help="Entrenar con N días de datos sintéticos")
    arguments = parser.parse_args()
    settings = get_settings()

    try:
        if arguments.synthetic_days:
            start = datetime.now(UTC) - timedelta(days=arguments.synthetic_days)
            readings = generate_synthetic_readings(start, arguments.synthetic_days)
            result = train_peak_model(resample_power_series(readings), settings.local_timezone)
            save_model_artifact(result.artifact, settings.model_path)
        else:
            readings_repository = ReadingsRepository(Database(settings.database_path))
            result = train_and_save_from_database(readings_repository, settings, arguments.device_id)
    except InsufficientDataError as error:
        print(f"Datos insuficientes: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    print(f"Modelo guardado en {settings.model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
