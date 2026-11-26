import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, precision_score, recall_score

from app.ml.features import FEATURE_COLUMNS, POWER_COLUMN, build_training_dataset

logger = logging.getLogger(__name__)

TRAIN_FRACTION = 0.8
PEAK_PERCENTILE = 90
MIN_TRAINING_ROWS = 100  # intervalos de 7 min útiles (~11,7 h con objetivo conocido)
ARTIFACT_FORMAT_VERSION = 3  # 3: intervalos de 7 min; 2: potencia en kW (15 min); 1: W
REQUIRED_ARTIFACT_KEYS = {"model", "peak_threshold", "feature_columns"}


class InsufficientDataError(Exception):
    pass


class InvalidModelArtifactError(Exception):
    pass


@dataclass(frozen=True)
class TrainingResult:
    artifact: dict
    metrics: dict[str, float]


def chronological_split(
    features: pd.DataFrame, target: pd.Series, train_fraction: float = TRAIN_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide por tiempo, nunca al azar: el set de prueba siempre es posterior al de entrenamiento."""
    split_position = int(len(features) * train_fraction)
    return (
        features.iloc[:split_position],
        features.iloc[split_position:],
        target.iloc[:split_position],
        target.iloc[split_position:],
    )


def train_peak_model(power_series: pd.Series, local_timezone: str, now: datetime | None = None) -> TrainingResult:
    features, target = build_training_dataset(power_series, local_timezone)
    if len(features) < MIN_TRAINING_ROWS:
        raise InsufficientDataError(
            f"Se necesitan al menos {MIN_TRAINING_ROWS} intervalos de 7 min útiles; hay {len(features)}"
        )

    train_features, test_features, train_target, test_target = chronological_split(features, target)
    train_features, features_without_data = _neutralize_empty_features(train_features)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
    model.fit(train_features, train_target)

    peak_threshold = float(np.percentile(train_features[POWER_COLUMN], PEAK_PERCENTILE))
    model_predictions = model.predict(test_features)
    persistence_predictions = test_features[POWER_COLUMN].to_numpy()  # baseline: el valor actual se mantiene

    metrics = {
        "mae_modelo_kw": float(mean_absolute_error(test_target, model_predictions)),
        "mae_persistencia_kw": float(mean_absolute_error(test_target, persistence_predictions)),
        **_peak_detection_metrics("modelo", test_target, model_predictions, peak_threshold),
        **_peak_detection_metrics("persistencia", test_target, persistence_predictions, peak_threshold),
        "umbral_pico_kw": peak_threshold,
        "filas_entrenamiento": float(len(train_features)),
        "filas_prueba": float(len(test_features)),
        # Si es 0, el recall no se puede evaluar: no hubo ningún pico real en el periodo de prueba.
        "picos_en_prueba": float((test_target.to_numpy() > peak_threshold).sum()),
        "features_sin_datos": float(len(features_without_data)),
    }
    artifact = {
        "model": model,
        "peak_threshold": peak_threshold,
        "feature_columns": list(FEATURE_COLUMNS),
        "format_version": ARTIFACT_FORMAT_VERSION,
        "trained_at": (now or datetime.now(UTC)).isoformat(),
        "metrics": metrics,
        "features_without_data": features_without_data,
    }
    return TrainingResult(artifact=artifact, metrics=metrics)


def save_model_artifact(artifact: dict, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".tmp")
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(model_path)  # reemplazo atómico: el backend nunca lee un archivo a medio escribir


def load_model_artifact(model_path: Path) -> dict:
    """Carga un artefacto propio. joblib puede ejecutar código al cargar: NUNCA usar con archivos de terceros."""
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or not REQUIRED_ARTIFACT_KEYS.issubset(artifact):
        raise InvalidModelArtifactError(f"{model_path} no tiene las claves {sorted(REQUIRED_ARTIFACT_KEYS)}")
    if list(artifact["feature_columns"]) != list(FEATURE_COLUMNS):
        raise InvalidModelArtifactError("Las columnas del modelo no coinciden con las features actuales; reentrene")
    return artifact


def _neutralize_empty_features(train_features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """scikit-learn no puede entrenar con una columna sin ningún dato (pasa con el lag de "ayer" cuando hay
    menos de un día de historial). Se llena con una constante: una columna constante no permite cortes en los
    árboles, así que el modelo la ignora y sigue funcionando cuando esa feature tenga datos al predecir."""
    empty_columns = [column for column in train_features.columns if train_features[column].isna().all()]
    if not empty_columns:
        return train_features, []
    logger.warning("Features sin datos en el entrenamiento (el modelo las ignorará): %s", empty_columns)
    return train_features.fillna({column: 0.0 for column in empty_columns}), empty_columns


def _peak_detection_metrics(
    prefix: str, actual_power: pd.Series, predicted_power: np.ndarray, peak_threshold: float
) -> dict[str, float]:
    actual_is_peak = actual_power.to_numpy() > peak_threshold
    predicted_is_peak = predicted_power > peak_threshold
    return {
        f"precision_picos_{prefix}": float(precision_score(actual_is_peak, predicted_is_peak, zero_division=0)),
        f"recall_picos_{prefix}": float(recall_score(actual_is_peak, predicted_is_peak, zero_division=0)),
    }
