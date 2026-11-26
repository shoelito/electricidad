from datetime import timedelta

import joblib
import pandas as pd
import pytest

from app.ml.features import FEATURE_COLUMNS, build_feature_frame, resample_power_series
from app.ml.synthetic_data import generate_synthetic_readings
from app.ml.training import (
    InsufficientDataError,
    InvalidModelArtifactError,
    chronological_split,
    load_model_artifact,
    save_model_artifact,
    train_peak_model,
)
from tests.conftest import FIXED_NOW


def test_chronological_split_keeps_test_set_after_train_set():
    index = pd.date_range("2026-11-01", periods=10, freq="15min", tz="UTC")
    features = pd.DataFrame({"value": range(10)}, index=index)
    target = pd.Series(range(10), index=index)

    train_features, test_features, train_target, test_target = chronological_split(features, target)

    assert len(train_features) == 8 and len(test_features) == 2
    assert train_features.index.max() < test_features.index.min()
    assert train_target.index.equals(train_features.index)
    assert test_target.index.equals(test_features.index)


def test_model_beats_persistence_baseline_on_synthetic_data(trained_result):
    metrics = trained_result.metrics

    assert metrics["mae_modelo_kw"] < metrics["mae_persistencia_kw"]
    assert metrics["recall_picos_modelo"] > metrics["recall_picos_persistencia"]
    assert metrics["recall_picos_modelo"] >= 0.8


def test_artifact_contains_required_keys(trained_result):
    artifact = trained_result.artifact

    assert {"model", "peak_threshold", "feature_columns"} <= artifact.keys()
    assert artifact["feature_columns"] == FEATURE_COLUMNS
    assert artifact["peak_threshold"] > 0


def test_training_with_too_little_data_raises():
    ten_hours_of_readings = generate_synthetic_readings(FIXED_NOW - timedelta(days=0.4), days=0.4)
    power = resample_power_series(ten_hours_of_readings)  # 74 intervalos útiles, el mínimo es 100

    with pytest.raises(InsufficientDataError):
        train_peak_model(power, "America/Costa_Rica")


def test_saved_artifact_can_be_loaded_back(trained_result, tmp_path):
    model_path = tmp_path / "models" / "modelo.joblib"

    save_model_artifact(trained_result.artifact, model_path)
    loaded_artifact = load_model_artifact(model_path)

    assert loaded_artifact["peak_threshold"] == trained_result.artifact["peak_threshold"]
    assert not model_path.with_suffix(".tmp").exists()


@pytest.mark.parametrize(
    "invalid_artifact",
    [
        ["no", "es", "un", "diccionario"],
        {"model": None},
        {"model": None, "peak_threshold": 1.0, "feature_columns": ["otra_columna"]},
    ],
)
def test_invalid_artifacts_are_rejected(tmp_path, invalid_artifact):
    model_path = tmp_path / "invalido.joblib"
    joblib.dump(invalid_artifact, model_path)

    with pytest.raises(InvalidModelArtifactError):
        load_model_artifact(model_path)


def test_training_with_minimum_data_and_no_previous_day_succeeds():
    """Con ~13 h no existe el lag de "ayer" (206 intervalos): esa feature queda vacía y debe ignorarse."""
    thirteen_hours = generate_synthetic_readings(FIXED_NOW - timedelta(days=0.55), days=0.55)
    power = resample_power_series(thirteen_hours)

    result = train_peak_model(power, "America/Costa_Rica")

    assert result.metrics["filas_entrenamiento"] + result.metrics["filas_prueba"] >= 100
    assert result.artifact["features_without_data"] == ["power_lag_206"]
    assert result.metrics["features_sin_datos"] == 1


def test_model_trained_without_previous_day_can_predict_once_it_exists():
    thirteen_hours = generate_synthetic_readings(FIXED_NOW - timedelta(days=0.55), days=0.55)
    model = train_peak_model(resample_power_series(thirteen_hours), "America/Costa_Rica").artifact["model"]
    two_days = resample_power_series(generate_synthetic_readings(FIXED_NOW - timedelta(days=2), days=2))
    features_with_yesterday = build_feature_frame(two_days, "America/Costa_Rica").iloc[[-1]]

    assert features_with_yesterday["power_lag_206"].notna().all()
    assert model.predict(features_with_yesterday)[0] >= 0


def test_metrics_report_how_many_real_peaks_the_test_period_had(trained_result):
    assert trained_result.metrics["picos_en_prueba"] > 0
    assert trained_result.metrics["features_sin_datos"] == 0
