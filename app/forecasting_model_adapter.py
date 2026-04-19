from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .date_feature_generator import GeneratedFeatureRow, SeasonalFeatureGenerator


BASE_DIR = Path(__file__).resolve().parents[1]
LOCAL_STACK_DIR = BASE_DIR / "forecasting_stack"
LOCAL_ASSET_DIR = BASE_DIR / "forecasting_assets"
LEGACY_STACK_DIR = BASE_DIR.parent / "openvpp_forecasting_stack"
STACK_DIR = (
    LOCAL_STACK_DIR
    if (LOCAL_STACK_DIR / "scripts" / "train_three_year_models.py").exists()
    else LEGACY_STACK_DIR
)
ASSET_DIR = (
    LOCAL_ASSET_DIR
    if (LOCAL_ASSET_DIR / "data" / "training_3yr").exists()
    else STACK_DIR
)


@dataclass(frozen=True)
class HistoricalForecast:
    requested_date: str
    resolved_date: str
    match_type: str
    eia_prob: float
    weather_prob: float
    price_prob: float
    fusion_prob: float
    fusion_threshold: float
    weather_raw: dict[str, float]
    price_raw: dict[str, float]
    eia_raw: dict[str, float]
    source: str


@dataclass(frozen=True)
class RuntimeBundle:
    training: object
    eia_fit: object
    weather_fit: object
    price_fit: object
    fusion_fit: object
    eia_features: list[str]
    weather_features: list[str]
    price_features: list[str]
    fusion_features: list[str]
    fusion_threshold: float
    eia_generator: SeasonalFeatureGenerator
    weather_generator: SeasonalFeatureGenerator
    price_generator: SeasonalFeatureGenerator
    eia_exact_rows: dict[str, dict[str, float]]
    weather_exact_rows: dict[str, dict[str, float]]
    price_exact_rows: dict[str, dict[str, float]]


def _load_training_module():
    module_path = STACK_DIR / "scripts" / "train_three_year_models.py"
    spec = importlib.util.spec_from_file_location("openvpp_train_three_year_models", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BASE_DIR = ASSET_DIR
    module.DATA_DIR = ASSET_DIR / "data" / "training_3yr"
    module.ARTIFACT_DIR = ASSET_DIR / "artifacts" / "training_3yr"
    module.RAW_DIR = module.DATA_DIR / "raw"
    module.REPORT_DIR = module.ARTIFACT_DIR / "reports"
    return module


def _predict_single_prob(generated: GeneratedFeatureRow, fit_result, feature_cols: list[str]) -> float:
    frame = pd.DataFrame([{col: generated.values[col] for col in feature_cols}], columns=feature_cols)
    scaled = fit_result.scaler.transform(frame.fillna(0.0))
    return float(fit_result.model.predict_proba(scaled)[0, 1])


def _maybe_tenths_to_c(value: float, month: int) -> float:
    if value > 150:
        candidate_c = value / 10.0
        if candidate_c <= 55:
            return candidate_c
        candidate_f = value / 10.0
        return (candidate_f - 32.0) * 5.0 / 9.0
    if month in {6, 7, 8, 9} and value < 20:
        return value + 15.0
    return value


def _resolve_nearest_generated(requested_date: str, bundle: RuntimeBundle) -> tuple[GeneratedFeatureRow, GeneratedFeatureRow, GeneratedFeatureRow]:
    eia_row = bundle.eia_generator.generate(requested_date)
    weather_row = bundle.weather_generator.generate(requested_date)
    price_row = bundle.price_generator.generate(requested_date)
    return eia_row, weather_row, price_row


def _build_exact_row_map(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.sort_values("date")
    numeric = work[feature_columns].astype(float)
    means = numeric.mean()
    row_map: dict[str, dict[str, float]] = {}
    for idx, date_value in enumerate(work["date"]):
        row = numeric.iloc[idx].fillna(means)
        values = {col: float(row[col]) for col in feature_columns}
        row_map[date_value.date().isoformat()] = values
    return row_map


def _resolve_feature_rows(requested_date: str, bundle: RuntimeBundle) -> tuple[GeneratedFeatureRow, GeneratedFeatureRow, GeneratedFeatureRow]:
    target = pd.Timestamp(requested_date).normalize().date().isoformat()
    eia_exact = bundle.eia_exact_rows.get(target)
    weather_exact = bundle.weather_exact_rows.get(target)
    price_exact = bundle.price_exact_rows.get(target)
    if eia_exact is not None and weather_exact is not None and price_exact is not None:
        return (
            GeneratedFeatureRow(date=target, match_type="exact_historical", values=eia_exact),
            GeneratedFeatureRow(date=target, match_type="exact_historical", values=weather_exact),
            GeneratedFeatureRow(date=target, match_type="exact_historical", values=price_exact),
        )
    return _resolve_nearest_generated(requested_date, bundle)


def _build_runtime_bundle() -> RuntimeBundle:
    training = _load_training_module()
    datasets = training.prepare_datasets()
    eia_df = datasets["eia"]
    weather_df = datasets["weather"]
    price_df = datasets["price"]

    eia_features = training.pick_feature_columns(eia_df, exclude=["stress_label", "stress_score"])
    weather_features = training.pick_feature_columns(weather_df, exclude=["stress_label"])
    price_features = training.pick_feature_columns(price_df, exclude=["stress_label"])

    eia_train, eia_val, _ = training.time_split(eia_df)
    weather_train, weather_val, _ = training.time_split(weather_df)
    price_train, price_val, _ = training.time_split(price_df)

    eia_fit = training.fit_epoch_mlp(
        eia_train,
        eia_val,
        eia_features,
        hidden_layer_sizes=(64, 32),
        max_epochs=80,
        patience=12,
        random_state=1,
    )
    weather_fit = training.fit_epoch_mlp(
        weather_train,
        weather_val,
        weather_features,
        hidden_layer_sizes=(48, 24),
        max_epochs=80,
        patience=12,
        random_state=2,
    )
    price_fit = training.fit_epoch_mlp(
        price_train,
        price_val,
        price_features,
        hidden_layer_sizes=(32, 16),
        max_epochs=80,
        patience=12,
        random_state=3,
    )

    fusion_df = training.build_fusion_dataset(
        eia_df,
        weather_df,
        price_df,
        eia_fit,
        weather_fit,
        price_fit,
        eia_features,
        weather_features,
        price_features,
    )
    fusion_features = training.pick_feature_columns(fusion_df, exclude=["stress_label"])
    fusion_train, fusion_val, _ = training.time_split(fusion_df)
    fusion_fit = training.fit_epoch_mlp(
        fusion_train,
        fusion_val,
        fusion_features,
        hidden_layer_sizes=(32, 16),
        max_epochs=80,
        patience=12,
        random_state=4,
    )
    fusion_val_probs = fusion_fit.model.predict_proba(
        fusion_fit.scaler.transform(fusion_val[fusion_features].fillna(0.0))
    )[:, 1]
    tuned = training.search_threshold(fusion_val["stress_label"].astype(int).to_numpy(), fusion_val_probs)

    return RuntimeBundle(
        training=training,
        eia_fit=eia_fit,
        weather_fit=weather_fit,
        price_fit=price_fit,
        fusion_fit=fusion_fit,
        eia_features=eia_features,
        weather_features=weather_features,
        price_features=price_features,
        fusion_features=fusion_features,
        fusion_threshold=float(tuned["threshold"]),
        eia_generator=SeasonalFeatureGenerator(eia_train, feature_columns=eia_features),
        weather_generator=SeasonalFeatureGenerator(weather_train, feature_columns=weather_features),
        price_generator=SeasonalFeatureGenerator(price_train, feature_columns=price_features),
        eia_exact_rows=_build_exact_row_map(eia_df, eia_features),
        weather_exact_rows=_build_exact_row_map(weather_df, weather_features),
        price_exact_rows=_build_exact_row_map(price_df, price_features),
    )


@lru_cache(maxsize=1)
def _load_runtime_bundle() -> RuntimeBundle:
    return _build_runtime_bundle()


def get_historical_forecast_for_date(requested_date: str) -> HistoricalForecast:
    bundle = _load_runtime_bundle()
    eia_row, weather_row, price_row = _resolve_feature_rows(requested_date, bundle)

    eia_prob = _predict_single_prob(eia_row, bundle.eia_fit, bundle.eia_features)
    weather_prob = _predict_single_prob(weather_row, bundle.weather_fit, bundle.weather_features)
    price_prob = _predict_single_prob(price_row, bundle.price_fit, bundle.price_features)

    target = pd.Timestamp(requested_date).normalize()
    fusion_frame = pd.DataFrame(
        [
            {
                "eia_prob": eia_prob,
                "weather_prob": weather_prob,
                "price_prob": price_prob,
                "day_of_week": float(target.dayofweek),
                "month": float(target.month),
                "day_of_year": float(target.dayofyear),
                "is_weekend": float(1 if target.dayofweek >= 5 else 0),
                "prob_mean": float((eia_prob + weather_prob + price_prob) / 3.0),
                "prob_max": float(max(eia_prob, weather_prob, price_prob)),
                "prob_min": float(min(eia_prob, weather_prob, price_prob)),
            }
        ],
        columns=bundle.fusion_features,
    )
    fusion_scaled = bundle.fusion_fit.scaler.transform(fusion_frame.fillna(0.0))
    fusion_prob = float(bundle.fusion_fit.model.predict_proba(fusion_scaled)[0, 1])

    month = int(target.month)
    resolved_date = target.date().isoformat()
    if eia_row.match_type == "exact_historical":
        match_type = "exact_historical"
    elif eia_row.match_type == "generated" and weather_row.match_type == "generated" and price_row.match_type == "generated":
        match_type = "generated"
    else:
        match_type = "generated_seasonal"

    return HistoricalForecast(
        requested_date=requested_date,
        resolved_date=resolved_date,
        match_type=match_type,
        eia_prob=eia_prob,
        weather_prob=weather_prob,
        price_prob=price_prob,
        fusion_prob=fusion_prob,
        fusion_threshold=bundle.fusion_threshold,
        weather_raw={
            "tmax_c_est": round(_maybe_tenths_to_c(float(weather_row.values["tmax_mean"]), month), 3),
            "tmin_c_est": round(_maybe_tenths_to_c(float(weather_row.values["tmin_mean"]), month), 3),
            "temp_range_raw": round(float(weather_row.values.get("temp_range", 0.0)), 3),
            "rolling_7d_tmax_raw": round(float(weather_row.values.get("rolling_7d_tmax", 0.0)), 3),
        },
        price_raw={
            "price_all_sectors_cents_per_kwh": round(float(price_row.values["price_all_sectors"]), 3),
            "price_rolling_3m_cents_per_kwh": round(float(price_row.values.get("price_rolling_3m", 0.0)), 3),
            "price_rank": round(float(price_row.values.get("price_rank", 0.0)), 4),
        },
        eia_raw={
            "demand_mean_mw": round(float(eia_row.values["demand_mean"]), 3),
            "forecast_mean_mw": round(float(eia_row.values["forecast_mean"]), 3),
            "forecast_gap_mw": round(float(eia_row.values.get("forecast_gap", 0.0)), 3),
            "stress_score": round(float(eia_row.values.get("rolling_7d_gap", 0.0)), 4),
        },
        source="openvpp_forecasting_stack_exact_features"
        if match_type == "exact_historical"
        else "openvpp_forecasting_stack_generated_features",
    )

def evaluate_generated_forecasting_pipeline() -> dict[str, object]:
    bundle = _load_runtime_bundle()
    training = bundle.training
    datasets = training.prepare_datasets()
    eia_df = datasets["eia"]
    weather_df = datasets["weather"]
    price_df = datasets["price"]

    fusion_df = training.build_fusion_dataset(
        eia_df,
        weather_df,
        price_df,
        bundle.eia_fit,
        bundle.weather_fit,
        bundle.price_fit,
        bundle.eia_features,
        bundle.weather_features,
        bundle.price_features,
    )
    _, fusion_val, fusion_test = training.time_split(fusion_df)

    def _evaluate_frame(frame: pd.DataFrame, *, shift_years: int = 0) -> tuple[dict[str, object], pd.DataFrame]:
        rows: list[dict[str, object]] = []
        for date_value in frame["date"].astype(str).tolist():
            lookup_date = (pd.Timestamp(date_value) + pd.DateOffset(years=shift_years)).date().isoformat()
            forecast = get_historical_forecast_for_date(lookup_date)
            pred_label = 1 if forecast.fusion_prob >= bundle.fusion_threshold else 0
            actual_row = frame.loc[frame["date"].astype(str) == date_value].iloc[0]
            rows.append(
                {
                    "date": date_value,
                    "requested_date": lookup_date,
                    "match_type": forecast.match_type,
                    "actual_label": int(actual_row["stress_label"]),
                    "predicted_label": pred_label,
                    "fusion_probability": forecast.fusion_prob,
                    "eia_probability": forecast.eia_prob,
                    "weather_probability": forecast.weather_prob,
                    "price_probability": forecast.price_prob,
                }
            )

        result_df = pd.DataFrame(rows)
        y_true = result_df["actual_label"].astype(int)
        y_pred = result_df["predicted_label"].astype(int)
        metrics = {
            "rows": int(len(result_df)),
            "threshold": bundle.fusion_threshold,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        return metrics, result_df

    exact_metrics, exact_df = _evaluate_frame(fusion_test)
    generated_val_metrics, _ = _evaluate_frame(fusion_val, shift_years=2)
    generated_metrics, generated_df = _evaluate_frame(fusion_test, shift_years=2)
    return {
        "metrics": {
            "exact_historical_in_range": {**exact_metrics, "source": "exact_feature_rows_vs_test_labels"},
            "generated_out_of_range_validation": {
                **generated_val_metrics,
                "source": "generated_features_shifted_validation_vs_labels",
            },
            "generated_out_of_range_test": {
                **generated_metrics,
                "source": "generated_features_shifted_test_vs_labels",
            },
        },
        "predictions": {
            "exact_historical_in_range": exact_df,
            "generated_out_of_range_test": generated_df,
        },
    }
