import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import optuna

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

SEED = 42
NUMERIC = ["area", "rooms"]
CATEGORICAL = ["district"]

def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "r2": float(r2_score(y, preds)),
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mape": float(mean_absolute_percentage_error(y, preds)),
    }

def build_xgb_pipeline(params: dict) -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", TransformedTargetRegressor(
            regressor=XGBRegressor(**params, random_state=SEED),
            func=np.log1p,
            inverse_func=np.expm1
        )),
    ])

def build_ridge_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", TransformedTargetRegressor(
            regressor=Ridge(random_state=SEED),
            func=np.log1p,
            inverse_func=np.expm1
        )),
    ])

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "house-price")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    client = MlflowClient()

    print("--- Завантаження датасету ---")
    df = pd.read_csv("data.csv")
    X = df[NUMERIC + CATEGORICAL]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # 1. Baseline Ridge run (R2.1)
    print("--- 1. Навчання Baseline (Ridge) ---")
    with mlflow.start_run(run_name="baseline-ridge") as baseline_run:
        ridge_pipe = build_ridge_pipeline()
        ridge_pipe.fit(X_train, y_train)
        baseline_metrics = evaluate(ridge_pipe, X_test, y_test)
        mlflow.log_metrics(baseline_metrics)
        sig = infer_signature(X_train, ridge_pipe.predict(X_train))
        mlflow.sklearn.log_model(ridge_pipe, artifact_path="model", signature=sig)
        v1 = mlflow.register_model(f"runs:/{baseline_run.info.run_id}/model", "house-price")
        print(f"Baseline зареєстровано як house-price версії {v1.version}, R2 = {baseline_metrics['r2']:.4f}")

    # 2. Optuna Search з вкладеними запусками (R2.4, R2.5)
    print("--- 2. Пошук гіперпараметрів Optuna (20 спроб) ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with mlflow.start_run(run_name="optuna-search"):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                pipe = build_xgb_pipeline(params)
                kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
                scores = cross_val_score(pipe, X_train, y_train, scoring="r2", cv=kf)
                cv_r2 = float(scores.mean())
                mlflow.log_params(params)
                mlflow.log_metric("cv_r2", cv_r2)
                return cv_r2

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=20)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_r2", study.best_value)
        print(f"Найкращі параметри Optuna: {study.best_params}, best CV R2 = {study.best_value:.4f}")

    # 3. Final XGBoost model (R2.6 - R2.11)
    print("--- 3. Фінальне навчання XGBoost ---")
    with mlflow.start_run(run_name="xgboost-final") as final_run:
        final_model = build_xgb_pipeline(study.best_params)
        final_model.fit(X_train, y_train)

        mlflow.log_params(study.best_params)
        final_metrics = evaluate(final_model, X_test, y_test)
        mlflow.log_metrics(final_metrics)
        print(f"Фінальні метрики XGBoost: {final_metrics}")

        # Реєстрація вхідного датасету (R2.8)
        dataset_source = mlflow.data.from_pandas(df, targets="price")
        mlflow.log_input(dataset_source, context="train")

        # Описові статистики data_stats.json (R2.9)
        stats = {
            "row_count": int(len(df)),
            "numeric_stats": df[NUMERIC + ["price"]].describe().to_dict(),
            "categorical_dist": {col: df[col].value_counts().to_dict() for col in CATEGORICAL},
            "target_skew_raw": float(y.skew()),
            "target_skew_log": float(np.log1p(y).skew()),
        }
        mlflow.log_dict(stats, "data_stats.json")
        mlflow.log_metric("row_count", stats["row_count"])
        mlflow.log_metric("target_skew_raw", stats["target_skew_raw"])
        mlflow.log_metric("target_skew_log", stats["target_skew_log"])

        # Графік розподілу ціни (R2.9)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(y, bins=25, color="skyblue", edgecolor="black")
        axes[0].set_title("Розподіл ціни (USD)")
        axes[1].hist(np.log1p(y), bins=25, color="salmon", edgecolor="black")
        axes[1].set_title("Розподіл після log1p")
        plt.tight_layout()
        mlflow.log_figure(fig, "target_distributions.png")
        plt.close()

        # Еталонний датасет reference_data.parquet (R2.10)
        X_train.to_parquet("reference_data.parquet")
        mlflow.log_artifact("reference_data.parquet")

        # Збереження моделі з сигнатурою (R2.7)
        signature = infer_signature(X_train, final_model.predict(X_train))
        mlflow.sklearn.log_model(final_model, artifact_path="model", signature=signature)

        # Реєстрація моделі та призначення аліаса @prod (R2.11)
        version = mlflow.register_model(f"runs:/{final_run.info.run_id}/model", "house-price")
        client.set_registered_model_alias("house-price", "prod", version.version)
        print(f"Модель XGBoost зареєстровано як house-price версії {version.version} з аліасом @prod!")

if __name__ == "__main__":
    main()