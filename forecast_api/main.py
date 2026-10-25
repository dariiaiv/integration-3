import os
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast-api")

MODEL_NAME = os.getenv("MODEL_NAME", "house-price")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "prod")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = MlflowClient()
    logger.info(f"Завантаження моделі {MODEL_NAME} за аліасом @{MODEL_ALIAS} з MLflow...")
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        app.state.model = mlflow.pyfunc.load_model(model_uri)
        app.state.model_version = str(mv.version)
        logger.info(f"Успішно завантажено модель версії {app.state.model_version}")

        # Безпечне завантаження статистики
        try:
            stats_path = mlflow.artifacts.download_artifacts(
                run_id=mv.run_id, artifact_path="data_stats.json"
            )
            app.state.data_stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        except Exception:
            app.state.data_stats = {"message": "Stats available for XGBoost model only"}
    except Exception as e:
        logger.error(f"Помилка завантаження моделі з реєстру: {e}")
        raise RuntimeError(f"Fail-fast: не вдалося завантажити модель за аліасом @{MODEL_ALIAS}") from e
    yield

app = FastAPI(title="Forecast API", lifespan=lifespan)

class PropertyData(BaseModel):
    area: float
    rooms: int
    district: str = "Центр"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": getattr(app.state, "model_version", "unknown")
    }

@app.get("/model/data-stats")
def get_data_stats():
    return getattr(app.state, "data_stats", {})

@app.post("/predict")
def predict_price(prop: PropertyData):
    df = pd.DataFrame([{
        "area": prop.area,
        "rooms": prop.rooms,
        "district": prop.district
    }])
    pred = app.state.model.predict(df)[0]
    return {
        "predicted_price": round(float(pred), 2),
        "model_version": app.state.model_version
    }