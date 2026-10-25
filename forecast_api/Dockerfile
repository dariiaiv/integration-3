FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    "mlflow==2.19.0" \
    "sqlalchemy<=2.0.35" \
    scikit-learn \
    xgboost \
    pandas \
    numpy \
    pyarrow

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]