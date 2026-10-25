FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir \
    "mlflow==2.19.0" \
    "sqlalchemy<=2.0.35" \
    xgboost \
    optuna \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    pyarrow \
    psycopg2-binary

COPY data.csv .
COPY train.py .

CMD ["python", "train.py"]