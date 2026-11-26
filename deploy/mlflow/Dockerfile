# The official MLflow image has no Postgres driver, so build a minimal one.
FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==3.18.0 psycopg2-binary "sqlalchemy<2.1"
EXPOSE 5000
