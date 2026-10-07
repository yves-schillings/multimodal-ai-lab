FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
# Same MLflow version as requirements.txt; psycopg2 for the PostgreSQL backend store.
RUN pip install --no-cache-dir mlflow==3.16.1 psycopg2-binary==2.9.10 && useradd --create-home --uid 10002 mlflow && mkdir -p /mlflow/artifacts && chown -R mlflow:mlflow /mlflow
USER mlflow
EXPOSE 5000
