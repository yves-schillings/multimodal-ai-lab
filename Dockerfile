FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home --uid 10001 lab && mkdir -p /app/data /app/models && chown -R lab:lab /app
COPY --chown=lab:lab app.py .
COPY --chown=lab:lab lab ./lab
COPY --chown=lab:lab static ./static
USER lab
EXPOSE 8770
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8770/health/ready')"
CMD ["python","-m","uvicorn","app:app","--host","127.0.0.1","--port","8770","--no-access-log"]
