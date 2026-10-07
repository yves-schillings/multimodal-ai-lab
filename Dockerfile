FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 HF_HUB_DISABLE_TELEMETRY=1 HOME=/tmp
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home --uid 10001 lab && mkdir -p /app/data /app/models && chown -R 10001:0 /app && chmod -R g=u /app
COPY --chown=10001:0 app.py .
COPY --chown=10001:0 lab ./lab
COPY --chown=10001:0 static ./static
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-nld fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
USER 10001
EXPOSE 8770
# Fail-closed defaults: loopback binding and loopback-only network mode. Docker Compose and
# OpenShift set LAB_BIND_HOST=0.0.0.0 and LAB_NETWORK_MODE=container explicitly.
ENV LAB_BIND_HOST=127.0.0.1 LAB_PORT=8770 LAB_NETWORK_MODE=loopback LAB_DATA_DIR=/app/data LAB_WHISPER_MODEL_DIR=/app/models/whisper-base
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health/ready' % os.environ.get('LAB_PORT','8770'))"
CMD ["sh","-c","exec python -m uvicorn app:app --host \"$LAB_BIND_HOST\" --port \"$LAB_PORT\" --no-access-log"]
