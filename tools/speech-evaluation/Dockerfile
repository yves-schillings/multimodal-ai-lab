# Optional evaluation helper, separate from the application runtime.
# eSpeak NG remains a third-party GPL-3.0-or-later tool, not Apache-licensed software.
FROM multimodal-ai-lab-app
USER root
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng && rm -rf /var/lib/apt/lists/*
USER 10001
ENTRYPOINT ["python", "/work/scripts/evaluate_speech_corpus.py"]
