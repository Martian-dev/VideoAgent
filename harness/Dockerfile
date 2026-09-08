# The environment is the capability. Everything the model can do, it does
# through these binaries; the harness itself adds no video knowledge.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openai>=1.60 pywhispercpp>=1.5

# Bake the whisper model so a run never pauses to download it.
ENV HARNESS_WHISPER_DIR=/opt/whisper-models
ENV HARNESS_WHISPER_MODEL=medium.en-q5_0
RUN mkdir -p /opt/whisper-models && python -c "\
from pywhispercpp.model import Model; \
Model('medium.en-q5_0', models_dir='/opt/whisper-models')" 2>/dev/null || true

COPY bin/transcribe /usr/local/bin/transcribe
RUN chmod +x /usr/local/bin/transcribe

COPY . /opt/harness/harness/
ENV PYTHONPATH=/opt/harness

WORKDIR /app
RUN mkdir -p /app/data/source_pack /app/work /app/output
