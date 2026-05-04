# syntax=docker/dockerfile:1
FROM farmer-advisor-base
WORKDIR /app

COPY vad_service/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install silero-vad --no-deps

COPY vad_service/main.py /app/main.py
COPY vad_service/vad_engine.py /app/vad_engine.py
COPY vad_service/audio_utils.py /app/audio_utils.py
COPY vad_service/asr_client.py /app/asr_client.py

RUN mkdir -p /app/utterances
EXPOSE 8010
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]