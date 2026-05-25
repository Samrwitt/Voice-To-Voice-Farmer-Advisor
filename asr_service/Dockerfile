# syntax=docker/dockerfile:1
FROM farmer-advisor-base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY asr_service/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 -r requirements.txt || \
    pip install --default-timeout=100 -r requirements.txt

COPY asr_service/ /app/

ENV ASR_DEVICE=cpu
ENV ASR_COMPUTE_TYPE=int8
ENV ASR_MAX_NEW_TOKENS=160
ENV ASR_USE_VAD=true

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]