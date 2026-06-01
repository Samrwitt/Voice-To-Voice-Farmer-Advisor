# # syntax=docker/dockerfile:1
# FROM farmer-advisor-base
# WORKDIR /app

# COPY requirements-common.txt .
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install -r requirements-common.txt

# COPY asr_service/requirements.txt .
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install -r requirements.txt

# COPY asr_service/ /app/

# EXPOSE 8001
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

FROM farmer-advisor-base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY asr_service/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 --retries=20 -r requirements.txt

COPY asr_service/ /app/

# One-time SpeechBrain + wav2vec2-large download baked into the image (not the host volume).
ARG HF_TOKEN=
ARG BAKE_SPEECHBRAIN=1
ENV ASR_SPEECHBRAIN_SOURCE=speechbrain/asr-wav2vec2-dvoice-amharic
ENV ASR_SPEECHBRAIN_SAVEDIR=/opt/asr-models/speechbrain-wav2vec2-dvoice-amharic
ENV HF_HOME=/opt/asr-models/hf-cache
ENV HF_TOKEN=${HF_TOKEN}
RUN mkdir -p /opt/asr-models
RUN if [ "$BAKE_SPEECHBRAIN" = "1" ]; then python bake_speechbrain_model.py; else echo "[bake] skipped (BAKE_SPEECHBRAIN=0)"; fi

ENV ASR_DEVICE=cpu
ENV ASR_COMPUTE_TYPE=int8
ENV ASR_MAX_NEW_TOKENS=160
ENV ASR_USE_VAD=true
ENV ASR_SPEECHBRAIN_SAVEDIR=/opt/asr-models/speechbrain-wav2vec2-dvoice-amharic
ENV HF_HOME=/opt/asr-models/hf-cache

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]