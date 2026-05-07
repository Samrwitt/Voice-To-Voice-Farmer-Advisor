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
RUN pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt || \
    pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt

COPY asr_service/ /app/

ENV ASR_DEVICE=cpu
ENV ASR_COMPUTE_TYPE=int8
ENV ASR_MAX_NEW_TOKENS=160
ENV ASR_USE_VAD=true

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]