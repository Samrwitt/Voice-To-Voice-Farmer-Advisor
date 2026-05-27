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

COPY shared/ /app/shared/
COPY asr_service/ /app/

ENV PYTHONPATH=/app
# auto = cuda when the container sees a GPU, else cpu+int8
ENV ASR_DEVICE=auto
ENV ASR_GPU_COMPUTE_TYPE=float16
ENV ASR_CPU_COMPUTE_TYPE=int8
ENV ASR_MAX_NEW_TOKENS=384
ENV ASR_MAX_NEW_TOKENS_DYNAMIC=1
ENV ASR_USE_VAD=true
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]