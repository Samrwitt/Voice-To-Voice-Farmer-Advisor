# syntax=docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential python3-dev cmake \
    && rm -rf /var/lib/apt/lists/*

# ── CPU-only torch (no CUDA packages) ─────────────────────────────────────────
# sentence-transformers requires torch; pre-installing it with --no-deps
# means pip won't re-resolve or download CUDA packages when installing
# sentence-transformers later (it sees torch already satisfied).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.2 --no-deps && \
    pip install filelock typing-extensions sympy networkx jinja2 fsspec

# ── Everything else ───────────────────────────────────────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --default-timeout=300 -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
