# syntax=docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /app

# ── CPU-only torch (no CUDA packages downloaded) ──────────────────────────────
# torch's CUDA packages (nvidia-cudnn, nvidia-cublas, etc.) are optional —
# they're only needed if you call .cuda(). We never do on this hardware.
# --no-deps skips their download entirely; we add torch's actual runtime deps manually.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.2 torchaudio==2.2.2 --no-deps && \
    pip install filelock typing-extensions sympy networkx jinja2 fsspec

# ── Everything else ───────────────────────────────────────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=300 -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
