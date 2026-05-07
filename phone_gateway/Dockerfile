FROM python:3.11-slim
WORKDIR /app

COPY requirements-common.txt /app/requirements-common.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 --retries=10 -r /app/requirements-common.txt

COPY phone_gateway/backend/requirements.txt /app/backend/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 --retries=10 -r /app/backend/requirements.txt

COPY phone_gateway/backend /app/backend
COPY phone_gateway/frontend /app/frontend

RUN mkdir -p /app/recordings/audio
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]