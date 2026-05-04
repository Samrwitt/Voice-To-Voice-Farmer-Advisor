# syntax=docker/dockerfile:1
FROM farmer-advisor-base
WORKDIR /app

COPY requirements-common.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-common.txt

COPY asr_service/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY asr_service/ /app/

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]