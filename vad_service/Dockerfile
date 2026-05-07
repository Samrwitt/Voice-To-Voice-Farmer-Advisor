FROM farmer-advisor-base
WORKDIR /app

COPY vad_service/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 --retries=10 -r /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 --retries=10 silero-vad --no-deps

COPY vad_service/*.py /app/

RUN mkdir -p /app/utterances
EXPOSE 8010
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]