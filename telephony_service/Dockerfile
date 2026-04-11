FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

COPY . .

CMD ["python", "main.py"]
