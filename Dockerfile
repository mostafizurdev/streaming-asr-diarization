FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl bzip2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY scripts/download_models.sh scripts/
RUN bash scripts/download_models.sh

COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
RUN pip install --no-cache-dir .

ENV NUM_THREADS=2
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -fs http://localhost:8000/healthz || exit 1
CMD ["rtspeech-server", "--host", "0.0.0.0", "--port", "8000"]
