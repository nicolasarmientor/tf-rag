FROM python:3.13-slim

WORKDIR /app
ENV HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY conftest.py pytest.ini ./

RUN git clone --depth 1 https://github.com/tensorflow/docs.git /tmp/tf-docs \
    && mkdir -p data/raw \
    && cp -r /tmp/tf-docs/site/en/guide data/raw/guide \
    && rm -rf /tmp/tf-docs \
    && python -m src.retrieval.vector_store

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]