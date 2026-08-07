FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY conftest.py pytest.ini ./

# Bake the vector store into the image: pull the source docs and run the
# same ingestion pipeline documented in the README, so the container is
# self-contained (data/raw and chroma_data are gitignored, not shipped
# in the repo).
RUN git clone --depth 1 https://github.com/tensorflow/docs.git /tmp/tf-docs \
    && mkdir -p data/raw \
    && cp -r /tmp/tf-docs/site/en/guide data/raw/guide \
    && rm -rf /tmp/tf-docs \
    && python -m src.retrieval.vector_store

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
