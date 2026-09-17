# syntax=docker/dockerfile:1
# Jarvis — production container. The ~100 MB embedding model is pre-cached
# during the build so the first request never pays the download cost.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE=/opt/fastembed \
    MPLBACKEND=Agg

WORKDIR /app

# Install dependencies first (layer caching) and pre-download the embedding
# model so the runtime image needs no internet for embeddings. The OCR engine
# is instantiated once here to prove the scan-parsing stack works in the image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" \
    && python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

COPY . .

# data/ holds the SQLite DB, config and usage meter; mount a volume for it in
# production so documents/chats/memory survive container replacement.
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8000
# Render (and most hosts) assign the listen port via $PORT. A hardcoded port
# means the proxy routes to a port nothing listens on -> HTTP 502 on every API
# call (uploads, chat, state). `exec` keeps uvicorn as PID 1 for clean SIGTERM.
CMD ["sh", "-c", "exec python -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
