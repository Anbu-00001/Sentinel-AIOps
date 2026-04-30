FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY mcp_server/ mcp_server/
COPY database/ database/
COPY dashboard/ dashboard/
COPY scripts/ scripts/
COPY config.py .
COPY sentinel_logging.py .

# Create writable data directory (no pre-existing db)
RUN mkdir -p /app/data/feedback && \
    chown -R 1000:1000 /app/data

# Ensure all Python imports resolve from /app
ENV PYTHONPATH=/app

EXPOSE 7860

USER 1000

# Run uvicorn directly on 7860 — no nginx/supervisord needed on HF Spaces
CMD ["python", "-m", "uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
