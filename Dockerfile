FROM python:3.12-slim

LABEL maintainer="sentinel-aiops" \
      description="Sentinel-AIOps MCP Inference Server"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models/ ./models/
COPY mcp-server/ ./mcp-server/
COPY data/ ./data/

# Expose Prometheus metrics port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:9090/metrics')" || exit 1

# Run the MCP server with Prometheus metrics
CMD ["python", "mcp-server/server.py"]
