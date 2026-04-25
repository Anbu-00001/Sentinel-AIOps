FROM python:3.12-slim

LABEL maintainer="sentinel-aiops" \
      description="Sentinel-AIOps FastMCP Inference Server"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY the modules needed by the MCP server
COPY models/ ./models/
COPY mcp_server/ ./mcp_server/
COPY database/ ./database/
COPY data/ ./data/
COPY config.py ./config.py

# Expose Prometheus metrics port
EXPOSE 9090

# Health check using stdlib (no external dependencies)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9090/metrics')" || exit 1

# Run the MCP server
CMD ["python", "mcp_server/server.py"]
