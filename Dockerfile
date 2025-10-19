# HexDetector - Docker Container
# Multi-stage build for optimized image size

FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create necessary directories
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/logs \
    /app/output \
    /app/models

# Copy application code
COPY src/ /app/src/
COPY setup.py /app/
COPY README.md /app/
COPY requirements.txt /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    IOT23_DATA_DIR=/app/data/raw \
    IOT23_PROCESSED_DIR=/app/data/processed

# Create non-root user for security
RUN useradd -m -u 1000 hexdetector && \
    chown -R hexdetector:hexdetector /app

USER hexdetector

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command - show help
CMD ["python", "src/main.py", "--help"]
