# This is an example Dockerfile that builds a minimal container for running LK Agents
# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Set Prometheus directory for multiprocess metrics
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
RUN mkdir -p ${PROMETHEUS_MULTIPROC_DIR}

# Download model files
RUN python fast-preresponse.py download-files || true

# Run the application
CMD ["python", "fast-preresponse.py", "dev"]