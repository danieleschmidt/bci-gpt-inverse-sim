# Multi-stage production Docker image for BCI-GPT
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r bci && useradd -r -g bci -d /app -s /bin/bash bci

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-timeout flake8 bandit safety mypy black isort

# Copy source code
COPY . .

# Change ownership to bci user
RUN chown -R bci:bci /app

# Switch to non-root user
USER bci

# Expose ports
EXPOSE 8000 8080

# Development entrypoint
CMD ["python", "-m", "bci_gpt.cli", "--help"]

# Production stage
FROM base as production

# Install production-only dependencies
RUN pip install gunicorn

# Copy only necessary files
COPY bci_gpt/ ./bci_gpt/
COPY config/ ./config/
COPY models/ ./models/
COPY run_quality_gates.py .
COPY setup.py .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache /app/tmp

# Change ownership to bci user
RUN chown -R bci:bci /app

# Switch to non-root user
USER bci

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from bci_gpt.utils.monitoring import get_health_checker; \
                   health = get_health_checker().get_health_status(); \
                   exit(0 if health.overall_status != 'critical' else 1)"

# Expose ports
EXPOSE 8000

# Production entrypoint
CMD ["python", "-m", "bci_gpt.api.server", "--host", "0.0.0.0", "--port", "8000"]