# Multi-stage build for BCI-GPT production deployment
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r bci && useradd -r -g bci -u 1000 bci

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY . .

# Install BCI-GPT in development mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/cache && \
    chown -R bci:bci /app

# Production stage
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production
ENV LOG_LEVEL=info

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    wget \
    libssl3 \
    libffi8 \
    libjpeg8 \
    libpng16-16 \
    libhdf5-103 \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r bci && useradd -r -g bci -u 1000 bci

# Set working directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder --chown=bci:bci /app /app

# Create additional directories for production
RUN mkdir -p /app/logs /app/data /app/models /app/cache /app/tmp && \
    chown -R bci:bci /app && \
    chmod +x /app/deployment/scripts/*.sh

# Switch to non-root user
USER bci

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/app/deployment/scripts/start.sh"]

# Labels for metadata
LABEL maintainer="Terragon Labs <info@terragonlabs.com>"
LABEL version="1.0.0"
LABEL description="BCI-GPT: Brain-Computer Interface GPT Inverse Simulator"
LABEL org.opencontainers.image.source="https://github.com/danieleschmidt/quantum-inspired-task-planner"
LABEL org.opencontainers.image.documentation="https://docs.terragonlabs.com/bci-gpt"
LABEL org.opencontainers.image.licenses="MIT"