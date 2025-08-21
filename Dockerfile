# Production-Ready BCI-GPT Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash bci-gpt
USER bci-gpt

# Copy requirements first for better caching
COPY --chown=bci-gpt:bci-gpt requirements*.txt ./
COPY --chown=bci-gpt:bci-gpt pyproject.toml ./

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=bci-gpt:bci-gpt . .

# Install application
RUN pip install --user -e .

# Add user site-packages to PATH
ENV PATH="/home/bci-gpt/.local/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Security configurations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import bci_gpt; print('Health OK')" || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "bci_gpt.deployment.server"]
