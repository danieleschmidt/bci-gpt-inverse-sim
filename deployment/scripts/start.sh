#!/bin/bash
set -e

# BCI-GPT Production Startup Script
echo "Starting BCI-GPT in production mode..."

# Set default values
export ENVIRONMENT=${ENVIRONMENT:-production}
export LOG_LEVEL=${LOG_LEVEL:-info}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-4}

# Create necessary directories
mkdir -p /app/logs /app/data /app/cache /app/tmp

# Set permissions
chmod -R 755 /app/logs /app/data /app/cache /app/tmp

# Initialize logging
echo "$(date): Starting BCI-GPT server" >> /app/logs/startup.log

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python -m bci_gpt.database.migrate
fi

# Run pre-startup checks
echo "Running pre-startup health checks..."
python -m bci_gpt.utils.health_check

# Download/verify models if needed
if [ "$AUTO_DOWNLOAD_MODELS" = "true" ]; then
    echo "Downloading/verifying models..."
    python -m bci_gpt.models.download_models
fi

# Set up monitoring
if [ "$ENABLE_METRICS" = "true" ]; then
    echo "Enabling metrics collection..."
    export PROMETHEUS_MULTIPROC_DIR=/app/tmp/prometheus
    mkdir -p $PROMETHEUS_MULTIPROC_DIR
fi

# Start the application based on service type
case "${SERVICE_TYPE:-api}" in
    "api")
        echo "Starting BCI-GPT API server..."
        exec uvicorn bci_gpt.api.main:app \
            --host $HOST \
            --port $PORT \
            --workers $WORKERS \
            --log-level $LOG_LEVEL \
            --access-log \
            --use-colors \
            --loop uvloop \
            --http httptools
        ;;
    "worker")
        echo "Starting BCI-GPT Celery worker..."
        exec celery -A bci_gpt.workers.celery_app worker \
            --loglevel=$LOG_LEVEL \
            --concurrency=4 \
            --max-tasks-per-child=1000 \
            --time-limit=3600 \
            --soft-time-limit=3300
        ;;
    "beat")
        echo "Starting BCI-GPT Celery beat scheduler..."
        exec celery -A bci_gpt.workers.celery_app beat \
            --loglevel=$LOG_LEVEL \
            --schedule=/app/data/celerybeat-schedule
        ;;
    "flower")
        echo "Starting Flower monitoring..."
        exec celery -A bci_gpt.workers.celery_app flower \
            --port=5555 \
            --basic_auth=${FLOWER_USER}:${FLOWER_PASS}
        ;;
    "cli")
        echo "Starting BCI-GPT CLI interface..."
        exec python -m bci_gpt.cli "$@"
        ;;
    *)
        echo "Unknown service type: $SERVICE_TYPE"
        echo "Available types: api, worker, beat, flower, cli"
        exit 1
        ;;
esac