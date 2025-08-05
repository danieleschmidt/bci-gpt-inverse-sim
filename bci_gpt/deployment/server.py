"""
Production server implementation for BCI-GPT.

This module provides the actual production server that gets generated
by the deployment utilities and serves the BCI-GPT API.
"""

import asyncio
import logging
import signal
import sys
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import torch

# Prometheus metrics (optional)
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
    
    # Metrics
    REQUEST_COUNT = Counter('bci_gpt_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
    REQUEST_DURATION = Histogram('bci_gpt_request_duration_seconds', 'Request duration')
    ACTIVE_CONNECTIONS = Gauge('bci_gpt_active_connections', 'Active connections')
    MODEL_INFERENCE_TIME = Histogram('bci_gpt_inference_duration_seconds', 'Model inference time')
except ImportError:
    HAS_PROMETHEUS = False

# Import BCI-GPT components
from bci_gpt.core.models import BCIGPTModel
from bci_gpt.decoding.realtime_decoder import RealtimeDecoder
from bci_gpt.utils.monitoring import get_metrics_collector
from bci_gpt.utils.security import InputValidation, PrivacyProtection

# Configuration from environment
APP_NAME = os.getenv("BCI_GPT_APP_NAME", "bci-gpt")
APP_VERSION = os.getenv("BCI_GPT_VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
PORT = int(os.getenv("BCI_GPT_PORT", "8000"))
WORKERS = int(os.getenv("BCI_GPT_WORKERS", "4"))
LOG_LEVEL = os.getenv("BCI_GPT_LOG_LEVEL", "INFO")
HEALTH_CHECK_PATH = os.getenv("HEALTH_CHECK_PATH", "/health")
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"

# Global application state
app_state = {
    "model": None,
    "decoder": None,
    "start_time": time.time(),
    "healthy": False,
    "metrics_collector": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logging.info("Starting BCI-GPT production server...")
    
    try:
        # Initialize metrics collector
        if METRICS_ENABLED:
            app_state["metrics_collector"] = get_metrics_collector()
        
        # Load model (in production, this would load from persistent storage)
        logging.info("Loading BCI-GPT model...")
        try:
            app_state["model"] = BCIGPTModel()
            # In production, load trained weights:
            # app_state["model"].load_state_dict(torch.load("/app/models/bci_gpt_model.pt"))
            app_state["model"].eval()
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            # Continue without model for health checks
        
        # Initialize decoder
        try:
            app_state["decoder"] = RealtimeDecoder()
        except Exception as e:
            logging.error(f"Failed to initialize decoder: {e}")
        
        app_state["healthy"] = True
        logging.info("BCI-GPT server started successfully")
        
        yield
        
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        app_state["healthy"] = False
        raise
    finally:
        # Shutdown
        logging.info("Shutting down BCI-GPT server...")
        if app_state["metrics_collector"]:
            app_state["metrics_collector"].stop_collection()
        app_state["healthy"] = False


# Create FastAPI application
app = FastAPI(
    title="BCI-GPT API",
    description="Brain-Computer Interface GPT API for imagined speech decoding",
    version=APP_VERSION,
    lifespan=lifespan
)

# Add security middleware
allowed_hosts = ["*"] if ENVIRONMENT == "development" else [
    "bci-gpt.example.com", 
    "localhost",
    f"{APP_NAME}-{ENVIRONMENT}.example.com"
]

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=allowed_hosts
)

cors_origins = [
    f"https://{APP_NAME}.example.com",
    f"https://{APP_NAME}-{ENVIRONMENT}.example.com"
] if ENVIRONMENT == "production" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Request metrics middleware."""
    if not HAS_PROMETHEUS or not METRICS_ENABLED:
        return await call_next(request)
    
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        raise
    finally:
        ACTIVE_CONNECTIONS.dec()


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request logging middleware."""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logging.info(
        f"{request.method} {request.url.path} - "
        f"{response.status_code} - {duration:.3f}s - "
        f"{request.client.host if request.client else 'unknown'}"
    )
    
    return response


@app.get(HEALTH_CHECK_PATH)
async def health_check():
    """Health check endpoint."""
    if not app_state["healthy"]:
        raise HTTPException(status_code=503, detail="Service unhealthy")
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - app_state["start_time"],
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "model_loaded": app_state["model"] is not None,
        "decoder_ready": app_state["decoder"] is not None
    }


@app.get("/readiness")
async def readiness_check():
    """Readiness probe endpoint."""
    ready = (
        app_state["healthy"] and 
        app_state["model"] is not None and 
        app_state["decoder"] is not None
    )
    
    if not ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "ready",
        "timestamp": time.time(),
        "components": {
            "model": app_state["model"] is not None,
            "decoder": app_state["decoder"] is not None,
            "metrics_collector": app_state["metrics_collector"] is not None
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not HAS_PROMETHEUS or not METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not available")
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/decode")
async def decode_eeg(request: Dict[str, Any]):
    """Decode EEG data to text."""
    try:
        # Validate request
        if "eeg_data" not in request:
            raise HTTPException(status_code=400, detail="Missing eeg_data field")
        
        eeg_data = request["eeg_data"]
        if not eeg_data:
            raise HTTPException(status_code=400, detail="Empty eeg_data")
        
        # Convert and validate EEG data
        try:
            eeg_array = np.array(eeg_data, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid EEG data format: {e}")
        
        # Validate EEG data constraints
        try:
            InputValidation.validate_eeg_data(
                eeg_array,
                expected_channels=request.get("expected_channels"),
                max_duration_seconds=300.0  # 5 minute limit
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"EEG validation failed: {e}")
        
        # Apply privacy protection
        privacy_level = request.get("privacy_level", 0.1)
        if not 0.0 <= privacy_level <= 1.0:
            raise HTTPException(status_code=400, detail="privacy_level must be between 0.0 and 1.0")
        
        protected_eeg = PrivacyProtection.anonymize_eeg_data(
            eeg_array, 
            privacy_level=privacy_level
        )
        
        # Check if model is available
        if not app_state["model"]:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Measure inference time
        start_time = time.time()
        
        # Perform inference
        try:
            # Convert to tensor
            eeg_tensor = torch.from_numpy(protected_eeg).float()
            if eeg_tensor.dim() == 2:
                eeg_tensor = eeg_tensor.unsqueeze(0)  # Add batch dimension
            
            # Decode
            with torch.no_grad():
                decoded_text = app_state["model"].generate_text_from_eeg(
                    eeg_tensor, 
                    max_length=min(request.get("max_length", 50), 200)  # Limit max length
                )
            
            inference_time = time.time() - start_time
            
            # Record metrics
            if HAS_PROMETHEUS and METRICS_ENABLED:
                MODEL_INFERENCE_TIME.observe(inference_time)
            
            if app_state["metrics_collector"]:
                app_state["metrics_collector"].record_model_metrics(
                    inference_time_ms=inference_time * 1000,
                    confidence=0.8  # Would be computed by model in practice
                )
            
            # Prepare response
            result_text = decoded_text[0] if isinstance(decoded_text, list) else decoded_text
            
            return {
                "decoded_text": result_text,
                "inference_time_ms": inference_time * 1000,
                "privacy_protected": privacy_level > 0.0,
                "timestamp": time.time(),
                "model_version": APP_VERSION,
                "eeg_shape": eeg_array.shape
            }
            
        except Exception as e:
            logging.error(f"Model inference error: {e}")
            raise HTTPException(status_code=500, detail="Model inference failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected decoding error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/decode_stream")
async def decode_eeg_stream(request: Dict[str, Any]):
    """Decode streaming EEG data (placeholder for real-time streaming)."""
    # This would integrate with real-time streaming in production
    raise HTTPException(
        status_code=501, 
        detail="Real-time streaming endpoint not yet implemented"
    )


@app.get("/info")
async def server_info():
    """Server information endpoint."""
    device_info = "cpu"
    if torch.cuda.is_available():
        device_info = f"cuda:{torch.cuda.current_device()}"
    
    return {
        "name": "BCI-GPT API",
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "uptime": time.time() - app_state["start_time"],
        "healthy": app_state["healthy"],
        "device": device_info,
        "features": {
            "eeg_decoding": True,
            "privacy_protection": True,
            "metrics_collection": METRICS_ENABLED,
            "health_monitoring": True,
            "real_time_streaming": False  # Not yet implemented
        },
        "limits": {
            "max_eeg_duration_seconds": 300,
            "max_text_length": 200,
            "supported_channels": "1-64"
        }
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BCI-GPT API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": HEALTH_CHECK_PATH,
        "info": "/info"
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": time.time()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "status_code": 500,
                "timestamp": time.time()
            }
        }
    )


def setup_logging():
    """Setup production logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def setup_signal_handlers():
    """Setup graceful shutdown signal handlers."""
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main server entry point."""
    setup_logging()
    setup_signal_handlers()
    
    logging.info(f"Starting {APP_NAME} v{APP_VERSION} in {ENVIRONMENT} mode")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Configure uvicorn
    config = {
        "app": "bci_gpt.deployment.server:app",
        "host": "0.0.0.0",
        "port": PORT,
        "log_level": LOG_LEVEL.lower(),
        "access_log": True,
        "reload": False,
        "workers": 1,  # Use async single worker for better resource usage
    }
    
    # Production optimizations
    if ENVIRONMENT == "production":
        config.update({
            "loop": "uvloop",  # Faster event loop
            "http": "httptools",  # Faster HTTP parsing
            "reload": False,
            "debug": False,
        })
    
    # Run server
    uvicorn.run(**config)


if __name__ == "__main__":
    main()