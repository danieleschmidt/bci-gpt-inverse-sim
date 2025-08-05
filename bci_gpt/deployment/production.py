"""
Production deployment utilities for BCI-GPT.

This module provides comprehensive production deployment capabilities including:
- Docker containerization
- Kubernetes deployment
- Load balancing and auto-scaling
- Health checks and monitoring
- Configuration management
- CI/CD pipeline integration
"""

import os
import json
import yaml
import logging
import subprocess
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for production deployment."""
    # Application settings
    app_name: str = "bci-gpt"
    app_version: str = "1.0.0"
    environment: str = "production"  # development, staging, production
    
    # Container settings
    base_image: str = "python:3.9-slim"
    port: int = 8000
    workers: int = 4
    memory_limit: str = "2Gi"
    cpu_limit: str = "1000m"
    
    # Security settings
    enable_https: bool = True
    security_context_user: int = 1000
    enable_rbac: bool = True
    
    # Scaling settings
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    
    # Storage settings
    enable_persistent_storage: bool = True
    storage_size: str = "10Gi"
    storage_class: str = "fast-ssd"
    
    # Monitoring settings
    enable_metrics: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Health check settings
    health_check_path: str = "/health"
    readiness_probe_delay: int = 30
    liveness_probe_delay: int = 60


class DockerBuilder:
    """Build Docker containers for BCI-GPT."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize Docker builder.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_dockerfile(self, output_path: Path) -> None:
        """Generate optimized Dockerfile.
        
        Args:
            output_path: Path to write Dockerfile
        """
        dockerfile_content = self._create_dockerfile_content()
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.logger.info(f"Generated Dockerfile at {output_path}")
    
    def _create_dockerfile_content(self) -> str:
        """Create Dockerfile content."""
        return f"""# Multi-stage Dockerfile for BCI-GPT Production Deployment
# Stage 1: Build environment
FROM {self.config.base_image} as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
COPY pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Run tests to ensure build quality
RUN python -m pytest tests/ -v --tb=short

# Stage 2: Production runtime
FROM {self.config.base_image} as runtime

# Create non-root user for security
RUN groupadd -r bci && useradd -r -g bci -u {self.config.security_context_user} bci

# Set working directory
WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY --from=builder --chown=bci:bci /app /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models \\
    && chown -R bci:bci /app

# Switch to non-root user
USER bci

# Expose application port
EXPOSE {self.config.port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{self.config.port}{self.config.health_check_path} || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV BCI_GPT_PORT={self.config.port}
ENV BCI_GPT_WORKERS={self.config.workers}
ENV BCI_GPT_LOG_LEVEL={self.config.log_level}

# Default command
CMD ["python", "-m", "bci_gpt.deployment.server", "--host", "0.0.0.0", "--port", "{self.config.port}", "--workers", "{self.config.workers}"]
"""

    def generate_dockerignore(self, output_path: Path) -> None:
        """Generate .dockerignore file.
        
        Args:
            output_path: Path to write .dockerignore
        """
        dockerignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation
docs/_build/

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Docker
Dockerfile*
.dockerignore

# Development
*.log
.env
.env.local

# Data files (too large for container)
data/
models/
*.h5
*.pt
*.pth
*.onnx

# Temporary files
tmp/
temp/
"""
        
        with open(output_path, 'w') as f:
            f.write(dockerignore_content)
        
        self.logger.info(f"Generated .dockerignore at {output_path}")
    
    def build_image(self, dockerfile_path: Path, tag: Optional[str] = None) -> str:
        """Build Docker image.
        
        Args:
            dockerfile_path: Path to Dockerfile
            tag: Optional image tag
            
        Returns:
            Built image tag
        """
        if tag is None:
            tag = f"{self.config.app_name}:{self.config.app_version}"
        
        build_context = dockerfile_path.parent
        
        cmd = [
            "docker", "build",
            "-t", tag,
            "-f", str(dockerfile_path),
            "--build-arg", f"VERSION={self.config.app_version}",
            "--build-arg", f"ENVIRONMENT={self.config.environment}",
            str(build_context)
        ]
        
        self.logger.info(f"Building Docker image: {tag}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info("Docker image built successfully")
            return tag
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Docker build failed: {e.stderr}")
            raise


class KubernetesDeployer:
    """Deploy BCI-GPT to Kubernetes."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize Kubernetes deployer.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_manifests(self, output_dir: Path) -> None:
        """Generate Kubernetes manifests.
        
        Args:
            output_dir: Directory to write manifests
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all manifest files
        self._generate_namespace(output_dir / "namespace.yaml")
        self._generate_configmap(output_dir / "configmap.yaml")
        self._generate_secret(output_dir / "secret.yaml")
        self._generate_deployment(output_dir / "deployment.yaml")
        self._generate_service(output_dir / "service.yaml")
        self._generate_ingress(output_dir / "ingress.yaml")
        self._generate_hpa(output_dir / "hpa.yaml")
        self._generate_pvc(output_dir / "pvc.yaml")
        self._generate_rbac(output_dir / "rbac.yaml")
        self._generate_network_policy(output_dir / "network-policy.yaml")
        
        self.logger.info(f"Generated Kubernetes manifests in {output_dir}")
    
    def _generate_namespace(self, output_path: Path) -> None:
        """Generate namespace manifest."""
        manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": f"bci-gpt-{self.config.environment}",
                "labels": {
                    "app": self.config.app_name,
                    "environment": self.config.environment,
                    "version": self.config.app_version
                }
            }
        }
        
        self._write_yaml(manifest, output_path)
    
    def _generate_configmap(self, output_path: Path) -> None:
        """Generate ConfigMap manifest."""
        manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.config.app_name}-config",
                "namespace": f"bci-gpt-{self.config.environment}"
            },
            "data": {
                "LOG_LEVEL": self.config.log_level,
                "WORKERS": str(self.config.workers),
                "PORT": str(self.config.port),
                "ENVIRONMENT": self.config.environment,
                "METRICS_ENABLED": str(self.config.enable_metrics).lower(),
                "HEALTH_CHECK_PATH": self.config.health_check_path
            }
        }
        
        self._write_yaml(manifest, output_path)
    
    def _generate_secret(self, output_path: Path) -> None:
        """Generate Secret manifest."""
        import base64
        
        # These would be provided by external secret management in production
        secrets = {
            "database-url": base64.b64encode(b"postgresql://user:pass@db:5432/bci").decode(),
            "jwt-secret": base64.b64encode(b"your-jwt-secret-key-here").decode(),
            "encryption-key": base64.b64encode(b"your-encryption-key-here").decode()
        }
        
        manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": f"{self.config.app_name}-secrets",
                "namespace": f"bci-gpt-{self.config.environment}"
            },
            "type": "Opaque",
            "data": secrets
        }
        
        self._write_yaml(manifest, output_path)
    
    def _generate_deployment(self, output_path: Path) -> None:
        """Generate Deployment manifest."""
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.config.app_name}-deployment",
                "namespace": f"bci-gpt-{self.config.environment}",
                "labels": {
                    "app": self.config.app_name,
                    "version": self.config.app_version
                }
            },
            "spec": {
                "replicas": self.config.min_replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.config.app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.app_name,
                            "version": self.config.app_version
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true" if self.config.enable_metrics else "false",
                            "prometheus.io/port": str(self.config.port),
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "serviceAccountName": f"{self.config.app_name}-serviceaccount",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": self.config.security_context_user,
                            "fsGroup": self.config.security_context_user
                        },
                        "containers": [
                            {
                                "name": self.config.app_name,
                                "image": f"{self.config.app_name}:{self.config.app_version}",
                                "imagePullPolicy": "Always",
                                "ports": [
                                    {
                                        "containerPort": self.config.port,
                                        "name": "http"
                                    }
                                ],
                                "env": [
                                    {
                                        "name": "POD_NAME",
                                        "valueFrom": {
                                            "fieldRef": {
                                                "fieldPath": "metadata.name"
                                            }
                                        }
                                    },
                                    {
                                        "name": "POD_NAMESPACE",
                                        "valueFrom": {
                                            "fieldRef": {
                                                "fieldPath": "metadata.namespace"
                                            }
                                        }
                                    }
                                ],
                                "envFrom": [
                                    {
                                        "configMapRef": {
                                            "name": f"{self.config.app_name}-config"
                                        }
                                    },
                                    {
                                        "secretRef": {
                                            "name": f"{self.config.app_name}-secrets"
                                        }
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "memory": self.config.memory_limit,
                                        "cpu": self.config.cpu_limit
                                    },
                                    "limits": {
                                        "memory": self.config.memory_limit,
                                        "cpu": self.config.cpu_limit
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": self.config.health_check_path,
                                        "port": "http"
                                    },
                                    "initialDelaySeconds": self.config.liveness_probe_delay,
                                    "periodSeconds": 30,
                                    "timeoutSeconds": 10,
                                    "failureThreshold": 3
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": self.config.health_check_path,
                                        "port": "http"
                                    },
                                    "initialDelaySeconds": self.config.readiness_probe_delay,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 3
                                },
                                "volumeMounts": [
                                    {
                                        "name": "data-storage",
                                        "mountPath": "/app/data"
                                    },
                                    {
                                        "name": "model-storage",
                                        "mountPath": "/app/models"
                                    }
                                ] if self.config.enable_persistent_storage else []
                            }
                        ],
                        "volumes": [
                            {
                                "name": "data-storage",
                                "persistentVolumeClaim": {
                                    "claimName": f"{self.config.app_name}-data-pvc"
                                }
                            },
                            {
                                "name": "model-storage",
                                "persistentVolumeClaim": {
                                    "claimName": f"{self.config.app_name}-models-pvc"
                                }
                            }
                        ] if self.config.enable_persistent_storage else []
                    }
                }
            }
        }
        
        self._write_yaml(manifest, output_path)
    
    def _generate_service(self, output_path: Path) -> None:
        """Generate Service manifest."""
        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.config.app_name}-service",
                "namespace": f"bci-gpt-{self.config.environment}",
                "labels": {
                    "app": self.config.app_name
                }
            },
            "spec": {
                "selector": {
                    "app": self.config.app_name
                },
                "ports": [
                    {
                        "name": "http",
                        "port": 80,
                        "targetPort": self.config.port,
                        "protocol": "TCP"
                    }
                ],
                "type": "ClusterIP"
            }
        }
        
        self._write_yaml(manifest, output_path)
    
    def _generate_ingress(self, output_path: Path) -> None:
        """Generate Ingress manifest."""
        manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.config.app_name}-ingress",
                "namespace": f"bci-gpt-{self.config.environment}",
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod" if self.config.enable_https else "",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true" if self.config.enable_https else "false",
                    "nginx.ingress.kubernetes.io/proxy-body-size": "10m",
                    "nginx.ingress.kubernetes.io/proxy-read-timeout": "300",
                    "nginx.ingress.kubernetes.io/proxy-connect-timeout": "300"
                }
            },
            "spec": {
                "tls": [
                    {
                        "hosts": [f"{self.config.app_name}-{self.config.environment}.example.com"],
                        "secretName": f"{self.config.app_name}-tls"
                    }
                ] if self.config.enable_https else [],
                "rules": [
                    {
                        "host": f"{self.config.app_name}-{self.config.environment}.example.com",
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"{self.config.app_name}-service",
                                            "port": {
                                                "number": 80
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        self._write_yaml(manifest, output_path)
    
    def _generate_hpa(self, output_path: Path) -> None:
        """Generate HorizontalPodAutoscaler manifest."""
        manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.config.app_name}-hpa",
                "namespace": f"bci-gpt-{self.config.environment}"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{self.config.app_name}-deployment"
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_cpu_utilization
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 50,
                                "periodSeconds": 60
                            }
                        ]
                    },
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 100,
                                "periodSeconds": 60
                            }
                        ]
                    }
                }
            }
        }
        
        self._write_yaml(manifest, output_path)
    
    def _generate_pvc(self, output_path: Path) -> None:
        """Generate PersistentVolumeClaim manifests."""
        if not self.config.enable_persistent_storage:
            return
        
        manifests = []
        
        # Data storage PVC
        data_pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.config.app_name}-data-pvc",
                "namespace": f"bci-gpt-{self.config.environment}"
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.config.storage_class,
                "resources": {
                    "requests": {
                        "storage": self.config.storage_size
                    }
                }
            }
        }
        manifests.append(data_pvc)
        
        # Models storage PVC
        models_pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.config.app_name}-models-pvc",
                "namespace": f"bci-gpt-{self.config.environment}"
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.config.storage_class,
                "resources": {
                    "requests": {
                        "storage": "50Gi"  # Models need more space
                    }
                }
            }
        }
        manifests.append(models_pvc)
        
        # Write all PVC manifests
        with open(output_path, 'w') as f:
            for i, manifest in enumerate(manifests):
                if i > 0:
                    f.write("---\n")
                yaml.dump(manifest, f, default_flow_style=False)
    
    def _generate_rbac(self, output_path: Path) -> None:
        """Generate RBAC manifests."""
        if not self.config.enable_rbac:
            return
        
        manifests = []
        
        # ServiceAccount
        service_account = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": f"{self.config.app_name}-serviceaccount",
                "namespace": f"bci-gpt-{self.config.environment}"
            }
        }
        manifests.append(service_account)
        
        # Role
        role = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {
                "name": f"{self.config.app_name}-role",
                "namespace": f"bci-gpt-{self.config.environment}"
            },
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["configmaps", "secrets"],
                    "verbs": ["get", "list", "watch"]
                },
                {
                    "apiGroups": [""],
                    "resources": ["pods"],
                    "verbs": ["get", "list"]
                }
            ]
        }
        manifests.append(role)
        
        # RoleBinding
        role_binding = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {
                "name": f"{self.config.app_name}-rolebinding",
                "namespace": f"bci-gpt-{self.config.environment}"
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": f"{self.config.app_name}-serviceaccount",
                    "namespace": f"bci-gpt-{self.config.environment}"
                }
            ],
            "roleRef": {
                "kind": "Role",
                "name": f"{self.config.app_name}-role",
                "apiGroup": "rbac.authorization.k8s.io"
            }
        }
        manifests.append(role_binding)
        
        # Write all RBAC manifests
        with open(output_path, 'w') as f:
            for i, manifest in enumerate(manifests):
                if i > 0:
                    f.write("---\n")
                yaml.dump(manifest, f, default_flow_style=False)
    
    def _generate_network_policy(self, output_path: Path) -> None:
        """Generate NetworkPolicy manifest."""
        manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{self.config.app_name}-network-policy",
                "namespace": f"bci-gpt-{self.config.environment}"
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": self.config.app_name
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "ingress-nginx"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": self.config.port
                            }
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 53
                            },
                            {
                                "protocol": "UDP",
                                "port": 53
                            }
                        ]
                    },
                    {
                        "to": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "database"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 5432
                            }
                        ]
                    }
                ]
            }
        }
        
        self._write_yaml(manifest, output_path)
    
    def _write_yaml(self, manifest: Dict[str, Any], output_path: Path) -> None:
        """Write YAML manifest to file."""
        with open(output_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
    
    def deploy(self, manifests_dir: Path, dry_run: bool = False) -> None:
        """Deploy to Kubernetes cluster.
        
        Args:
            manifests_dir: Directory containing manifest files
            dry_run: Whether to perform a dry run
        """
        kubectl_cmd = ["kubectl", "apply", "-f", str(manifests_dir)]
        
        if dry_run:
            kubectl_cmd.append("--dry-run=client")
        
        try:
            result = subprocess.run(kubectl_cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Kubernetes deployment {'simulated' if dry_run else 'completed'}")
            self.logger.info(result.stdout)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Kubernetes deployment failed: {e.stderr}")
            raise


class ProductionServer:
    """Production-ready server implementation."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize production server.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_server_code(self, output_path: Path) -> None:
        """Generate production server code.
        
        Args:
            output_path: Path to write server code
        """
        server_code = self._create_server_code()
        
        with open(output_path, 'w') as f:
            f.write(server_code)
        
        self.logger.info(f"Generated server code at {output_path}")
    
    def _create_server_code(self) -> str:
        """Create production server code."""
        return f'''"""
Production server for BCI-GPT.

This module provides a production-ready server implementation with:
- FastAPI web framework
- Health checks and metrics
- Graceful shutdown
- Request/response logging
- Error handling
- Security middleware
"""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import BCI-GPT components
from bci_gpt.core.models import BCIGPTModel
from bci_gpt.decoding.realtime_decoder import RealtimeDecoder
from bci_gpt.utils.monitoring import get_metrics_collector
from bci_gpt.utils.security import InputValidation, PrivacyProtection

# Metrics
REQUEST_COUNT = Counter('bci_gpt_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('bci_gpt_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('bci_gpt_active_connections', 'Active connections')
MODEL_INFERENCE_TIME = Histogram('bci_gpt_inference_duration_seconds', 'Model inference time')

# Global application state
app_state = {{
    "model": None,
    "decoder": None,
    "start_time": time.time(),
    "healthy": False,
    "metrics_collector": None
}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logging.info("Starting BCI-GPT production server...")
    
    try:
        # Initialize metrics collector
        app_state["metrics_collector"] = get_metrics_collector()
        
        # Load model (in production, this would load from storage)
        logging.info("Loading BCI-GPT model...")
        app_state["model"] = BCIGPTModel()
        
        # Initialize decoder
        app_state["decoder"] = RealtimeDecoder()
        
        app_state["healthy"] = True
        logging.info("BCI-GPT server started successfully")
        
        yield
        
    except Exception as e:
        logging.error(f"Failed to start server: {{e}}")
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
    version="{self.config.app_version}",
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] if {self.config.environment} == "development" else ["bci-gpt.example.com"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bci-gpt.example.com"] if {self.config.environment} == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Request metrics middleware."""
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
        f"{{request.method}} {{request.url.path}} - "
        f"{{response.status_code}} - {{duration:.3f}}s"
    )
    
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not app_state["healthy"]:
        raise HTTPException(status_code=503, detail="Service unhealthy")
    
    return {{
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - app_state["start_time"],
        "version": "{self.config.app_version}",
        "environment": "{self.config.environment}"
    }}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/decode")
async def decode_eeg(request: Dict[str, Any]):
    """Decode EEG data to text."""
    try:
        # Validate input
        if "eeg_data" not in request:
            raise HTTPException(status_code=400, detail="Missing eeg_data")
        
        eeg_data = request["eeg_data"]
        
        # Convert and validate
        import numpy as np
        eeg_array = np.array(eeg_data, dtype=np.float32)
        
        InputValidation.validate_eeg_data(eeg_array)
        
        # Apply privacy protection
        protected_eeg = PrivacyProtection.anonymize_eeg_data(
            eeg_array, 
            privacy_level=request.get("privacy_level", 0.1)
        )
        
        # Measure inference time
        start_time = time.time()
        
        # Perform inference
        if app_state["decoder"] and app_state["model"]:
            # Convert to tensor
            import torch
            eeg_tensor = torch.from_numpy(protected_eeg).float().unsqueeze(0)
            
            # Decode
            with torch.no_grad():
                decoded_text = app_state["model"].generate_text_from_eeg(
                    eeg_tensor, 
                    max_length=request.get("max_length", 50)
                )
            
            inference_time = time.time() - start_time
            MODEL_INFERENCE_TIME.observe(inference_time)
            
            # Record metrics
            if app_state["metrics_collector"]:
                app_state["metrics_collector"].record_model_metrics(
                    inference_time_ms=inference_time * 1000,
                    confidence=0.8  # Would be computed by model
                )
            
            return {{
                "decoded_text": decoded_text[0] if isinstance(decoded_text, list) else decoded_text,
                "inference_time_ms": inference_time * 1000,
                "privacy_protected": True,
                "timestamp": time.time()
            }}
        else:
            raise HTTPException(status_code=503, detail="Model not available")
            
    except Exception as e:
        logging.error(f"Decoding error: {{e}}")
        
        if isinstance(e, HTTPException):
            raise
        
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/info")
async def server_info():
    """Server information endpoint."""
    return {{
        "name": "BCI-GPT API",
        "version": "{self.config.app_version}",
        "environment": "{self.config.environment}",
        "uptime": time.time() - app_state["start_time"],
        "healthy": app_state["healthy"],
        "features": {{
            "real_time_decoding": True,
            "privacy_protection": True,
            "metrics_collection": True,
            "health_monitoring": True
        }}
    }}


def setup_logging():
    """Setup production logging."""
    logging.basicConfig(
        level=getattr(logging, "{self.config.log_level}"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def setup_signal_handlers():
    """Setup graceful shutdown signal handlers."""
    def signal_handler(signum, frame):
        logging.info(f"Received signal {{signum}}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main server entry point."""
    setup_logging()
    setup_signal_handlers()
    
    # Run server
    uvicorn.run(
        "bci_gpt.deployment.server:app",
        host="0.0.0.0",
        port={self.config.port},
        workers=1,  # Use single worker with async
        access_log=True,
        log_level="{self.config.log_level.lower()}",
        reload=False,
        loop="uvloop" if {self.config.environment} == "production" else "asyncio"
    )


if __name__ == "__main__":
    main()
'''


class CICDPipeline:
    """Generate CI/CD pipeline configurations."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize CI/CD pipeline generator.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_github_actions(self, output_path: Path) -> None:
        """Generate GitHub Actions workflow.
        
        Args:
            output_path: Path to write workflow file
        """
        workflow_content = f"""name: BCI-GPT CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{{{ github.repository }}}}

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{{{ matrix.python-version }}}}
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ matrix.python-version }}}}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{{{ runner.os }}}}-pip-${{{{ hashFiles('**/requirements*.txt') }}}}
        restore-keys: |
          ${{{{ runner.os }}}}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Lint with flake8
      run: |
        flake8 bci_gpt tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 bci_gpt tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Type check with mypy
      run: |
        mypy bci_gpt --ignore-missing-imports
    
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=bci_gpt --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run security scan
      uses: pypa/gh-action-pip-audit@v1.0.8
      with:
        inputs: requirements.txt requirements-dev.txt
    
    - name: Run Bandit security scan
      run: |
        pip install bandit
        bandit -r bci_gpt -f json -o bandit-report.json || true
    
    - name: Upload security scan results
      uses: actions/upload-artifact@v3
      with:
        name: security-scan-results
        path: bandit-report.json

  build-and-push:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{{{branch}}}}-
          type=raw,value=latest,enable={{{{is_default_branch}}}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          VERSION=${{{{ github.sha }}}}
          ENVIRONMENT={self.config.environment}

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'
    
    - name: Configure kubectl
      run: |
        echo "${{{{ secrets.KUBE_CONFIG_STAGING }}}}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to staging
      run: |
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/bci-gpt-deployment bci-gpt=${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}:develop -n bci-gpt-staging
        kubectl rollout status deployment/bci-gpt-deployment -n bci-gpt-staging

  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'
    
    - name: Configure kubectl
      run: |
        echo "${{{{ secrets.KUBE_CONFIG_PRODUCTION }}}}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to production
      run: |
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/bci-gpt-deployment bci-gpt=${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}:latest -n bci-gpt-production
        kubectl rollout status deployment/bci-gpt-deployment -n bci-gpt-production
    
    - name: Run post-deployment tests
      run: |
        # Add post-deployment health checks and smoke tests
        kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -- curl -f http://bci-gpt-service/health
"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(workflow_content)
        
        self.logger.info(f"Generated GitHub Actions workflow at {output_path}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create deployment configuration
    config = DeploymentConfig(
        app_name="bci-gpt",
        app_version="1.0.0",
        environment="production",
        min_replicas=3,
        max_replicas=10,
        enable_https=True,
        enable_metrics=True
    )
    
    print(f"Testing deployment utilities for {config.app_name} v{config.app_version}...")
    
    # Test Docker builder
    docker_builder = DockerBuilder(config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Generate Docker files
        docker_builder.generate_dockerfile(tmpdir / "Dockerfile")
        docker_builder.generate_dockerignore(tmpdir / ".dockerignore")
        
        print(f"Generated Docker files in {tmpdir}")
        
        # Test Kubernetes deployer
        k8s_deployer = KubernetesDeployer(config)
        k8s_dir = tmpdir / "k8s"
        k8s_deployer.generate_manifests(k8s_dir)
        
        print(f"Generated Kubernetes manifests in {k8s_dir}")
        
        # Test server generator
        server = ProductionServer(config)
        server.generate_server_code(tmpdir / "server.py")
        
        print(f"Generated server code")
        
        # Test CI/CD pipeline
        cicd = CICDPipeline(config)
        cicd.generate_github_actions(tmpdir / ".github" / "workflows" / "ci-cd.yml")
        
        print(f"Generated CI/CD pipeline")
        
        # List generated files
        print("\nGenerated files:")
        for file_path in tmpdir.rglob("*"):
            if file_path.is_file():
                print(f"  {file_path.relative_to(tmpdir)}")
    
    print("Production deployment utilities test completed!")