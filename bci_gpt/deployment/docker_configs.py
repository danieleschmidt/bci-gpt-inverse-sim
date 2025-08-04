"""Docker and containerization configurations for BCI-GPT deployment."""

from typing import Dict, List, Optional
from pathlib import Path


class DockerConfigGenerator:
    """Generate Docker configurations for different deployment scenarios."""
    
    def __init__(self):
        """Initialize Docker config generator."""
        pass
    
    def generate_dockerfile(self,
                           base_image: str = "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
                           python_version: str = "3.9",
                           target_env: str = "production") -> str:
        """Generate Dockerfile for BCI-GPT.
        
        Args:
            base_image: Base Docker image
            python_version: Python version
            target_env: Target environment (development, production, edge)
            
        Returns:
            Dockerfile content as string
        """
        
        if target_env == "edge":
            # Lightweight image for edge deployment
            dockerfile = f"""
# BCI-GPT Edge Deployment
FROM python:{python_version}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    libfftw3-dev \\
    libhdf5-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-edge.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-edge.txt

# Copy application code
COPY bci_gpt/ ./bci_gpt/
COPY setup.py .
COPY README.md .

# Install BCI-GPT package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 bci && chown -R bci:bci /app
USER bci

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import bci_gpt; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "bci_gpt.cli", "info"]
"""
        
        elif target_env == "production":
            # Full production image
            dockerfile = f"""
# BCI-GPT Production Deployment
FROM {base_image}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    curl \\
    wget \\
    libfftw3-dev \\
    libhdf5-dev \\
    libeigen3-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV CUDA_HOME=/usr/local/cuda

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY bci_gpt/ ./bci_gpt/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Install BCI-GPT package
RUN pip install -e .

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs
RUN chmod 755 /app/models /app/data /app/logs

# Create non-root user
RUN useradd -m -u 1000 bci && chown -R bci:bci /app
USER bci

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "bci_gpt.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        else:  # development
            dockerfile = f"""
# BCI-GPT Development Environment
FROM {base_image}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    curl \\
    wget \\
    vim \\
    htop \\
    tmux \\
    libfftw3-dev \\
    libhdf5-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and development tools
RUN pip install jupyter jupyterlab ipywidgets

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Install BCI-GPT in development mode
RUN pip install -e .

# Expose ports for Jupyter and API
EXPOSE 8888 8000

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
"""
        
        return dockerfile.strip()
    
    def generate_docker_compose(self,
                               include_redis: bool = True,
                               include_monitoring: bool = True,
                               include_db: bool = False) -> str:
        """Generate docker-compose.yml for multi-service deployment.
        
        Args:
            include_redis: Whether to include Redis service
            include_monitoring: Whether to include monitoring services
            include_db: Whether to include database service
            
        Returns:
            Docker Compose configuration
        """
        
        compose_config = """version: '3.8'

services:
  bci-gpt:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: bci-gpt-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/app
      - BCI_GPT_ENV=production
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  bci-gpt-worker:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: bci-gpt-worker
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - PYTHONPATH=/app
      - BCI_GPT_ENV=production
      - WORKER_MODE=true
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ["python", "-m", "bci_gpt.workers.inference_worker"]
"""

        if include_redis:
            compose_config += """
  redis:
    image: redis:7-alpine
    container_name: bci-gpt-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
"""

        if include_monitoring:
            compose_config += """
  prometheus:
    image: prom/prometheus:latest
    container_name: bci-gpt-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: bci-gpt-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
"""

        if include_db:
            compose_config += """
  postgres:
    image: postgres:15-alpine
    container_name: bci-gpt-postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init:/docker-entrypoint-initdb.d
    environment:
      - POSTGRES_DB=bci_gpt
      - POSTGRES_USER=bci_user
      - POSTGRES_PASSWORD=bci_password
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U bci_user -d bci_gpt"]
      interval: 10s
      timeout: 5s
      retries: 5
"""

        # Add volumes section
        volumes = ["redis_data:", "prometheus_data:", "grafana_data:"]
        if include_db:
            volumes.append("postgres_data:")
        
        compose_config += f"""
volumes:
{chr(10).join([f"  {vol}" for vol in volumes])}

networks:
  default:
    driver: bridge
"""

        return compose_config
    
    def generate_kubernetes_deployment(self,
                                     namespace: str = "bci-gpt",
                                     replicas: int = 2,
                                     resource_limits: Dict[str, str] = None) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests.
        
        Args:
            namespace: Kubernetes namespace
            replicas: Number of replicas
            resource_limits: Resource limits dict
            
        Returns:
            Dictionary of Kubernetes manifest files
        """
        
        if resource_limits is None:
            resource_limits = {
                'cpu': '2',
                'memory': '4Gi',
                'nvidia.com/gpu': '1'
            }
        
        # Namespace
        namespace_yaml = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
"""
        
        # Deployment
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bci-gpt-api
  namespace: {namespace}
  labels:
    app: bci-gpt-api
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: bci-gpt-api
  template:
    metadata:
      labels:
        app: bci-gpt-api
    spec:
      containers:
      - name: bci-gpt
        image: bci-gpt:latest
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: BCI_GPT_ENV
          value: "production"
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "{resource_limits['cpu']}"
            memory: "{resource_limits['memory']}"
            nvidia.com/gpu: "{resource_limits.get('nvidia.com/gpu', '1')}"
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
          readOnly: true
        - name: data-volume
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: bci-gpt-models-pvc
      - name: data-volume
        persistentVolumeClaim:
          claimName: bci-gpt-data-pvc
      nodeSelector:
        accelerator: nvidia-tesla-v100
"""
        
        # Service
        service_yaml = f"""
apiVersion: v1
kind: Service
metadata:
  name: bci-gpt-service
  namespace: {namespace}
spec:
  selector:
    app: bci-gpt-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer
"""
        
        # Horizontal Pod Autoscaler
        hpa_yaml = f"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: bci-gpt-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bci-gpt-api
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
        
        # Persistent Volume Claims
        pvc_yaml = f"""
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: bci-gpt-models-pvc
  namespace: {namespace}
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: bci-gpt-data-pvc
  namespace: {namespace}
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard
"""
        
        # ConfigMap for application configuration
        configmap_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: bci-gpt-config
  namespace: {namespace}
data:
  config.yaml: |
    model:
      batch_size: 16
      max_sequence_length: 1000
      device: cuda
    
    cache:
      enabled: true
      redis_host: redis-service
      redis_port: 6379
      max_memory_mb: 500
    
    logging:
      level: INFO
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    monitoring:
      prometheus_port: 9090
      metrics_enabled: true
"""
        
        return {
            'namespace.yaml': namespace_yaml.strip(),
            'deployment.yaml': deployment_yaml.strip(),
            'service.yaml': service_yaml.strip(),
            'hpa.yaml': hpa_yaml.strip(),
            'pvc.yaml': pvc_yaml.strip(),
            'configmap.yaml': configmap_yaml.strip()
        }
    
    def generate_helm_chart(self,
                           chart_name: str = "bci-gpt",
                           chart_version: str = "0.1.0") -> Dict[str, str]:
        """Generate Helm chart for BCI-GPT deployment.
        
        Args:
            chart_name: Helm chart name
            chart_version: Chart version
            
        Returns:
            Dictionary of Helm chart files
        """
        
        # Chart.yaml
        chart_yaml = f"""
apiVersion: v2
name: {chart_name}
description: A Helm chart for BCI-GPT deployment
type: application
version: {chart_version}
appVersion: "1.0.0"
keywords:
  - bci
  - brain-computer-interface
  - machine-learning
  - pytorch
home: https://github.com/danieleschmidt/bci-gpt-inverse-sim
sources:
  - https://github.com/danieleschmidt/bci-gpt-inverse-sim
maintainers:
  - name: Daniel Schmidt
    email: daniel@terragonlabs.com
"""
        
        # values.yaml
        values_yaml = """
# Default values for bci-gpt
replicaCount: 2

image:
  repository: bci-gpt
  pullPolicy: IfNotPresent
  tag: "latest"

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext:
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  runAsNonRoot: true
  runAsUser: 1000

service:
  type: LoadBalancer
  port: 80
  targetPort: 8000

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: bci-gpt.local
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 2
    memory: 4Gi
    nvidia.com/gpu: 1
  requests:
    cpu: 1
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector:
  accelerator: nvidia-tesla-v100

tolerations: []

affinity: {}

persistence:
  models:
    enabled: true
    storageClass: "fast-ssd"
    accessMode: ReadOnlyMany
    size: 10Gi
  data:
    enabled: true
    storageClass: "standard"
    accessMode: ReadWriteMany
    size: 50Gi

redis:
  enabled: true
  auth:
    enabled: false
  master:
    resources:
      requests:
        memory: 256Mi
        cpu: 100m
      limits:
        memory: 512Mi
        cpu: 200m

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true

config:
  model:
    batch_size: 16
    max_sequence_length: 1000
    device: cuda
  cache:
    enabled: true
    max_memory_mb: 500
  logging:
    level: INFO
"""
        
        # Templates
        deployment_template = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "bci-gpt.fullname" . }}
  labels:
    {{- include "bci-gpt.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "bci-gpt.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      {{- with .Values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "bci-gpt.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "bci-gpt.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            - name: BCI_GPT_ENV
              value: "production"
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
          volumeMounts:
            - name: models-volume
              mountPath: /app/models
              readOnly: true
            - name: data-volume
              mountPath: /app/data
            - name: config
              mountPath: /app/config
      volumes:
        - name: models-volume
          persistentVolumeClaim:
            claimName: {{ include "bci-gpt.fullname" . }}-models
        - name: data-volume
          persistentVolumeClaim:
            claimName: {{ include "bci-gpt.fullname" . }}-data
        - name: config
          configMap:
            name: {{ include "bci-gpt.fullname" . }}-config
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
"""
        
        return {
            'Chart.yaml': chart_yaml.strip(),
            'values.yaml': values_yaml.strip(),
            'templates/deployment.yaml': deployment_template.strip()
        }
    
    def generate_requirements_files(self) -> Dict[str, str]:
        """Generate different requirements files for different environments."""
        
        # Edge deployment (minimal dependencies)
        requirements_edge = """
torch>=2.0.0
numpy>=1.21.0
scipy>=1.8.0
scikit-learn>=1.1.0
typer>=0.7.0
rich>=12.0.0
pydantic>=1.10.0
fastapi>=0.95.0
uvicorn>=0.20.0
"""
        
        # Production deployment
        requirements_prod = """
-r requirements.txt
gunicorn>=20.1.0
redis>=4.5.0
prometheus-client>=0.16.0
psutil>=5.9.0
structlog>=22.0.0
sentry-sdk>=1.20.0
"""
        
        return {
            'requirements-edge.txt': requirements_edge.strip(),
            'requirements-prod.txt': requirements_prod.strip()
        }