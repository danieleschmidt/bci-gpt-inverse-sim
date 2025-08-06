# BCI-GPT Production Deployment Guide

This guide covers the complete production deployment of the BCI-GPT Brain-Computer Interface system.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose v2+
- NVIDIA Docker Runtime (for GPU support)
- Kubernetes cluster (optional, for K8s deployment)
- Minimum 16GB RAM, 4 CPU cores, 1 GPU

### Environment Setup
```bash
# Clone and setup
git clone https://github.com/danieleschmidt/quantum-inspired-task-planner.git
cd quantum-inspired-task-planner

# Create environment file
cp deployment/.env.example deployment/.env
# Edit deployment/.env with your configuration

# Build and deploy
cd deployment
docker-compose -f docker-compose.prod.yml up -d
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   BCI-GPT API   │────│  BCI-GPT Worker │
│     (Nginx)     │    │   (FastAPI)     │    │    (Celery)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │      Redis      │    │   PostgreSQL    │
         │              │    (Cache)      │    │   (Database)    │
         │              └─────────────────┘    └─────────────────┘
         │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │    Logging      │    │     Security    │
│ (Prometheus)    │    │     (Loki)      │    │   (Encryption)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🐳 Docker Deployment

### 1. Build Images
```bash
# Build production image
docker build -f deployment/Dockerfile -t bci-gpt:latest .

# Or pull from registry
docker pull terragonlabs/bci-gpt:latest
```

### 2. Configuration
Edit `deployment/.env`:
```env
# Core Configuration
ENVIRONMENT=production
LOG_LEVEL=info
SECRET_KEY=your-super-secret-key-here

# Database
POSTGRES_PASSWORD=secure-postgres-password
REDIS_PASSWORD=secure-redis-password

# Security
ENABLE_SECURITY=true
ENCRYPTION_KEY=your-32-char-encryption-key

# Features
ENABLE_METRICS=true
ENABLE_TRACING=true
AUTO_DOWNLOAD_MODELS=true

# Monitoring
GRAFANA_PASSWORD=secure-grafana-password
```

### 3. Deploy Services
```bash
cd deployment

# Deploy full stack
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f bci-gpt-api
```

### 4. Health Checks
```bash
# API health
curl http://localhost/health

# Metrics endpoint
curl http://localhost/metrics

# Grafana dashboard
open http://localhost:3000
```

## ☸️ Kubernetes Deployment

### 1. Prerequisites
```bash
# Install kubectl, helm
kubectl version --client
helm version

# Create namespace
kubectl create namespace bci-gpt

# Create secrets
kubectl create secret generic bci-gpt-secrets \
  --from-literal=postgres-url="postgresql://user:pass@postgres:5432/bci_gpt" \
  --from-literal=redis-url="redis://:pass@redis:6379" \
  --from-literal=secret-key="your-secret-key" \
  -n bci-gpt
```

### 2. Storage Setup
```bash
# Apply persistent volume claims
kubectl apply -f deployment/kubernetes/storage.yaml
```

### 3. Deploy Application
```bash
# Deploy BCI-GPT services
kubectl apply -f deployment/kubernetes/bci-gpt-deployment.yaml

# Deploy monitoring stack
kubectl apply -f deployment/kubernetes/monitoring.yaml

# Check deployment status
kubectl get pods -n bci-gpt
kubectl get services -n bci-gpt
```

### 4. Configure Ingress
```bash
# Install nginx-ingress controller
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx --create-namespace

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Apply ingress configuration
kubectl apply -f deployment/kubernetes/ingress.yaml
```

## 🔒 Security Configuration

### 1. Enable Encryption
```yaml
# In docker-compose.prod.yml
environment:
  - ENABLE_ENCRYPTION=true
  - ENCRYPTION_KEY=${ENCRYPTION_KEY}
  - HTTPS_ONLY=true
```

### 2. Authentication Setup
```bash
# Generate API keys
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Configure OAuth (optional)
export OAUTH_CLIENT_ID=your-client-id
export OAUTH_CLIENT_SECRET=your-client-secret
```

### 3. Network Security
```nginx
# nginx.conf security headers
add_header X-Content-Type-Options nosniff;
add_header X-Frame-Options DENY;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
```

## 📊 Monitoring & Observability

### 1. Metrics Collection
- **Prometheus**: System and application metrics
- **Grafana**: Visualization dashboards
- **Custom Metrics**: EEG processing performance, model accuracy

### 2. Logging
- **Structured Logging**: JSON format with correlation IDs
- **Centralized**: Loki for log aggregation
- **Retention**: 30-day default retention policy

### 3. Distributed Tracing
- **Jaeger**: Request tracing across services
- **OpenTelemetry**: Instrumentation standards
- **Performance**: Latency and bottleneck identification

### 4. Alerting
```yaml
# Prometheus alerting rules
groups:
- name: bci-gpt
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: "High error rate detected"
```

## 🧪 Testing Production Deployment

### 1. Smoke Tests
```bash
# API endpoints
curl -f http://localhost/health
curl -f http://localhost/ready
curl -f http://localhost/info

# Authentication
curl -H "Authorization: Bearer $API_KEY" http://localhost/api/v1/status
```

### 2. Load Testing
```bash
# Install k6
snap install k6

# Run load tests
k6 run deployment/tests/load-test.js
```

### 3. Security Testing
```bash
# SSL/TLS validation
testssl.sh https://your-domain.com

# Vulnerability scanning
docker run --rm -v $(pwd):/zap/wrk/:rw \
  owasp/zap2docker-stable zap-baseline.py \
  -t https://your-domain.com
```

## 📈 Performance Optimization

### 1. Model Optimization
```python
# Enable quantization
ENABLE_MODEL_QUANTIZATION=true
QUANTIZATION_METHOD=dynamic

# GPU optimization
CUDA_VISIBLE_DEVICES=0
ENABLE_MIXED_PRECISION=true
```

### 2. Caching Strategy
```yaml
# Redis configuration
redis:
  maxmemory: 512mb
  maxmemory-policy: allkeys-lru
  save: "900 1 300 10 60 10000"
```

### 3. Database Tuning
```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
```

## 🔄 CI/CD Pipeline

### 1. GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build and Deploy
      run: |
        docker build -t bci-gpt:${{ github.sha }} .
        docker push registry.terragonlabs.com/bci-gpt:${{ github.sha }}
```

### 2. Blue-Green Deployment
```bash
# Deploy to blue environment
kubectl set image deployment/bci-gpt-api bci-gpt-api=bci-gpt:new-version -n bci-gpt-blue

# Validate deployment
kubectl rollout status deployment/bci-gpt-api -n bci-gpt-blue

# Switch traffic
kubectl patch service bci-gpt-api -p '{"spec":{"selector":{"version":"new-version"}}}' -n bci-gpt
```

## 🚨 Disaster Recovery

### 1. Backup Strategy
```bash
# Database backups
pg_dump -h postgres -U postgres bci_gpt > backup-$(date +%Y%m%d).sql

# Model backups
aws s3 sync /app/models s3://bci-gpt-models-backup/$(date +%Y%m%d)/
```

### 2. Recovery Procedures
```bash
# Database restore
psql -h postgres -U postgres bci_gpt < backup-20231201.sql

# Service restart
kubectl rollout restart deployment/bci-gpt-api -n bci-gpt
```

## 🔧 Troubleshooting

### Common Issues

#### 1. GPU Not Available
```bash
# Check GPU status
nvidia-smi
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi

# Fix: Install nvidia-docker2
sudo apt install nvidia-docker2
sudo systemctl restart docker
```

#### 2. Out of Memory
```bash
# Check memory usage
docker stats

# Fix: Increase memory limits
docker-compose -f docker-compose.prod.yml up -d \
  --scale bci-gpt-api=2 \
  --memory=4g
```

#### 3. Model Loading Failures
```bash
# Check model files
ls -la /app/models/
du -sh /app/models/*

# Fix: Re-download models
python -m bci_gpt.models.download_models --force
```

### Performance Debugging
```bash
# CPU profiling
docker exec -it bci-gpt-api py-spy top -p 1

# Memory profiling
docker exec -it bci-gpt-api memory_profiler python -m bci_gpt.api.main

# Network debugging
docker exec -it bci-gpt-api netstat -tulpn
```

## 📋 Maintenance Tasks

### Daily
- [ ] Check service health dashboard
- [ ] Review error logs and alerts
- [ ] Monitor resource usage

### Weekly
- [ ] Update security patches
- [ ] Run backup verification
- [ ] Performance review

### Monthly
- [ ] Security audit
- [ ] Capacity planning review
- [ ] Model performance analysis
- [ ] Dependencies update

## 🆘 Support & Contact

- **Documentation**: https://docs.terragonlabs.com/bci-gpt
- **Issues**: https://github.com/danieleschmidt/quantum-inspired-task-planner/issues
- **Security**: security@terragonlabs.com
- **Support**: support@terragonlabs.com

## 📄 License

This deployment guide is part of the BCI-GPT project, licensed under the MIT License.

---

**⚠️ Security Notice**: Always use HTTPS in production, rotate secrets regularly, and keep all components updated with the latest security patches.