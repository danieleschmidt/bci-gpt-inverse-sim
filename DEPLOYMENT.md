# BCI-GPT Deployment Guide

This document provides comprehensive deployment instructions for BCI-GPT in various environments.

## 🚀 Quick Start

```bash
# Quality gates and testing
./deploy.sh test

# Local development setup
./deploy.sh local

# Docker deployment (staging)
./deploy.sh docker -e staging

# Kubernetes deployment (production)
./deploy.sh kubernetes -e production -v v1.0.0
```

## 🛠 Prerequisites

### General Requirements

- Python 3.11+
- Git
- Docker & Docker Compose (for containerized deployments)
- kubectl (for Kubernetes deployments)

### Hardware Requirements

| Environment | CPU | Memory | Storage | GPU (Optional) |
|-------------|-----|---------|---------|----------------|
| Development | 2 cores | 4GB | 20GB | - |
| Staging | 4 cores | 8GB | 50GB | - |
| Production | 8+ cores | 16GB+ | 100GB+ | NVIDIA GPU recommended |

### Dependencies

Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git curl wget

# macOS
brew install python git

# Install Python dependencies
pip install -r requirements.txt
```

## 🏗 Architecture Overview

BCI-GPT uses a microservices architecture with the following components:

- **BCI-GPT Core**: Main application with EEG processing and inverse simulation
- **Redis**: Caching and session storage
- **PostgreSQL**: Metadata and user management
- **Nginx**: Reverse proxy and load balancing
- **Monitoring Stack**: Prometheus, Grafana, ELK stack

## 📋 Deployment Options

### 1. Local Development

For development and testing:

```bash
./deploy.sh local
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Set up development configuration
- Enable hot-reloading and debugging

**Access:**
- Application: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

### 2. Docker Compose Deployment

For staging and testing environments:

```bash
# Development environment
./deploy.sh docker -e development

# Staging environment  
./deploy.sh docker -e staging
```

**Services:**
- BCI-GPT App: `http://localhost:8000`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- Kibana: `http://localhost:5601`

**Management:**
```bash
# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale bci-gpt=3

# Stop services
docker-compose down

# Update services
docker-compose pull && docker-compose up -d
```

### 3. Kubernetes Deployment

For production environments:

```bash
# Production deployment
./deploy.sh kubernetes -e production -v v1.0.0

# Staging deployment
./deploy.sh kubernetes -e staging
```

**Features:**
- Auto-scaling (2-10 replicas)
- Rolling updates
- Health checks
- Load balancing
- SSL termination
- Network policies

**Management:**
```bash
# Check deployment status
kubectl get all -n bci-gpt-production

# View logs
kubectl logs -n bci-gpt-production -l app=bci-gpt -f

# Scale manually
kubectl scale deployment bci-gpt-app --replicas=5 -n bci-gpt-production

# Update deployment
kubectl set image deployment/bci-gpt-app bci-gpt=bci-gpt:v1.1.0 -n bci-gpt-production
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `BCI_GPT_ENV` | Environment (development/staging/production) | `production` | No |
| `BCI_GPT_LOG_LEVEL` | Logging level | `INFO` | No |
| `BCI_GPT_DATA_DIR` | Data directory | `./data` | No |
| `BCI_GPT_MODEL_DIR` | Model directory | `./models` | No |
| `REDIS_URL` | Redis connection URL | - | Yes |
| `DATABASE_URL` | PostgreSQL connection URL | - | Yes |

### Configuration Files

- `config/development.yaml`: Development settings
- `config/staging.yaml`: Staging settings  
- `config/production.yaml`: Production settings

Example configuration:

```yaml
version: "1.0.0"
environment: "production"

model:
  hidden_size: 768
  num_layers: 12
  device: "auto"
  mixed_precision: true

security:
  encryption_enabled: true
  audit_logging: true
  max_file_size_mb: 50.0

monitoring:
  log_level: "INFO"
  metrics_enabled: true
  health_checks: true
```

## 🔐 Security

### Production Security Checklist

- [ ] Enable HTTPS/TLS encryption
- [ ] Configure proper authentication
- [ ] Enable audit logging
- [ ] Set up network policies
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] Access control (RBAC)
- [ ] Secret management

### TLS/SSL Configuration

For production deployments, configure TLS certificates:

```bash
# Using cert-manager (Kubernetes)
kubectl apply -f kubernetes/cert-manager.yaml

# Using Let's Encrypt
kubectl apply -f kubernetes/letsencrypt-issuer.yaml
```

### Secrets Management

Store sensitive configuration in Kubernetes secrets or external secret managers:

```bash
# Create secrets
kubectl create secret generic bci-gpt-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=redis-url="redis://..." \
  --from-literal=secret-key="..."
```

## 📊 Monitoring

### Health Checks

BCI-GPT provides several health check endpoints:

- `/health`: Overall application health
- `/ready`: Readiness for traffic
- `/metrics`: Prometheus metrics

### Monitoring Stack

The deployment includes comprehensive monitoring:

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **ELK Stack**: Centralized logging
- **Health Checker**: System health monitoring

### Key Metrics

- Request latency and throughput
- EEG processing performance
- Model inference times
- Memory and CPU usage
- Error rates and types

### Alerting

Configure alerts for critical metrics:

```yaml
# Prometheus alerting rules
groups:
  - name: bci-gpt
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency detected"
```

## 🧪 Quality Gates

Quality gates ensure deployment readiness:

```bash
# Run all quality gates
./deploy.sh test

# Specific quality gates
python run_quality_gates.py --output-dir ./reports
```

### Quality Gate Checks

1. **Code Quality**: Linting, static analysis
2. **Security Scan**: Vulnerability detection
3. **Unit Tests**: Component testing
4. **Integration Tests**: End-to-end testing
5. **Performance Tests**: Load and stress testing
6. **Memory Tests**: Leak detection
7. **Compliance**: Regulatory requirements
8. **System Health**: Infrastructure checks

### Quality Gate Thresholds

- Test coverage: ≥85%
- Performance: <100ms average response time
- Memory growth: <50MB over 10 operations
- Security: No critical vulnerabilities
- Uptime: >99.9% availability target

## 🔄 CI/CD Pipeline

### GitHub Actions

Example workflow for automated deployment:

```yaml
name: Deploy BCI-GPT

on:
  push:
    branches: [main]
    tags: [v*]

jobs:
  quality-gates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Quality Gates
        run: ./deploy.sh test

  deploy-staging:
    needs: quality-gates
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Staging
        run: ./deploy.sh kubernetes -e staging

  deploy-production:
    needs: quality-gates
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Production
        run: ./deploy.sh kubernetes -e production -v ${{ github.ref_name }}
```

### GitLab CI/CD

```yaml
stages:
  - test
  - build
  - deploy

quality-gates:
  stage: test
  script:
    - ./deploy.sh test

deploy-production:
  stage: deploy
  script:
    - ./deploy.sh kubernetes -e production -v $CI_COMMIT_TAG
  only:
    - tags
```

## 🆘 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Check Python path and dependencies
python -c "import bci_gpt; print('Import successful')"
pip install -e .
```

#### 2. Memory Issues

```bash
# Monitor memory usage
docker stats bci-gpt-app

# Kubernetes memory monitoring
kubectl top pods -n bci-gpt-production
```

#### 3. Performance Issues

```bash
# Check resource limits
kubectl describe pod -n bci-gpt-production

# Review performance metrics
curl http://localhost:8000/metrics
```

#### 4. Container Build Failures

```bash
# Check Docker logs
docker logs bci-gpt-app

# Rebuild with verbose output
docker build --no-cache -t bci-gpt .
```

### Debugging Commands

```bash
# Check application logs
kubectl logs -n bci-gpt-production -l app=bci-gpt --tail=100

# Get detailed pod information
kubectl describe pod <pod-name> -n bci-gpt-production

# Access pod shell
kubectl exec -it <pod-name> -n bci-gpt-production -- /bin/bash

# Port forward for debugging
kubectl port-forward svc/bci-gpt-service 8000:80 -n bci-gpt-production
```

### Performance Tuning

#### CPU Optimization

```yaml
# Kubernetes resource limits
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

#### Memory Optimization

```python
# Python memory optimization
import gc
gc.collect()  # Force garbage collection

# Reduce model precision
model = model.half()  # Use FP16
```

## 📚 Advanced Topics

### Multi-Region Deployment

Deploy across multiple regions for high availability:

```bash
# Deploy to multiple clusters
kubectl --context=us-west apply -f kubernetes/
kubectl --context=us-east apply -f kubernetes/
kubectl --context=eu-west apply -f kubernetes/
```

### Disaster Recovery

Implement backup and recovery procedures:

```bash
# Backup persistent volumes
kubectl create backup production-backup --include-resources="pv,pvc" --selector="app=bci-gpt"

# Database backup
pg_dump $DATABASE_URL > backup.sql
```

### Scaling Strategies

- **Horizontal Pod Autoscaling**: Automatic scaling based on CPU/memory
- **Vertical Pod Autoscaling**: Automatic resource allocation
- **Cluster Autoscaling**: Node-level scaling
- **Load Balancing**: Traffic distribution

### Security Hardening

- Enable Pod Security Standards
- Configure network policies
- Use security contexts
- Regular vulnerability scanning
- Secrets encryption at rest

## 📞 Support

### Getting Help

1. Check the [troubleshooting guide](#troubleshooting)
2. Review application logs
3. Search existing issues
4. Create a detailed issue report

### Issue Template

When reporting issues, please include:

- Deployment target (local/docker/kubernetes)
- Environment (development/staging/production)
- Error messages and logs
- Steps to reproduce
- System information

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and contribution guidelines.

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.