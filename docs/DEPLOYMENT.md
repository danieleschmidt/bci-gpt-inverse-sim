# BCI-GPT Production Deployment Guide

## Overview

This guide covers the complete production deployment of BCI-GPT using Kubernetes, Docker, and modern DevOps practices.

## Prerequisites

- Kubernetes cluster (1.21+)
- Docker (20.10+)
- kubectl configured
- Helm 3+ (optional)
- cert-manager (for SSL)

## Quick Start

```bash
# Clone repository
git clone https://github.com/danieleschmidt/bci-gpt-inverse-sim.git
cd bci-gpt-inverse-sim

# Deploy to production
./scripts/deploy.sh
```

## Architecture

### Components

- **Application**: BCI-GPT Python application
- **Database**: PostgreSQL for persistent storage
- **Cache**: Redis for caching
- **Load Balancer**: NGINX Ingress Controller
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack

### Scaling

- **Horizontal**: 3-10 replicas (auto-scaling)
- **Vertical**: Up to 2 CPU, 4Gi memory per pod
- **Database**: Read replicas for scaling reads

## Security

- TLS encryption (Let's Encrypt)
- Network policies
- RBAC enabled
- Security contexts
- Image scanning

## Monitoring

- Prometheus metrics collection
- Grafana dashboards
- Alertmanager notifications
- Health checks and probes

## Troubleshooting

### Common Issues

1. **Pod Not Starting**
   ```bash
   kubectl describe pod <pod-name> -n bci-gpt-prod
   kubectl logs <pod-name> -n bci-gpt-prod
   ```

2. **SSL Certificate Issues**
   ```bash
   kubectl describe certificate bci-gpt-tls -n bci-gpt-prod
   ```

3. **High Memory Usage**
   ```bash
   kubectl top pods -n bci-gpt-prod
   ```

### Rollback

```bash
./scripts/rollback.sh
```

## Maintenance

### Updates

1. Build new image with version tag
2. Update deployment with new image
3. Monitor deployment progress
4. Verify functionality

### Backup

- Database backups: Daily automated
- Configuration backups: Version controlled
- Persistent volumes: Snapshot enabled

## Performance Tuning

### Application Level

- Connection pooling
- Caching strategies
- Batch processing
- Async operations

### Infrastructure Level

- Resource limits optimization
- Node affinity rules
- Pod disruption budgets
- Network policies

## Support

For deployment issues, check:

1. Application logs: `kubectl logs -f deployment/bci-gpt -n bci-gpt-prod`
2. Events: `kubectl get events -n bci-gpt-prod --sort-by='.lastTimestamp'`
3. Resource usage: `kubectl top pods -n bci-gpt-prod`
4. Health endpoints: `https://api.bci-gpt.com/health`

## API Documentation

Base URL: `https://api.bci-gpt.com`

### Endpoints

- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics
- `POST /api/v1/decode` - EEG decoding
- `POST /api/v1/synthesize` - EEG synthesis

### Authentication

API key required in header: `Authorization: Bearer <api-key>`

### Rate Limits

- 100 requests per minute per IP
- 1000 requests per hour per API key

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `DB_HOST` | Database host | `localhost` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379` |
| `API_KEY_SECRET` | API key encryption | `required` |

## Disaster Recovery

### Backup Strategy

- **RTO**: 4 hours
- **RPO**: 1 hour
- **Backups**: Multiple regions
- **Testing**: Monthly DR drills

### Recovery Procedures

1. Assess damage and scope
2. Activate backup infrastructure
3. Restore from latest backup
4. Verify data integrity
5. Update DNS records
6. Monitor system health

---

*Last updated: {datetime.now().strftime("%Y-%m-%d")}*
