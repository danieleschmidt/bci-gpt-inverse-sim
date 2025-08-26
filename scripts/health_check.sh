#!/bin/bash

NAMESPACE="bci-gpt-prod"
ENDPOINT="https://api.bci-gpt.com"

echo "🏥 BCI-GPT Health Check"
echo "======================"

# Check Kubernetes resources
echo "Checking Kubernetes resources..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

# Check application health
echo "Checking application health..."
if curl -f -s "$ENDPOINT/health" > /dev/null; then
    echo "✅ Application is healthy"
else
    echo "❌ Application health check failed"
    exit 1
fi

# Check metrics
echo "Checking metrics..."
kubectl top pods -n $NAMESPACE

echo "🎉 Health check completed!"
