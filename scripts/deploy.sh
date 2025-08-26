#!/bin/bash
set -e

echo "🚀 BCI-GPT Production Deployment"
echo "================================"

# Configuration
NAMESPACE="bci-gpt-prod"
IMAGE_TAG=${IMAGE_TAG:-"latest"}
DOCKER_IMAGE="bci-gpt:$IMAGE_TAG"

# Functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

check_dependencies() {
    log "Checking dependencies..."
    command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed."; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed."; exit 1; }
    log "Dependencies checked ✅"
}

build_docker_image() {
    log "Building Docker image..."
    docker build -f Dockerfile.production -t $DOCKER_IMAGE .
    log "Docker image built ✅"
}

create_namespace() {
    log "Creating namespace..."
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    log "Namespace ready ✅"
}

deploy_certificates() {
    log "Deploying SSL certificates..."
    kubectl apply -f security/ -n $NAMESPACE
    log "SSL certificates deployed ✅"
}

deploy_application() {
    log "Deploying application..."
    kubectl apply -f k8s/ -n $NAMESPACE
    log "Application deployed ✅"
}

wait_for_deployment() {
    log "Waiting for deployment to be ready..."
    kubectl rollout status deployment/bci-gpt -n $NAMESPACE --timeout=300s
    log "Deployment ready ✅"
}

setup_monitoring() {
    log "Setting up monitoring..."
    kubectl apply -f monitoring/ -n $NAMESPACE
    log "Monitoring configured ✅"
}

verify_deployment() {
    log "Verifying deployment..."
    kubectl get pods -n $NAMESPACE
    kubectl get services -n $NAMESPACE
    log "Deployment verified ✅"
}

# Main execution
main() {
    log "Starting BCI-GPT production deployment..."
    
    check_dependencies
    build_docker_image
    create_namespace
    deploy_certificates
    deploy_application
    wait_for_deployment
    setup_monitoring
    verify_deployment
    
    log "🎉 Deployment completed successfully!"
    log "Access your application at: https://api.bci-gpt.com"
}

# Error handling
trap 'log "❌ Deployment failed at line $LINENO"' ERR

# Run main function
main "$@"
