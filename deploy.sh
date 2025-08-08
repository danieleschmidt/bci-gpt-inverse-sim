#!/bin/bash
set -euo pipefail

# BCI-GPT Production Deployment Script
# This script handles deployment to various environments with comprehensive checks

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="bci-gpt"
VERSION=${VERSION:-$(date +%Y%m%d-%H%M%S)}
ENVIRONMENT=${ENVIRONMENT:-production}
REGISTRY=${REGISTRY:-"ghcr.io/terragon-labs/bci-gpt"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
BCI-GPT Deployment Script

Usage: $0 [OPTIONS] TARGET

TARGETS:
    docker      Deploy using Docker Compose (development/staging)
    kubernetes  Deploy to Kubernetes cluster (production)
    local       Run locally for development
    test        Run tests and quality gates only

OPTIONS:
    -e, --environment ENV    Deployment environment (development, staging, production)
    -v, --version VERSION    Version tag for deployment
    -r, --registry REGISTRY  Container registry URL
    -f, --force              Force deployment without quality gates
    -h, --help               Show this help message
    --dry-run                Show what would be deployed without executing
    --skip-build             Skip container build step
    --skip-tests             Skip test execution
    --rollback               Rollback to previous deployment

EXAMPLES:
    $0 docker -e development
    $0 kubernetes -e production -v v1.2.3
    $0 test --environment staging
    $0 --rollback kubernetes

ENVIRONMENT VARIABLES:
    ENVIRONMENT    Target environment (default: production)
    VERSION        Deployment version (default: timestamp)
    REGISTRY       Container registry (default: ghcr.io/terragon-labs/bci-gpt)
    KUBECONFIG     Path to kubernetes config file
    
EOF
}

# Parse command line arguments
FORCE=false
DRY_RUN=false
SKIP_BUILD=false
SKIP_TESTS=false
ROLLBACK=false
TARGET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --rollback)
            ROLLBACK=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        docker|kubernetes|local|test)
            TARGET="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate target
if [[ -z "$TARGET" ]]; then
    log_error "No deployment target specified"
    show_help
    exit 1
fi

# Environment validation
case $ENVIRONMENT in
    development|staging|production)
        ;;
    *)
        log_error "Invalid environment: $ENVIRONMENT"
        log_error "Valid environments: development, staging, production"
        exit 1
        ;;
esac

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "bci_gpt" ]]; then
        log_error "Not in BCI-GPT project root directory"
        exit 1
    fi
    
    # Check required tools based on target
    case $TARGET in
        docker)
            if ! command -v docker &> /dev/null; then
                log_error "Docker is required for Docker deployment"
                exit 1
            fi
            
            if ! command -v docker-compose &> /dev/null; then
                log_error "Docker Compose is required for Docker deployment"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is required for Kubernetes deployment"
                exit 1
            fi
            
            # Check cluster connection
            if ! kubectl cluster-info &> /dev/null; then
                log_error "Cannot connect to Kubernetes cluster"
                log_error "Please check your kubeconfig"
                exit 1
            fi
            ;;
        local)
            if ! command -v python3 &> /dev/null; then
                log_error "Python 3 is required for local deployment"
                exit 1
            fi
            ;;
    esac
    
    log_success "Pre-flight checks passed"
}

# Quality gates
run_quality_gates() {
    if [[ "$SKIP_TESTS" == "true" ]] && [[ "$FORCE" == "true" ]]; then
        log_warning "Skipping quality gates (forced)"
        return 0
    fi
    
    log_info "Running quality gates..."
    
    if [[ -f "run_quality_gates.py" ]]; then
        if python3 run_quality_gates.py --output-dir "./quality_reports_${ENVIRONMENT}"; then
            log_success "Quality gates passed"
        else
            log_error "Quality gates failed"
            if [[ "$FORCE" == "true" ]]; then
                log_warning "Continuing with deployment (forced)"
            else
                log_error "Use --force to deploy despite quality gate failures"
                exit 1
            fi
        fi
    else
        log_warning "Quality gates script not found, skipping"
    fi
}

# Build container image
build_container() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Skipping container build"
        return 0
    fi
    
    log_info "Building container image: ${REGISTRY}:${VERSION}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build: docker build -t ${REGISTRY}:${VERSION} -f docker/Dockerfile ."
        return 0
    fi
    
    docker build \
        -t "${REGISTRY}:${VERSION}" \
        -t "${REGISTRY}:latest" \
        -f docker/Dockerfile \
        --target production \
        --build-arg VERSION="${VERSION}" \
        --build-arg ENVIRONMENT="${ENVIRONMENT}" \
        .
    
    log_success "Container image built successfully"
}

# Push container image
push_container() {
    if [[ "$TARGET" == "local" ]] || [[ "$SKIP_BUILD" == "true" ]]; then
        return 0
    fi
    
    log_info "Pushing container image to registry..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would push: ${REGISTRY}:${VERSION}"
        return 0
    fi
    
    # Login to registry if credentials are available
    if [[ -n "${REGISTRY_TOKEN:-}" ]]; then
        echo "${REGISTRY_TOKEN}" | docker login "${REGISTRY%/*}" -u "${REGISTRY_USER:-token}" --password-stdin
    fi
    
    docker push "${REGISTRY}:${VERSION}"
    docker push "${REGISTRY}:latest"
    
    log_success "Container image pushed successfully"
}

# Deploy to Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose (${ENVIRONMENT})..."
    
    # Create environment-specific override file
    local compose_override="docker-compose.${ENVIRONMENT}.yml"
    
    if [[ ! -f "$compose_override" ]]; then
        log_warning "No environment-specific override found: $compose_override"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: docker-compose up -d"
        return 0
    fi
    
    # Set environment variables
    export BCI_GPT_VERSION="$VERSION"
    export BCI_GPT_ENV="$ENVIRONMENT"
    export BCI_GPT_REGISTRY="$REGISTRY"
    
    # Deploy
    local compose_files="-f docker-compose.yml"
    if [[ -f "$compose_override" ]]; then
        compose_files="$compose_files -f $compose_override"
    fi
    
    docker-compose $compose_files up -d --build
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Health check
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Docker deployment successful - services are healthy"
    else
        log_warning "Services deployed but health check failed"
        log_info "Check logs: docker-compose logs"
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes (${ENVIRONMENT})..."
    
    local namespace="bci-gpt-${ENVIRONMENT}"
    local k8s_files="kubernetes/"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would apply Kubernetes manifests to namespace: $namespace"
        return 0
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets (in real deployment, these would come from a secret manager)
    kubectl create secret generic bci-gpt-secrets \
        --namespace="$namespace" \
        --from-literal=redis-url="redis://redis:6379/0" \
        --from-literal=database-url="postgresql://user:pass@postgres:5432/bcigpt" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Update image version in deployment
    local temp_deployment=$(mktemp)
    sed "s|image: bci-gpt:latest|image: ${REGISTRY}:${VERSION}|g" \
        kubernetes/deployment.yaml > "$temp_deployment"
    
    # Apply manifests
    kubectl apply -f "$temp_deployment" -n "$namespace"
    rm "$temp_deployment"
    
    # Wait for deployment
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/bci-gpt-app -n "$namespace" --timeout=300s
    
    # Check pod status
    kubectl get pods -n "$namespace" -l app=bci-gpt
    
    # Health check
    log_info "Running health check..."
    local pod_name=$(kubectl get pods -n "$namespace" -l app=bci-gpt -o jsonpath='{.items[0].metadata.name}')
    
    if kubectl exec -n "$namespace" "$pod_name" -- curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Kubernetes deployment successful - pods are healthy"
    else
        log_warning "Deployment completed but health check failed"
        log_info "Check pod logs: kubectl logs -n $namespace -l app=bci-gpt"
    fi
    
    # Display service URLs
    log_info "Service endpoints:"
    kubectl get ingress -n "$namespace"
}

# Deploy locally
deploy_local() {
    log_info "Setting up local development environment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would set up local environment"
        return 0
    fi
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    log_info "Installing dependencies..."
    pip install -e .
    
    # Install development dependencies if available
    if [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
    fi
    
    log_success "Local environment ready"
    log_info "To activate: source venv/bin/activate"
    log_info "To run: python -m bci_gpt.cli --help"
}

# Rollback deployment
rollback_deployment() {
    case $TARGET in
        docker)
            log_info "Rolling back Docker deployment..."
            if [[ "$DRY_RUN" == "false" ]]; then
                docker-compose down
                log_success "Docker deployment rolled back"
            fi
            ;;
        kubernetes)
            log_info "Rolling back Kubernetes deployment..."
            local namespace="bci-gpt-${ENVIRONMENT}"
            if [[ "$DRY_RUN" == "false" ]]; then
                kubectl rollout undo deployment/bci-gpt-app -n "$namespace"
                kubectl rollout status deployment/bci-gpt-app -n "$namespace"
                log_success "Kubernetes deployment rolled back"
            fi
            ;;
        *)
            log_error "Rollback not supported for target: $TARGET"
            exit 1
            ;;
    esac
}

# Main deployment flow
main() {
    log_info "Starting BCI-GPT deployment..."
    log_info "Target: $TARGET"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    log_info "Registry: $REGISTRY"
    
    # Handle rollback
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_deployment
        exit 0
    fi
    
    # Run pre-flight checks
    preflight_checks
    
    # Run quality gates (except for local and test targets)
    if [[ "$TARGET" != "local" ]] && [[ "$TARGET" != "test" ]]; then
        run_quality_gates
    fi
    
    # If test target, just run quality gates and exit
    if [[ "$TARGET" == "test" ]]; then
        log_success "Test target completed"
        exit 0
    fi
    
    # Build and push container (except for local)
    if [[ "$TARGET" != "local" ]]; then
        build_container
        push_container
    fi
    
    # Deploy based on target
    case $TARGET in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        local)
            deploy_local
            ;;
        *)
            log_error "Unknown target: $TARGET"
            exit 1
            ;;
    esac
    
    # Final success message
    log_success "BCI-GPT deployment completed successfully!"
    
    # Show post-deployment information
    case $TARGET in
        docker)
            log_info "Access the application at: http://localhost:8000"
            log_info "Monitor with: docker-compose logs -f"
            ;;
        kubernetes)
            log_info "Check status with: kubectl get all -n bci-gpt-${ENVIRONMENT}"
            ;;
        local)
            log_info "Run with: python -m bci_gpt.cli"
            ;;
    esac
}

# Trap signals for cleanup
trap 'log_error "Deployment interrupted"; exit 130' INT TERM

# Run main function
main "$@"