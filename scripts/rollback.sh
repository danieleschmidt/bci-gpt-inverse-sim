#!/bin/bash
set -e

echo "🔄 BCI-GPT Rollback"
echo "=================="

NAMESPACE="bci-gpt-prod"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

rollback_deployment() {
    log "Rolling back deployment..."
    kubectl rollout undo deployment/bci-gpt -n $NAMESPACE
    kubectl rollout status deployment/bci-gpt -n $NAMESPACE --timeout=300s
    log "Rollback completed ✅"
}

verify_rollback() {
    log "Verifying rollback..."
    kubectl get pods -n $NAMESPACE
    log "Rollback verified ✅"
}

main() {
    log "Starting rollback..."
    rollback_deployment
    verify_rollback
    log "🎉 Rollback completed successfully!"
}

main "$@"
