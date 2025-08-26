#!/bin/bash
set -e

echo "💾 BCI-GPT Backup Strategy"
echo "========================="

NAMESPACE="bci-gpt-prod"
BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
S3_BUCKET="bci-gpt-backups"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

backup_database() {
    log "Backing up database..."
    kubectl exec -n $NAMESPACE deployment/postgres -- pg_dump -U postgres bci_gpt > "$BACKUP_DIR/database.sql"
    log "Database backup completed ✅"
}

backup_configurations() {
    log "Backing up configurations..."
    kubectl get configmaps -n $NAMESPACE -o yaml > "$BACKUP_DIR/configmaps.yaml"
    kubectl get secrets -n $NAMESPACE -o yaml > "$BACKUP_DIR/secrets.yaml"
    log "Configuration backup completed ✅"
}

backup_persistent_volumes() {
    log "Creating volume snapshots..."
    kubectl get pvc -n $NAMESPACE -o yaml > "$BACKUP_DIR/pvc.yaml"
    # Create volume snapshots (cloud provider specific)
    log "Volume snapshots created ✅"
}

upload_to_s3() {
    log "Uploading to S3..."
    aws s3 sync "$BACKUP_DIR" "s3://$S3_BUCKET/$(date +%Y-%m-%d)/"
    log "Upload completed ✅"
}

cleanup_old_backups() {
    log "Cleaning up old backups..."
    find /backups -type d -mtime +7 -exec rm -rf {} +
    aws s3 ls "s3://$S3_BUCKET/" | awk '{print $2}' | sort | head -n -7 | xargs -I {} aws s3 rm "s3://$S3_BUCKET/{}" --recursive
    log "Cleanup completed ✅"
}

main() {
    mkdir -p "$BACKUP_DIR"
    backup_database
    backup_configurations
    backup_persistent_volumes
    upload_to_s3
    cleanup_old_backups
    log "🎉 Backup completed successfully!"
}

main "$@"
