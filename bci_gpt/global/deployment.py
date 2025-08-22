"""Global deployment infrastructure for BCI-GPT with multi-region support."""

import json
import yaml
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
from datetime import datetime
import hashlib
import subprocess
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    RESEARCH = "research"


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"
    ON_PREMISE = "on_premise"


@dataclass
class RegionConfig:
    """Configuration for a specific deployment region."""
    region_id: str
    region_name: str
    cloud_provider: CloudProvider
    compliance_requirements: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    timezone: str = "UTC"
    
    # Infrastructure settings
    kubernetes_cluster: Optional[str] = None
    container_registry: Optional[str] = None
    database_endpoint: Optional[str] = None
    storage_bucket: Optional[str] = None
    
    # Capacity settings
    min_replicas: int = 2
    max_replicas: int = 10
    cpu_limit: str = "2000m"
    memory_limit: str = "4Gi"
    
    # Network settings
    load_balancer_endpoint: Optional[str] = None
    cdn_endpoint: Optional[str] = None
    ssl_certificate_arn: Optional[str] = None
    
    # Monitoring
    monitoring_enabled: bool = True
    log_retention_days: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'region_id': self.region_id,
            'region_name': self.region_name,
            'cloud_provider': self.cloud_provider.value,
            'compliance_requirements': self.compliance_requirements,
            'languages': self.languages,
            'timezone': self.timezone,
            'kubernetes_cluster': self.kubernetes_cluster,
            'container_registry': self.container_registry,
            'database_endpoint': self.database_endpoint,
            'storage_bucket': self.storage_bucket,
            'min_replicas': self.min_replicas,
            'max_replicas': self.max_replicas,
            'cpu_limit': self.cpu_limit,
            'memory_limit': self.memory_limit,
            'load_balancer_endpoint': self.load_balancer_endpoint,
            'cdn_endpoint': self.cdn_endpoint,
            'ssl_certificate_arn': self.ssl_certificate_arn,
            'monitoring_enabled': self.monitoring_enabled,
            'log_retention_days': self.log_retention_days
        }


@dataclass
class GlobalDeploymentConfig:
    """Global deployment configuration."""
    application_name: str = "bci-gpt"
    version: str = "v1.0.0"
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    
    # Global settings
    enable_multi_region: bool = True
    enable_auto_scaling: bool = True
    enable_blue_green_deployment: bool = True
    enable_canary_deployment: bool = False
    
    # Regional configurations
    regions: Dict[str, RegionConfig] = field(default_factory=dict)
    primary_region: str = "us-east-1"
    fallback_regions: List[str] = field(default_factory=list)
    
    # Container settings
    container_image: str = "bci-gpt:latest"
    image_pull_policy: str = "Always"
    
    # Security
    enable_pod_security_policy: bool = True
    enable_network_policy: bool = True
    enable_rbac: bool = True
    
    # Global load balancing
    global_load_balancer_enabled: bool = True
    health_check_path: str = "/health"
    health_check_interval: int = 30
    
    # Deployment strategy
    rolling_update_max_surge: str = "25%"
    rolling_update_max_unavailable: str = "25%"
    deployment_timeout: int = 600  # seconds
    
    def add_region(self, region: RegionConfig):
        """Add a region configuration."""
        self.regions[region.region_id] = region
        logger.info(f"Added region: {region.region_id} ({region.region_name})")
    
    def get_region(self, region_id: str) -> Optional[RegionConfig]:
        """Get region configuration by ID."""
        return self.regions.get(region_id)
    
    def list_regions(self) -> List[str]:
        """List all configured regions."""
        return list(self.regions.keys())


class KubernetesManifestGenerator:
    """Generate Kubernetes manifests for global deployment."""
    
    def __init__(self, config: GlobalDeploymentConfig):
        self.config = config
        
    def generate_deployment_manifest(self, region: RegionConfig) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest for a region."""
        
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.config.application_name}-{region.region_id}",
                "namespace": "bci-gpt",
                "labels": {
                    "app": self.config.application_name,
                    "region": region.region_id,
                    "version": self.config.version,
                    "environment": self.config.environment.value
                }
            },
            "spec": {
                "replicas": region.min_replicas,
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxSurge": self.config.rolling_update_max_surge,
                        "maxUnavailable": self.config.rolling_update_max_unavailable
                    }
                },
                "selector": {
                    "matchLabels": {
                        "app": self.config.application_name,
                        "region": region.region_id
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.application_name,
                            "region": region.region_id,
                            "version": self.config.version
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": self.config.application_name,
                                "image": self.config.container_image,
                                "imagePullPolicy": self.config.image_pull_policy,
                                "ports": [
                                    {
                                        "containerPort": 8080,
                                        "name": "http"
                                    }
                                ],
                                "env": [
                                    {
                                        "name": "REGION_ID",
                                        "value": region.region_id
                                    },
                                    {
                                        "name": "ENVIRONMENT",
                                        "value": self.config.environment.value
                                    },
                                    {
                                        "name": "LANGUAGES",
                                        "value": ",".join(region.languages)
                                    },
                                    {
                                        "name": "TIMEZONE",
                                        "value": region.timezone
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "500m",
                                        "memory": "1Gi"
                                    },
                                    "limits": {
                                        "cpu": region.cpu_limit,
                                        "memory": region.memory_limit
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": self.config.health_check_path,
                                        "port": "http"
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": self.config.health_check_interval
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": self.config.health_check_path,
                                        "port": "http"
                                    },
                                    "initialDelaySeconds": 10,
                                    "periodSeconds": 5
                                }
                            }
                        ],
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 2000
                        } if self.config.enable_pod_security_policy else {}
                    }
                }
            }
        }
        
        return manifest
    
    def generate_service_manifest(self, region: RegionConfig) -> Dict[str, Any]:
        """Generate Kubernetes service manifest for a region."""
        
        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.config.application_name}-service-{region.region_id}",
                "namespace": "bci-gpt",
                "labels": {
                    "app": self.config.application_name,
                    "region": region.region_id
                }
            },
            "spec": {
                "selector": {
                    "app": self.config.application_name,
                    "region": region.region_id
                },
                "ports": [
                    {
                        "port": 80,
                        "targetPort": 8080,
                        "name": "http"
                    }
                ],
                "type": "ClusterIP"
            }
        }
        
        return manifest
    
    def generate_hpa_manifest(self, region: RegionConfig) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler manifest."""
        
        manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.config.application_name}-hpa-{region.region_id}",
                "namespace": "bci-gpt"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{self.config.application_name}-{region.region_id}"
                },
                "minReplicas": region.min_replicas,
                "maxReplicas": region.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
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
                ]
            }
        }
        
        return manifest
    
    def generate_ingress_manifest(self, region: RegionConfig) -> Dict[str, Any]:
        """Generate ingress manifest for external access."""
        
        manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.config.application_name}-ingress-{region.region_id}",
                "namespace": "bci-gpt",
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/rate-limit": "100",
                    "nginx.ingress.kubernetes.io/rate-limit-window": "1m"
                }
            },
            "spec": {
                "tls": [
                    {
                        "hosts": [f"{region.region_id}.bci-gpt.com"],
                        "secretName": f"bci-gpt-tls-{region.region_id}"
                    }
                ],
                "rules": [
                    {
                        "host": f"{region.region_id}.bci-gpt.com",
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"{self.config.application_name}-service-{region.region_id}",
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
        
        return manifest


class GlobalDeploymentManager:
    """Manage global BCI-GPT deployments across multiple regions."""
    
    def __init__(self, config: GlobalDeploymentConfig, output_dir: Path = Path("./deploy")):
        self.config = config
        self.output_dir = output_dir
        self.manifest_generator = KubernetesManifestGenerator(config)
        self.deployment_history = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all_manifests(self) -> Dict[str, Dict[str, Any]]:
        """Generate all Kubernetes manifests for global deployment."""
        
        logger.info("Generating Kubernetes manifests for all regions...")
        
        all_manifests = {}
        
        for region_id, region in self.config.regions.items():
            logger.info(f"Generating manifests for region: {region_id}")
            
            region_manifests = {
                'deployment': self.manifest_generator.generate_deployment_manifest(region),
                'service': self.manifest_generator.generate_service_manifest(region),
                'ingress': self.manifest_generator.generate_ingress_manifest(region)
            }
            
            if self.config.enable_auto_scaling:
                region_manifests['hpa'] = self.manifest_generator.generate_hpa_manifest(region)
            
            all_manifests[region_id] = region_manifests
            
            # Save manifests to files
            self._save_region_manifests(region_id, region_manifests)
        
        # Generate global manifests
        global_manifests = self._generate_global_manifests()
        all_manifests['global'] = global_manifests
        
        logger.info(f"Generated manifests for {len(self.config.regions)} regions")
        return all_manifests
    
    def _save_region_manifests(self, region_id: str, manifests: Dict[str, Any]):
        """Save region-specific manifests to files."""
        
        region_dir = self.output_dir / region_id
        region_dir.mkdir(exist_ok=True)
        
        for manifest_type, manifest in manifests.items():
            manifest_file = region_dir / f"{manifest_type}.yaml"
            
            with open(manifest_file, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
            
            logger.debug(f"Saved {manifest_type} manifest: {manifest_file}")
    
    def _generate_global_manifests(self) -> Dict[str, Any]:
        """Generate global-level manifests."""
        
        global_manifests = {}
        
        # Namespace manifest
        namespace_manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": "bci-gpt",
                "labels": {
                    "app": self.config.application_name,
                    "environment": self.config.environment.value
                }
            }
        }
        global_manifests['namespace'] = namespace_manifest
        
        # Global ConfigMap
        configmap_manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "bci-gpt-global-config",
                "namespace": "bci-gpt"
            },
            "data": {
                "application.name": self.config.application_name,
                "application.version": self.config.version,
                "environment": self.config.environment.value,
                "regions": ",".join(self.config.list_regions()),
                "primary.region": self.config.primary_region
            }
        }
        global_manifests['configmap'] = configmap_manifest
        
        # Save global manifests
        global_dir = self.output_dir / "global"
        global_dir.mkdir(exist_ok=True)
        
        for manifest_type, manifest in global_manifests.items():
            manifest_file = global_dir / f"{manifest_type}.yaml"
            
            with open(manifest_file, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
        
        return global_manifests
    
    def generate_deployment_script(self) -> Path:
        """Generate deployment script for all regions."""
        
        script_content = f"""#!/bin/bash
# BCI-GPT Global Deployment Script
# Generated on: {datetime.now().isoformat()}

set -e

echo "🚀 Starting BCI-GPT global deployment..."

# Deploy global manifests
echo "📋 Deploying global manifests..."
kubectl apply -f global/

# Wait for namespace to be ready
kubectl wait --for=condition=Ready namespace/bci-gpt --timeout=60s

# Deploy to each region
"""
        
        for region_id in self.config.list_regions():
            script_content += f"""
echo "🌍 Deploying to region: {region_id}..."
kubectl apply -f {region_id}/

# Wait for deployment to be ready
kubectl wait --for=condition=Available deployment/bci-gpt-{region_id} -n bci-gpt --timeout=300s
echo "✅ Region {region_id} deployment complete"
"""
        
        script_content += """
echo "🎉 Global deployment complete!"
echo "📊 Checking deployment status..."

# Show deployment status
kubectl get deployments -n bci-gpt
kubectl get services -n bci-gpt
kubectl get ingress -n bci-gpt

echo "🔍 Verifying health checks..."
for deployment in $(kubectl get deployments -n bci-gpt -o name); do
    kubectl rollout status $deployment -n bci-gpt
done

echo "✅ BCI-GPT is now globally deployed!"
"""
        
        script_path = self.output_dir / "deploy.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Generated deployment script: {script_path}")
        return script_path
    
    def validate_deployment(self, region_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate deployment configuration."""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'regions_validated': []
        }
        
        # Validate global configuration
        if not self.config.primary_region in self.config.regions:
            validation_results['errors'].append(f"Primary region '{self.config.primary_region}' not configured")
            validation_results['valid'] = False
        
        if self.config.enable_multi_region and len(self.config.regions) < 2:
            validation_results['warnings'].append("Multi-region enabled but only one region configured")
        
        # Validate specific region or all regions
        regions_to_validate = [region_id] if region_id else self.config.list_regions()
        
        for rid in regions_to_validate:
            region = self.config.get_region(rid)
            if not region:
                validation_results['errors'].append(f"Region '{rid}' not found")
                validation_results['valid'] = False
                continue
            
            region_validation = self._validate_region(region)
            validation_results['regions_validated'].append({
                'region_id': rid,
                'validation': region_validation
            })
            
            if not region_validation['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(region_validation['errors'])
            
            validation_results['warnings'].extend(region_validation['warnings'])
        
        return validation_results
    
    def _validate_region(self, region: RegionConfig) -> Dict[str, Any]:
        """Validate a specific region configuration."""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        if not region.kubernetes_cluster:
            validation['warnings'].append(f"No Kubernetes cluster specified for region {region.region_id}")
        
        if not region.container_registry:
            validation['warnings'].append(f"No container registry specified for region {region.region_id}")
        
        # Check resource limits
        try:
            cpu_value = int(region.cpu_limit.replace('m', ''))
            if cpu_value < 100:
                validation['warnings'].append(f"CPU limit very low for region {region.region_id}: {region.cpu_limit}")
        except ValueError:
            validation['errors'].append(f"Invalid CPU limit format for region {region.region_id}: {region.cpu_limit}")
            validation['valid'] = False
        
        # Check replica counts
        if region.min_replicas > region.max_replicas:
            validation['errors'].append(f"Min replicas > max replicas for region {region.region_id}")
            validation['valid'] = False
        
        if region.min_replicas < 1:
            validation['errors'].append(f"Min replicas must be at least 1 for region {region.region_id}")
            validation['valid'] = False
        
        return validation
    
    def export_configuration(self, export_path: Path) -> Path:
        """Export deployment configuration to file."""
        
        config_data = {
            'global_config': {
                'application_name': self.config.application_name,
                'version': self.config.version,
                'environment': self.config.environment.value,
                'enable_multi_region': self.config.enable_multi_region,
                'enable_auto_scaling': self.config.enable_auto_scaling,
                'primary_region': self.config.primary_region,
                'fallback_regions': self.config.fallback_regions
            },
            'regions': {
                region_id: region.to_dict() 
                for region_id, region in self.config.regions.items()
            },
            'deployment_metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': "1.0.0"
            }
        }
        
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Exported configuration to: {export_path}")
        return export_path


class MultiRegionOrchestrator:
    """Orchestrate multi-region deployments with traffic management."""
    
    def __init__(self, deployment_manager: GlobalDeploymentManager):
        self.deployment_manager = deployment_manager
        self.config = deployment_manager.config
        
    def plan_deployment_order(self) -> List[str]:
        """Plan the order of regional deployments."""
        
        # Start with primary region
        deployment_order = [self.config.primary_region]
        
        # Add other regions
        other_regions = [r for r in self.config.list_regions() if r != self.config.primary_region]
        deployment_order.extend(other_regions)
        
        logger.info(f"Planned deployment order: {deployment_order}")
        return deployment_order
    
    def generate_traffic_splitting_config(self) -> Dict[str, Any]:
        """Generate traffic splitting configuration for gradual rollout."""
        
        total_regions = len(self.config.regions)
        
        if total_regions == 1:
            return {self.config.primary_region: 100}
        
        # Start with 50% to primary region, distribute rest
        traffic_config = {self.config.primary_region: 50}
        
        remaining_percentage = 50
        other_regions = [r for r in self.config.list_regions() if r != self.config.primary_region]
        
        per_region_percentage = remaining_percentage // len(other_regions)
        leftover = remaining_percentage % len(other_regions)
        
        for i, region in enumerate(other_regions):
            percentage = per_region_percentage
            if i < leftover:
                percentage += 1
            traffic_config[region] = percentage
        
        logger.info(f"Traffic splitting configuration: {traffic_config}")
        return traffic_config
    
    def generate_rollback_plan(self) -> Dict[str, Any]:
        """Generate rollback plan for deployment failures."""
        
        rollback_plan = {
            'strategy': 'region_by_region',
            'primary_region_first': True,
            'rollback_triggers': [
                'error_rate > 5%',
                'response_time > 2s',
                'availability < 99%'
            ],
            'rollback_steps': []
        }
        
        # Generate rollback steps for each region
        for region_id in reversed(self.plan_deployment_order()):
            step = {
                'region': region_id,
                'action': 'rollback_to_previous_version',
                'verification': [
                    'check_health_endpoints',
                    'verify_error_rates',
                    'confirm_rollback_success'
                ]
            }
            rollback_plan['rollback_steps'].append(step)
        
        return rollback_plan


# Example usage and testing
if __name__ == "__main__":
    # Create global deployment configuration
    global_config = GlobalDeploymentConfig(
        application_name="bci-gpt",
        version="v1.0.0",
        environment=DeploymentEnvironment.PRODUCTION,
        enable_multi_region=True,
        enable_auto_scaling=True
    )
    
    # Add regions
    regions = [
        RegionConfig(
            region_id="us-east-1",
            region_name="US East (Virginia)",
            cloud_provider=CloudProvider.AWS,
            compliance_requirements=["HIPAA", "SOC2"],
            languages=["en", "es"],
            kubernetes_cluster="bci-gpt-us-east-1",
            min_replicas=3,
            max_replicas=15
        ),
        RegionConfig(
            region_id="eu-west-1", 
            region_name="Europe (Ireland)",
            cloud_provider=CloudProvider.AWS,
            compliance_requirements=["GDPR", "SOC2"],
            languages=["en", "de", "fr"],
            kubernetes_cluster="bci-gpt-eu-west-1",
            timezone="Europe/Dublin",
            min_replicas=2,
            max_replicas=10
        ),
        RegionConfig(
            region_id="ap-northeast-1",
            region_name="Asia Pacific (Tokyo)",
            cloud_provider=CloudProvider.AWS,
            compliance_requirements=["SOC2"],
            languages=["ja", "en"],
            kubernetes_cluster="bci-gpt-ap-northeast-1",
            timezone="Asia/Tokyo",
            min_replicas=2,
            max_replicas=8
        )
    ]
    
    for region in regions:
        global_config.add_region(region)
    
    # Create deployment manager
    deployment_manager = GlobalDeploymentManager(global_config)
    
    # Validate configuration
    validation = deployment_manager.validate_deployment()
    print(f"Configuration validation: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Generate manifests and deployment scripts
    print("\n🚀 Generating global deployment artifacts...")
    
    manifests = deployment_manager.generate_all_manifests()
    script_path = deployment_manager.generate_deployment_script()
    config_path = deployment_manager.export_configuration(Path("./deploy/bci-gpt-config.json"))
    
    print(f"✅ Generated manifests for {len(manifests) - 1} regions")
    print(f"✅ Generated deployment script: {script_path}")
    print(f"✅ Exported configuration: {config_path}")
    
    # Create orchestrator for advanced deployment planning
    orchestrator = MultiRegionOrchestrator(deployment_manager)
    
    deployment_order = orchestrator.plan_deployment_order()
    traffic_config = orchestrator.generate_traffic_splitting_config()
    rollback_plan = orchestrator.generate_rollback_plan()
    
    print(f"\n📋 Deployment order: {deployment_order}")
    print(f"🔀 Traffic splitting: {traffic_config}")
    print(f"🔙 Rollback plan: {len(rollback_plan['rollback_steps'])} steps")
    
    print("\n🌍 Global deployment infrastructure ready!")
    print("Deploy with: ./deploy/deploy.sh")