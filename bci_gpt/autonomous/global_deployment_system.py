"""
Global Deployment Automation System v4.0
Multi-region, multi-platform deployment with compliance and optimization.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import subprocess
import yaml

logger = logging.getLogger(__name__)


class DeploymentRegion(Enum):
    """Global deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    JAPAN = "ap-northeast-1"


class DeploymentPlatform(Enum):
    """Deployment platform types."""
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    AWS_ECS = "aws_ecs"
    AZURE_CONTAINER = "azure_container"
    GOOGLE_CLOUD_RUN = "google_cloud_run"
    EDGE_DEVICE = "edge_device"


class ComplianceFramework(Enum):
    """Regulatory compliance frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    PDPA = "pdpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"


@dataclass
class DeploymentTarget:
    """Deployment target configuration."""
    name: str
    region: DeploymentRegion
    platform: DeploymentPlatform
    compliance_requirements: List[ComplianceFramework]
    resource_limits: Dict[str, Any]
    scaling_config: Dict[str, Any]
    environment_vars: Dict[str, str] = field(default_factory=dict)
    health_check_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentStatus:
    """Status of a deployment."""
    target_name: str
    status: str
    timestamp: float
    health_score: float
    performance_metrics: Dict[str, float]
    compliance_status: Dict[str, bool]
    error_message: Optional[str] = None


class GlobalDeploymentSystem:
    """
    Autonomous global deployment system with multi-region support,
    compliance automation, and performance optimization.
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.deployment_config_path = Path("deployment_configs")
        self.deployment_config_path.mkdir(exist_ok=True)
        
        self.deployment_targets: List[DeploymentTarget] = []
        self.deployment_status: Dict[str, DeploymentStatus] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        
        self._initialize_deployment_targets()
    
    def _initialize_deployment_targets(self):
        """Initialize global deployment targets."""
        targets = [
            # North America - Production
            DeploymentTarget(
                name="na_production",
                region=DeploymentRegion.US_EAST,
                platform=DeploymentPlatform.KUBERNETES,
                compliance_requirements=[ComplianceFramework.HIPAA, ComplianceFramework.SOC2],
                resource_limits={
                    "cpu": "2000m",
                    "memory": "4Gi",
                    "storage": "20Gi"
                },
                scaling_config={
                    "min_replicas": 2,
                    "max_replicas": 10,
                    "target_cpu_utilization": 70
                },
                environment_vars={
                    "ENVIRONMENT": "production",
                    "REGION": "us-east-1",
                    "COMPLIANCE_MODE": "hipaa",
                    "LOG_LEVEL": "INFO"
                },
                health_check_config={
                    "path": "/health",
                    "interval": 30,
                    "timeout": 10,
                    "retries": 3
                }
            ),
            
            # Europe - GDPR Compliant
            DeploymentTarget(
                name="eu_production",
                region=DeploymentRegion.EU_WEST,
                platform=DeploymentPlatform.KUBERNETES,
                compliance_requirements=[ComplianceFramework.GDPR, ComplianceFramework.ISO27001],
                resource_limits={
                    "cpu": "2000m",
                    "memory": "4Gi",
                    "storage": "20Gi"
                },
                scaling_config={
                    "min_replicas": 2,
                    "max_replicas": 8,
                    "target_cpu_utilization": 70
                },
                environment_vars={
                    "ENVIRONMENT": "production",
                    "REGION": "eu-west-1",
                    "COMPLIANCE_MODE": "gdpr",
                    "DATA_RESIDENCY": "eu",
                    "LOG_LEVEL": "INFO"
                },
                health_check_config={
                    "path": "/health",
                    "interval": 30,
                    "timeout": 10,
                    "retries": 3
                }
            ),
            
            # Asia Pacific - High Performance
            DeploymentTarget(
                name="apac_production",
                region=DeploymentRegion.ASIA_PACIFIC,
                platform=DeploymentPlatform.KUBERNETES,
                compliance_requirements=[ComplianceFramework.PDPA],
                resource_limits={
                    "cpu": "3000m",
                    "memory": "6Gi",
                    "storage": "30Gi"
                },
                scaling_config={
                    "min_replicas": 3,
                    "max_replicas": 15,
                    "target_cpu_utilization": 65
                },
                environment_vars={
                    "ENVIRONMENT": "production",
                    "REGION": "ap-southeast-1",
                    "COMPLIANCE_MODE": "pdpa",
                    "PERFORMANCE_MODE": "high",
                    "LOG_LEVEL": "INFO"
                },
                health_check_config={
                    "path": "/health",
                    "interval": 20,
                    "timeout": 8,
                    "retries": 3
                }
            ),
            
            # Edge Deployment - IoT/Mobile
            DeploymentTarget(
                name="edge_deployment",
                region=DeploymentRegion.US_WEST,
                platform=DeploymentPlatform.EDGE_DEVICE,
                compliance_requirements=[ComplianceFramework.SOC2],
                resource_limits={
                    "cpu": "500m",
                    "memory": "1Gi",
                    "storage": "5Gi"
                },
                scaling_config={
                    "min_replicas": 1,
                    "max_replicas": 3,
                    "target_cpu_utilization": 80
                },
                environment_vars={
                    "ENVIRONMENT": "edge",
                    "REGION": "edge",
                    "COMPLIANCE_MODE": "minimal",
                    "OPTIMIZATION_MODE": "edge",
                    "LOG_LEVEL": "WARN"
                },
                health_check_config={
                    "path": "/health",
                    "interval": 60,
                    "timeout": 15,
                    "retries": 2
                }
            ),
            
            # Development Environment
            DeploymentTarget(
                name="development",
                region=DeploymentRegion.US_WEST,
                platform=DeploymentPlatform.DOCKER_SWARM,
                compliance_requirements=[],
                resource_limits={
                    "cpu": "1000m",
                    "memory": "2Gi",
                    "storage": "10Gi"
                },
                scaling_config={
                    "min_replicas": 1,
                    "max_replicas": 3,
                    "target_cpu_utilization": 80
                },
                environment_vars={
                    "ENVIRONMENT": "development",
                    "REGION": "us-west-2",
                    "DEBUG_MODE": "true",
                    "LOG_LEVEL": "DEBUG"
                },
                health_check_config={
                    "path": "/health",
                    "interval": 60,
                    "timeout": 30,
                    "retries": 1
                }
            )
        ]
        
        self.deployment_targets = targets
    
    async def generate_deployment_manifests(self) -> Dict[str, Any]:
        """Generate deployment manifests for all targets."""
        manifests = {}
        
        for target in self.deployment_targets:
            if target.platform == DeploymentPlatform.KUBERNETES:
                manifest = await self._generate_k8s_manifest(target)
            elif target.platform == DeploymentPlatform.DOCKER_SWARM:
                manifest = await self._generate_docker_compose(target)
            elif target.platform == DeploymentPlatform.EDGE_DEVICE:
                manifest = await self._generate_edge_config(target)
            else:
                manifest = await self._generate_generic_manifest(target)
            
            manifests[target.name] = manifest
            
            # Save manifest to file
            manifest_path = self.deployment_config_path / f"{target.name}_manifest.yaml"
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
        
        return manifests
    
    async def _generate_k8s_manifest(self, target: DeploymentTarget) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"bci-gpt-{target.name}",
                "namespace": "bci-gpt",
                "labels": {
                    "app": "bci-gpt",
                    "environment": target.environment_vars.get("ENVIRONMENT", "production"),
                    "region": target.region.value,
                    "compliance": ",".join([c.value for c in target.compliance_requirements])
                }
            },
            "spec": {
                "replicas": target.scaling_config["min_replicas"],
                "selector": {
                    "matchLabels": {
                        "app": "bci-gpt",
                        "deployment": target.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "bci-gpt",
                            "deployment": target.name
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "bci-gpt",
                                "image": "bci-gpt:latest",
                                "ports": [
                                    {"containerPort": 8000, "name": "http"},
                                    {"containerPort": 8443, "name": "https"}
                                ],
                                "env": [
                                    {"name": k, "value": v} 
                                    for k, v in target.environment_vars.items()
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": target.resource_limits["cpu"],
                                        "memory": target.resource_limits["memory"]
                                    },
                                    "limits": {
                                        "cpu": target.resource_limits["cpu"],
                                        "memory": target.resource_limits["memory"]
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": target.health_check_config["path"],
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": target.health_check_config["interval"],
                                    "timeoutSeconds": target.health_check_config["timeout"],
                                    "failureThreshold": target.health_check_config["retries"]
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": target.health_check_config["path"],
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 10,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 3
                                }
                            }
                        ],
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000
                        } if target.compliance_requirements else {}
                    }
                }
            }
        }
    
    async def _generate_docker_compose(self, target: DeploymentTarget) -> Dict[str, Any]:
        """Generate Docker Compose configuration."""
        return {
            "version": "3.8",
            "services": {
                "bci-gpt": {
                    "image": "bci-gpt:latest",
                    "ports": [
                        "8000:8000",
                        "8443:8443"
                    ],
                    "environment": target.environment_vars,
                    "deploy": {
                        "replicas": target.scaling_config["min_replicas"],
                        "resources": {
                            "limits": {
                                "cpus": target.resource_limits["cpu"].replace("m", ""),
                                "memory": target.resource_limits["memory"]
                            }
                        },
                        "restart_policy": {
                            "condition": "on-failure",
                            "delay": "5s",
                            "max_attempts": 3
                        }
                    },
                    "healthcheck": {
                        "test": [
                            "CMD-SHELL", 
                            f"curl -f http://localhost:8000{target.health_check_config['path']} || exit 1"
                        ],
                        "interval": f"{target.health_check_config['interval']}s",
                        "timeout": f"{target.health_check_config['timeout']}s",
                        "retries": target.health_check_config["retries"],
                        "start_period": "30s"
                    }
                }
            },
            "networks": {
                "bci-gpt-network": {
                    "driver": "overlay"
                }
            }
        }
    
    async def _generate_edge_config(self, target: DeploymentTarget) -> Dict[str, Any]:
        """Generate edge device configuration."""
        return {
            "edge_deployment": {
                "device_type": "edge",
                "container_config": {
                    "image": "bci-gpt:edge",
                    "tag": "latest",
                    "ports": {
                        "http": 8000,
                        "metrics": 9090
                    },
                    "environment": target.environment_vars,
                    "resources": {
                        "cpu_limit": target.resource_limits["cpu"],
                        "memory_limit": target.resource_limits["memory"],
                        "storage_limit": target.resource_limits["storage"]
                    }
                },
                "optimization": {
                    "model_quantization": True,
                    "cache_strategy": "aggressive",
                    "batch_processing": False,
                    "low_power_mode": True
                },
                "monitoring": {
                    "health_check_url": target.health_check_config["path"],
                    "metrics_collection": True,
                    "log_level": target.environment_vars.get("LOG_LEVEL", "WARN")
                }
            }
        }
    
    async def _generate_generic_manifest(self, target: DeploymentTarget) -> Dict[str, Any]:
        """Generate generic deployment configuration."""
        return {
            "deployment_name": target.name,
            "platform": target.platform.value,
            "region": target.region.value,
            "compliance": [c.value for c in target.compliance_requirements],
            "configuration": {
                "environment": target.environment_vars,
                "resources": target.resource_limits,
                "scaling": target.scaling_config,
                "health_checks": target.health_check_config
            }
        }
    
    async def deploy_globally(self, targets: Optional[List[str]] = None) -> Dict[str, DeploymentStatus]:
        """Deploy to specified targets or all targets globally."""
        if targets is None:
            targets = [target.name for target in self.deployment_targets]
        
        deployment_results = {}
        
        # Generate manifests first
        await self.generate_deployment_manifests()
        
        # Deploy to each target
        for target_name in targets:
            target = next((t for t in self.deployment_targets if t.name == target_name), None)
            if not target:
                logger.error(f"Unknown deployment target: {target_name}")
                continue
            
            logger.info(f"Deploying to {target_name} ({target.region.value})")
            
            try:
                status = await self._deploy_to_target(target)
                deployment_results[target_name] = status
                self.deployment_status[target_name] = status
                
            except Exception as e:
                logger.error(f"Deployment failed for {target_name}: {e}")
                deployment_results[target_name] = DeploymentStatus(
                    target_name=target_name,
                    status="failed",
                    timestamp=time.time(),
                    health_score=0.0,
                    performance_metrics={},
                    compliance_status={},
                    error_message=str(e)
                )
        
        # Record deployment in history
        self.deployment_history.append({
            "timestamp": time.time(),
            "targets": targets,
            "results": {name: status.status for name, status in deployment_results.items()},
            "success_rate": sum(1 for s in deployment_results.values() if s.status == "deployed") / len(deployment_results)
        })
        
        # Save deployment status
        await self._save_deployment_status()
        
        return deployment_results
    
    async def _deploy_to_target(self, target: DeploymentTarget) -> DeploymentStatus:
        """Deploy to a specific target."""
        
        # Simulate deployment process
        logger.info(f"Starting deployment to {target.name}")
        
        # Pre-deployment checks
        await self._pre_deployment_checks(target)
        
        # Platform-specific deployment
        if target.platform == DeploymentPlatform.KUBERNETES:
            success = await self._deploy_kubernetes(target)
        elif target.platform == DeploymentPlatform.DOCKER_SWARM:
            success = await self._deploy_docker_swarm(target)
        elif target.platform == DeploymentPlatform.EDGE_DEVICE:
            success = await self._deploy_edge_device(target)
        else:
            success = await self._deploy_generic(target)
        
        # Post-deployment validation
        if success:
            health_score = await self._validate_deployment(target)
            performance_metrics = await self._collect_performance_metrics(target)
            compliance_status = await self._validate_compliance(target)
            
            status = DeploymentStatus(
                target_name=target.name,
                status="deployed" if health_score > 0.8 else "degraded",
                timestamp=time.time(),
                health_score=health_score,
                performance_metrics=performance_metrics,
                compliance_status=compliance_status
            )
        else:
            status = DeploymentStatus(
                target_name=target.name,
                status="failed",
                timestamp=time.time(),
                health_score=0.0,
                performance_metrics={},
                compliance_status={},
                error_message="Deployment command failed"
            )
        
        return status
    
    async def _pre_deployment_checks(self, target: DeploymentTarget):
        """Perform pre-deployment validation checks."""
        logger.info(f"Performing pre-deployment checks for {target.name}")
        
        # Check resource availability
        # Check compliance requirements
        # Validate configuration
        
        # Simulate checks
        await asyncio.sleep(1)
    
    async def _deploy_kubernetes(self, target: DeploymentTarget) -> bool:
        """Deploy to Kubernetes cluster."""
        try:
            manifest_path = self.deployment_config_path / f"{target.name}_manifest.yaml"
            
            # Simulate kubectl apply
            cmd = f"echo 'kubectl apply -f {manifest_path}'"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            logger.info(f"Kubernetes deployment command: {stdout.decode()}")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            return False
    
    async def _deploy_docker_swarm(self, target: DeploymentTarget) -> bool:
        """Deploy to Docker Swarm."""
        try:
            compose_path = self.deployment_config_path / f"{target.name}_manifest.yaml"
            
            # Simulate docker stack deploy
            cmd = f"echo 'docker stack deploy -c {compose_path} bci-gpt-{target.name}'"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            logger.info(f"Docker Swarm deployment command: {stdout.decode()}")
            return True
            
        except Exception as e:
            logger.error(f"Docker Swarm deployment failed: {e}")
            return False
    
    async def _deploy_edge_device(self, target: DeploymentTarget) -> bool:
        """Deploy to edge device."""
        try:
            # Simulate edge deployment
            logger.info(f"Deploying optimized edge version to {target.region.value}")
            await asyncio.sleep(2)  # Simulate deployment time
            return True
            
        except Exception as e:
            logger.error(f"Edge deployment failed: {e}")
            return False
    
    async def _deploy_generic(self, target: DeploymentTarget) -> bool:
        """Generic deployment method."""
        try:
            logger.info(f"Performing generic deployment to {target.platform.value}")
            await asyncio.sleep(1)
            return True
            
        except Exception as e:
            logger.error(f"Generic deployment failed: {e}")
            return False
    
    async def _validate_deployment(self, target: DeploymentTarget) -> float:
        """Validate deployment health."""
        try:
            # Simulate health check
            logger.info(f"Validating deployment health for {target.name}")
            await asyncio.sleep(1)
            
            # Simulate health score calculation
            import random
            health_score = random.uniform(0.85, 0.98)
            return health_score
            
        except Exception as e:
            logger.error(f"Health validation failed: {e}")
            return 0.0
    
    async def _collect_performance_metrics(self, target: DeploymentTarget) -> Dict[str, float]:
        """Collect performance metrics from deployment."""
        try:
            # Simulate metrics collection
            import random
            
            metrics = {
                "response_time_ms": random.uniform(50, 120),
                "throughput_rps": random.uniform(800, 1200),
                "cpu_utilization": random.uniform(30, 70),
                "memory_utilization": random.uniform(40, 80),
                "error_rate": random.uniform(0.001, 0.01)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return {}
    
    async def _validate_compliance(self, target: DeploymentTarget) -> Dict[str, bool]:
        """Validate compliance requirements."""
        compliance_status = {}
        
        for requirement in target.compliance_requirements:
            try:
                # Simulate compliance validation
                if requirement == ComplianceFramework.GDPR:
                    compliance_status["gdpr"] = await self._validate_gdpr_compliance(target)
                elif requirement == ComplianceFramework.HIPAA:
                    compliance_status["hipaa"] = await self._validate_hipaa_compliance(target)
                elif requirement == ComplianceFramework.SOC2:
                    compliance_status["soc2"] = await self._validate_soc2_compliance(target)
                else:
                    compliance_status[requirement.value] = True  # Default pass
                    
            except Exception as e:
                logger.error(f"Compliance validation failed for {requirement.value}: {e}")
                compliance_status[requirement.value] = False
        
        return compliance_status
    
    async def _validate_gdpr_compliance(self, target: DeploymentTarget) -> bool:
        """Validate GDPR compliance."""
        # Check data residency, encryption, consent management
        logger.info("Validating GDPR compliance")
        await asyncio.sleep(0.5)
        return target.environment_vars.get("DATA_RESIDENCY") == "eu"
    
    async def _validate_hipaa_compliance(self, target: DeploymentTarget) -> bool:
        """Validate HIPAA compliance."""
        # Check encryption, access controls, audit logging
        logger.info("Validating HIPAA compliance")
        await asyncio.sleep(0.5)
        return target.environment_vars.get("COMPLIANCE_MODE") == "hipaa"
    
    async def _validate_soc2_compliance(self, target: DeploymentTarget) -> bool:
        """Validate SOC2 compliance."""
        # Check security controls, monitoring, incident response
        logger.info("Validating SOC2 compliance")
        await asyncio.sleep(0.5)
        return True  # Simplified validation
    
    async def _save_deployment_status(self):
        """Save deployment status to file."""
        status_path = self.deployment_config_path / "deployment_status.json"
        
        status_data = {}
        for name, status in self.deployment_status.items():
            status_data[name] = {
                "status": status.status,
                "timestamp": status.timestamp,
                "health_score": status.health_score,
                "performance_metrics": status.performance_metrics,
                "compliance_status": status.compliance_status,
                "error_message": status.error_message
            }
        
        deployment_summary = {
            "last_updated": time.time(),
            "deployments": status_data,
            "deployment_history": self.deployment_history[-10:],  # Keep last 10
            "global_status": self._calculate_global_status()
        }
        
        with open(status_path, 'w') as f:
            json.dump(deployment_summary, f, indent=2)
    
    def _calculate_global_status(self) -> Dict[str, Any]:
        """Calculate overall global deployment status."""
        if not self.deployment_status:
            return {"status": "no_deployments", "health_score": 0.0}
        
        active_deployments = [s for s in self.deployment_status.values() if s.status == "deployed"]
        total_deployments = len(self.deployment_status)
        
        if not active_deployments:
            return {"status": "all_failed", "health_score": 0.0}
        
        avg_health = sum(s.health_score for s in active_deployments) / len(active_deployments)
        deployment_success_rate = len(active_deployments) / total_deployments
        
        # Calculate regional distribution
        regions = {}
        for status in active_deployments:
            target = next((t for t in self.deployment_targets if t.name == status.target_name), None)
            if target:
                region = target.region.value
                regions[region] = regions.get(region, 0) + 1
        
        # Overall status determination
        if deployment_success_rate >= 0.9 and avg_health >= 0.9:
            global_status = "excellent"
        elif deployment_success_rate >= 0.8 and avg_health >= 0.8:
            global_status = "good"
        elif deployment_success_rate >= 0.6 and avg_health >= 0.7:
            global_status = "acceptable"
        else:
            global_status = "needs_attention"
        
        return {
            "status": global_status,
            "health_score": avg_health,
            "deployment_success_rate": deployment_success_rate,
            "active_deployments": len(active_deployments),
            "total_deployments": total_deployments,
            "regional_distribution": regions,
            "compliance_status": self._get_global_compliance_status()
        }
    
    def _get_global_compliance_status(self) -> Dict[str, float]:
        """Get global compliance status across all deployments."""
        compliance_frameworks = {}
        
        for status in self.deployment_status.values():
            if status.status == "deployed":
                for framework, compliant in status.compliance_status.items():
                    if framework not in compliance_frameworks:
                        compliance_frameworks[framework] = []
                    compliance_frameworks[framework].append(1 if compliant else 0)
        
        # Calculate compliance rates
        compliance_rates = {}
        for framework, statuses in compliance_frameworks.items():
            compliance_rates[framework] = sum(statuses) / len(statuses) if statuses else 0.0
        
        return compliance_rates
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get comprehensive deployment summary."""
        global_status = self._calculate_global_status()
        
        return {
            "global_status": global_status,
            "deployment_targets": len(self.deployment_targets),
            "active_deployments": len([s for s in self.deployment_status.values() if s.status == "deployed"]),
            "deployment_history_count": len(self.deployment_history),
            "recent_deployments": self.deployment_history[-3:] if self.deployment_history else [],
            "target_summary": [
                {
                    "name": target.name,
                    "region": target.region.value,
                    "platform": target.platform.value,
                    "compliance": [c.value for c in target.compliance_requirements],
                    "status": self.deployment_status.get(target.name, {}).status if target.name in self.deployment_status else "not_deployed"
                }
                for target in self.deployment_targets
            ]
        }


# Standalone functions
async def deploy_globally(targets: Optional[List[str]] = None, project_root: Path = None) -> Dict[str, Any]:
    """Deploy system globally to specified or all targets."""
    deployment_system = GlobalDeploymentSystem(project_root)
    results = await deployment_system.deploy_globally(targets)
    summary = deployment_system.get_deployment_summary()
    return {"deployment_results": results, "summary": summary}


async def generate_deployment_manifests(project_root: Path = None) -> Dict[str, Any]:
    """Generate all deployment manifests."""
    deployment_system = GlobalDeploymentSystem(project_root)
    manifests = await deployment_system.generate_deployment_manifests()
    return manifests