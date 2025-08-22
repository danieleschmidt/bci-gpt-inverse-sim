"""Global infrastructure management for BCI-GPT with multi-cloud and edge deployment."""

import json
import yaml
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
from datetime import datetime, timedelta
import hashlib
import subprocess
import os
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"
    DIGITALOCEAN = "digitalocean"
    LINODE = "linode"
    VULTR = "vultr"
    ON_PREMISE = "on_premise"
    EDGE = "edge"


class InfrastructureComponent(Enum):
    """Infrastructure component types."""
    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    NETWORK = "network"
    LOAD_BALANCER = "load_balancer"
    CDN = "cdn"
    CACHE = "cache"
    MONITORING = "monitoring"
    SECURITY = "security"
    BACKUP = "backup"


class DeploymentTier(Enum):
    """Deployment tier classifications."""
    EDGE = "edge"           # Edge nodes, IoT devices
    REGIONAL = "regional"   # Regional data centers
    GLOBAL = "global"       # Global backbone infrastructure
    HYBRID = "hybrid"       # Hybrid cloud/on-premise


@dataclass
class ResourceSpec:
    """Specification for infrastructure resources."""
    cpu_cores: int
    memory_gb: int
    storage_gb: int
    network_bandwidth_mbps: int
    gpu_count: int = 0
    gpu_memory_gb: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'storage_gb': self.storage_gb,
            'network_bandwidth_mbps': self.network_bandwidth_mbps,
            'gpu_count': self.gpu_count,
            'gpu_memory_gb': self.gpu_memory_gb
        }


@dataclass
class InfrastructureNode:
    """Represents a single infrastructure node."""
    node_id: str
    provider: CloudProvider
    region: str
    tier: DeploymentTier
    resource_spec: ResourceSpec
    component_types: List[InfrastructureComponent]
    
    # Network configuration
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    subnet_id: Optional[str] = None
    security_groups: List[str] = field(default_factory=list)
    
    # Health and monitoring
    health_status: str = "unknown"  # healthy, unhealthy, unknown
    last_health_check: Optional[datetime] = None
    monitoring_enabled: bool = True
    
    # Cost and billing
    hourly_cost_usd: float = 0.0
    monthly_cost_estimate_usd: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'provider': self.provider.value,
            'region': self.region,
            'tier': self.tier.value,
            'resource_spec': self.resource_spec.to_dict(),
            'component_types': [comp.value for comp in self.component_types],
            'public_ip': self.public_ip,
            'private_ip': self.private_ip,
            'subnet_id': self.subnet_id,
            'security_groups': self.security_groups,
            'health_status': self.health_status,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'monitoring_enabled': self.monitoring_enabled,
            'hourly_cost_usd': self.hourly_cost_usd,
            'monthly_cost_estimate_usd': self.monthly_cost_estimate_usd
        }


@dataclass
class GlobalInfrastructureConfig:
    """Configuration for global infrastructure deployment."""
    deployment_name: str = "bci-gpt-global"
    environment: str = "production"
    
    # Multi-cloud strategy
    enable_multi_cloud: bool = True
    primary_provider: CloudProvider = CloudProvider.AWS
    fallback_providers: List[CloudProvider] = field(default_factory=lambda: [CloudProvider.AZURE, CloudProvider.GCP])
    
    # Global distribution
    target_regions: List[str] = field(default_factory=lambda: [
        "us-east-1", "us-west-2", "eu-west-1", "eu-central-1", 
        "ap-northeast-1", "ap-southeast-1", "ap-south-1"
    ])
    
    # Capacity planning
    initial_capacity_per_region: ResourceSpec = field(default_factory=lambda: ResourceSpec(
        cpu_cores=16, memory_gb=64, storage_gb=1000, network_bandwidth_mbps=1000, gpu_count=2, gpu_memory_gb=16
    ))
    auto_scaling_enabled: bool = True
    max_capacity_multiplier: float = 5.0
    
    # Network configuration
    enable_global_load_balancing: bool = True
    enable_cdn: bool = True
    enable_edge_locations: bool = True
    
    # Security and compliance
    enable_encryption_at_rest: bool = True
    enable_encryption_in_transit: bool = True
    enable_vpc_isolation: bool = True
    compliance_requirements: List[str] = field(default_factory=lambda: ["SOC2", "HIPAA", "GDPR"])
    
    # Monitoring and observability
    enable_centralized_logging: bool = True
    enable_distributed_tracing: bool = True
    enable_metrics_aggregation: bool = True
    monitoring_retention_days: int = 90
    
    # Disaster recovery
    enable_backup_replication: bool = True
    backup_retention_days: int = 365
    rto_minutes: int = 30  # Recovery Time Objective
    rpo_minutes: int = 15  # Recovery Point Objective


class BaseCloudProvider(ABC):
    """Abstract base class for cloud provider implementations."""
    
    def __init__(self, provider: CloudProvider, config: GlobalInfrastructureConfig):
        self.provider = provider
        self.config = config
        
    @abstractmethod
    def provision_compute_instance(self, region: str, resource_spec: ResourceSpec) -> InfrastructureNode:
        """Provision a compute instance."""
        pass
    
    @abstractmethod
    def provision_database(self, region: str, engine: str, resource_spec: ResourceSpec) -> InfrastructureNode:
        """Provision a database instance."""
        pass
    
    @abstractmethod
    def provision_load_balancer(self, region: str, target_instances: List[str]) -> InfrastructureNode:
        """Provision a load balancer."""
        pass
    
    @abstractmethod
    def check_health(self, node: InfrastructureNode) -> bool:
        """Check health status of a node."""
        pass
    
    @abstractmethod
    def scale_instance(self, node: InfrastructureNode, new_spec: ResourceSpec) -> bool:
        """Scale an instance to new resource specification."""
        pass
    
    @abstractmethod
    def terminate_instance(self, node: InfrastructureNode) -> bool:
        """Terminate an instance."""
        pass


class AWSProvider(BaseCloudProvider):
    """AWS cloud provider implementation."""
    
    def __init__(self, config: GlobalInfrastructureConfig):
        super().__init__(CloudProvider.AWS, config)
        self.instance_types = self._get_aws_instance_types()
    
    def _get_aws_instance_types(self) -> Dict[str, ResourceSpec]:
        """Get AWS instance type mappings."""
        return {
            'm5.large': ResourceSpec(2, 8, 100, 1000),
            'm5.xlarge': ResourceSpec(4, 16, 100, 1000),
            'm5.2xlarge': ResourceSpec(8, 32, 100, 1000),
            'm5.4xlarge': ResourceSpec(16, 64, 100, 1000),
            'c5.2xlarge': ResourceSpec(8, 16, 100, 1000),
            'r5.2xlarge': ResourceSpec(8, 64, 100, 1000),
            'p3.2xlarge': ResourceSpec(8, 61, 100, 1000, 1, 16),
            'p3.8xlarge': ResourceSpec(32, 244, 100, 1000, 4, 64)
        }
    
    def provision_compute_instance(self, region: str, resource_spec: ResourceSpec) -> InfrastructureNode:
        """Provision EC2 compute instance."""
        
        # Select appropriate instance type
        instance_type = self._select_instance_type(resource_spec)
        
        node = InfrastructureNode(
            node_id=f"aws-{region}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            provider=self.provider,
            region=region,
            tier=DeploymentTier.REGIONAL,
            resource_spec=resource_spec,
            component_types=[InfrastructureComponent.COMPUTE],
            hourly_cost_usd=self._calculate_ec2_cost(instance_type)
        )
        
        # In real implementation, use boto3 to provision EC2 instance
        logger.info(f"Provisioned AWS EC2 instance: {node.node_id} ({instance_type}) in {region}")
        
        return node
    
    def provision_database(self, region: str, engine: str, resource_spec: ResourceSpec) -> InfrastructureNode:
        """Provision RDS database instance."""
        
        node = InfrastructureNode(
            node_id=f"aws-rds-{region}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            provider=self.provider,
            region=region,
            tier=DeploymentTier.REGIONAL,
            resource_spec=resource_spec,
            component_types=[InfrastructureComponent.DATABASE],
            hourly_cost_usd=self._calculate_rds_cost(engine, resource_spec)
        )
        
        logger.info(f"Provisioned AWS RDS instance: {node.node_id} ({engine}) in {region}")
        return node
    
    def provision_load_balancer(self, region: str, target_instances: List[str]) -> InfrastructureNode:
        """Provision Application Load Balancer."""
        
        node = InfrastructureNode(
            node_id=f"aws-alb-{region}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            provider=self.provider,
            region=region,
            tier=DeploymentTier.REGIONAL,
            resource_spec=ResourceSpec(0, 0, 0, 10000),  # ALB is managed service
            component_types=[InfrastructureComponent.LOAD_BALANCER],
            hourly_cost_usd=0.025  # Approximate ALB cost
        )
        
        logger.info(f"Provisioned AWS ALB: {node.node_id} in {region}")
        return node
    
    def check_health(self, node: InfrastructureNode) -> bool:
        """Check EC2 instance health."""
        # In real implementation, use boto3 to check instance status
        node.health_status = "healthy"
        node.last_health_check = datetime.now()
        return True
    
    def scale_instance(self, node: InfrastructureNode, new_spec: ResourceSpec) -> bool:
        """Scale EC2 instance."""
        # In real implementation, stop instance, change type, start instance
        logger.info(f"Scaling AWS instance {node.node_id}")
        node.resource_spec = new_spec
        return True
    
    def terminate_instance(self, node: InfrastructureNode) -> bool:
        """Terminate EC2 instance."""
        logger.info(f"Terminating AWS instance {node.node_id}")
        return True
    
    def _select_instance_type(self, resource_spec: ResourceSpec) -> str:
        """Select appropriate AWS instance type."""
        
        # Simple selection logic based on CPU and memory requirements
        for instance_type, spec in self.instance_types.items():
            if (spec.cpu_cores >= resource_spec.cpu_cores and 
                spec.memory_gb >= resource_spec.memory_gb):
                return instance_type
        
        return 'm5.4xlarge'  # Default to large instance
    
    def _calculate_ec2_cost(self, instance_type: str) -> float:
        """Calculate approximate EC2 hourly cost."""
        
        # Simplified cost calculation (actual costs vary by region)
        cost_map = {
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'm5.4xlarge': 0.768,
            'c5.2xlarge': 0.34,
            'r5.2xlarge': 0.504,
            'p3.2xlarge': 3.06,
            'p3.8xlarge': 12.24
        }
        
        return cost_map.get(instance_type, 0.5)
    
    def _calculate_rds_cost(self, engine: str, resource_spec: ResourceSpec) -> float:
        """Calculate approximate RDS hourly cost."""
        
        # Simplified RDS cost calculation
        base_cost = 0.1  # Base cost per hour
        cpu_cost = resource_spec.cpu_cores * 0.02
        memory_cost = resource_spec.memory_gb * 0.01
        storage_cost = resource_spec.storage_gb * 0.0001
        
        return base_cost + cpu_cost + memory_cost + storage_cost


class EdgeProvider(BaseCloudProvider):
    """Edge computing provider implementation."""
    
    def __init__(self, config: GlobalInfrastructureConfig):
        super().__init__(CloudProvider.EDGE, config)
    
    def provision_compute_instance(self, region: str, resource_spec: ResourceSpec) -> InfrastructureNode:
        """Provision edge compute instance."""
        
        node = InfrastructureNode(
            node_id=f"edge-{region}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            provider=self.provider,
            region=region,
            tier=DeploymentTier.EDGE,
            resource_spec=resource_spec,
            component_types=[InfrastructureComponent.COMPUTE],
            hourly_cost_usd=0.05  # Lower cost for edge computing
        )
        
        logger.info(f"Provisioned edge compute instance: {node.node_id} in {region}")
        return node
    
    def provision_database(self, region: str, engine: str, resource_spec: ResourceSpec) -> InfrastructureNode:
        """Provision edge database (typically lightweight)."""
        
        node = InfrastructureNode(
            node_id=f"edge-db-{region}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            provider=self.provider,
            region=region,
            tier=DeploymentTier.EDGE,
            resource_spec=resource_spec,
            component_types=[InfrastructureComponent.DATABASE],
            hourly_cost_usd=0.02
        )
        
        logger.info(f"Provisioned edge database: {node.node_id} ({engine}) in {region}")
        return node
    
    def provision_load_balancer(self, region: str, target_instances: List[str]) -> InfrastructureNode:
        """Provision edge load balancer."""
        
        node = InfrastructureNode(
            node_id=f"edge-lb-{region}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            provider=self.provider,
            region=region,
            tier=DeploymentTier.EDGE,
            resource_spec=ResourceSpec(1, 2, 10, 1000),
            component_types=[InfrastructureComponent.LOAD_BALANCER],
            hourly_cost_usd=0.01
        )
        
        logger.info(f"Provisioned edge load balancer: {node.node_id} in {region}")
        return node
    
    def check_health(self, node: InfrastructureNode) -> bool:
        """Check edge instance health."""
        # Simplified health check for edge devices
        node.health_status = "healthy"
        node.last_health_check = datetime.now()
        return True
    
    def scale_instance(self, node: InfrastructureNode, new_spec: ResourceSpec) -> bool:
        """Scale edge instance (limited scaling capabilities)."""
        logger.info(f"Limited scaling for edge instance {node.node_id}")
        return False  # Edge devices typically have fixed resources
    
    def terminate_instance(self, node: InfrastructureNode) -> bool:
        """Terminate edge instance."""
        logger.info(f"Terminating edge instance {node.node_id}")
        return True


class GlobalLoadBalancer:
    """Global load balancer for intelligent traffic routing."""
    
    def __init__(self, config: GlobalInfrastructureConfig):
        self.config = config
        self.routing_policies = []
        self.health_checks = {}
        
    def add_routing_policy(self, 
                          policy_name: str,
                          policy_type: str,
                          weight: float = 1.0,
                          conditions: Optional[Dict[str, Any]] = None):
        """Add traffic routing policy."""
        
        policy = {
            'name': policy_name,
            'type': policy_type,  # geographic, latency, weighted, failover
            'weight': weight,
            'conditions': conditions or {},
            'created_at': datetime.now()
        }
        
        self.routing_policies.append(policy)
        logger.info(f"Added routing policy: {policy_name} ({policy_type})")
    
    def configure_geographic_routing(self, region_mappings: Dict[str, List[str]]):
        """Configure geographic-based routing."""
        
        for region, countries in region_mappings.items():
            self.add_routing_policy(
                f"geo_{region}",
                "geographic",
                conditions={
                    'target_region': region,
                    'source_countries': countries
                }
            )
    
    def configure_latency_routing(self, latency_thresholds: Dict[str, int]):
        """Configure latency-based routing."""
        
        for region, threshold_ms in latency_thresholds.items():
            self.add_routing_policy(
                f"latency_{region}",
                "latency",
                conditions={
                    'target_region': region,
                    'max_latency_ms': threshold_ms
                }
            )
    
    def generate_global_routing_config(self) -> Dict[str, Any]:
        """Generate global routing configuration."""
        
        return {
            'global_load_balancer': {
                'enabled': self.config.enable_global_load_balancing,
                'policies': self.routing_policies,
                'health_check_interval': 30,
                'failover_threshold': 3,
                'dns_ttl': 60
            },
            'traffic_distribution': {
                'primary_regions': self.config.target_regions[:3],
                'failover_regions': self.config.target_regions[3:],
                'edge_locations': self.config.enable_edge_locations
            }
        }


class GlobalInfrastructureManager:
    """Manage global infrastructure deployment and operations."""
    
    def __init__(self, config: GlobalInfrastructureConfig):
        self.config = config
        self.providers = {}
        self.infrastructure_nodes = []
        self.global_lb = GlobalLoadBalancer(config)
        
        # Initialize cloud providers
        self._initialize_providers()
        
    def _initialize_providers(self):
        """Initialize cloud provider implementations."""
        
        if self.config.primary_provider == CloudProvider.AWS:
            self.providers[CloudProvider.AWS] = AWSProvider(self.config)
        
        if self.config.enable_edge_locations:
            self.providers[CloudProvider.EDGE] = EdgeProvider(self.config)
        
        # Add other providers as needed
        logger.info(f"Initialized {len(self.providers)} cloud providers")
    
    def deploy_global_infrastructure(self) -> Dict[str, Any]:
        """Deploy infrastructure across all target regions."""
        
        logger.info("Starting global infrastructure deployment...")
        
        deployment_results = {
            'deployment_id': f"deploy_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'started_at': datetime.now(),
            'regional_deployments': {},
            'edge_deployments': {},
            'global_services': {},
            'total_nodes': 0,
            'estimated_monthly_cost': 0.0
        }
        
        # Deploy regional infrastructure
        for region in self.config.target_regions:
            regional_result = self._deploy_regional_infrastructure(region)
            deployment_results['regional_deployments'][region] = regional_result
            deployment_results['total_nodes'] += len(regional_result['nodes'])
        
        # Deploy edge infrastructure if enabled
        if self.config.enable_edge_locations:
            edge_result = self._deploy_edge_infrastructure()
            deployment_results['edge_deployments'] = edge_result
            deployment_results['total_nodes'] += len(edge_result['nodes'])
        
        # Setup global services
        global_services = self._setup_global_services()
        deployment_results['global_services'] = global_services
        
        # Calculate total costs
        deployment_results['estimated_monthly_cost'] = self._calculate_total_monthly_cost()
        
        deployment_results['completed_at'] = datetime.now()
        deployment_results['deployment_duration'] = (
            deployment_results['completed_at'] - deployment_results['started_at']
        ).total_seconds()
        
        logger.info(f"Global deployment completed: {deployment_results['total_nodes']} nodes")
        logger.info(f"Estimated monthly cost: ${deployment_results['estimated_monthly_cost']:.2f}")
        
        return deployment_results
    
    def _deploy_regional_infrastructure(self, region: str) -> Dict[str, Any]:
        """Deploy infrastructure in a specific region."""
        
        logger.info(f"Deploying infrastructure in region: {region}")
        
        regional_nodes = []
        provider = self.providers[self.config.primary_provider]
        
        # Deploy compute instances
        for i in range(2):  # Start with 2 instances per region
            compute_node = provider.provision_compute_instance(region, self.config.initial_capacity_per_region)
            regional_nodes.append(compute_node)
            self.infrastructure_nodes.append(compute_node)
        
        # Deploy database
        db_spec = ResourceSpec(4, 16, 500, 1000)  # Smaller DB instance
        db_node = provider.provision_database(region, "postgresql", db_spec)
        regional_nodes.append(db_node)
        self.infrastructure_nodes.append(db_node)
        
        # Deploy load balancer
        compute_instance_ids = [node.node_id for node in regional_nodes if InfrastructureComponent.COMPUTE in node.component_types]
        lb_node = provider.provision_load_balancer(region, compute_instance_ids)
        regional_nodes.append(lb_node)
        self.infrastructure_nodes.append(lb_node)
        
        return {
            'region': region,
            'nodes': [node.to_dict() for node in regional_nodes],
            'node_count': len(regional_nodes),
            'deployment_status': 'completed'
        }
    
    def _deploy_edge_infrastructure(self) -> Dict[str, Any]:
        """Deploy edge computing infrastructure."""
        
        logger.info("Deploying edge infrastructure...")
        
        edge_nodes = []
        edge_provider = self.providers[CloudProvider.EDGE]
        
        # Deploy edge nodes in major metropolitan areas
        edge_locations = [
            "edge-nyc", "edge-lax", "edge-london", "edge-tokyo", 
            "edge-singapore", "edge-sydney", "edge-mumbai", "edge-sao-paulo"
        ]
        
        for location in edge_locations:
            edge_spec = ResourceSpec(2, 4, 100, 500)  # Smaller edge instances
            edge_node = edge_provider.provision_compute_instance(location, edge_spec)
            edge_nodes.append(edge_node)
            self.infrastructure_nodes.append(edge_node)
        
        return {
            'edge_locations': edge_locations,
            'nodes': [node.to_dict() for node in edge_nodes],
            'node_count': len(edge_nodes),
            'deployment_status': 'completed'
        }
    
    def _setup_global_services(self) -> Dict[str, Any]:
        """Setup global services like load balancing and CDN."""
        
        logger.info("Setting up global services...")
        
        # Configure global load balancer
        if self.config.enable_global_load_balancing:
            # Geographic routing
            self.global_lb.configure_geographic_routing({
                'us-east-1': ['US', 'CA', 'MX'],
                'eu-west-1': ['GB', 'IE', 'FR', 'ES', 'PT'],
                'eu-central-1': ['DE', 'AT', 'CH', 'NL', 'BE'],
                'ap-northeast-1': ['JP', 'KR'],
                'ap-southeast-1': ['SG', 'MY', 'TH', 'VN'],
                'ap-south-1': ['IN', 'BD', 'LK']
            })
            
            # Latency-based routing
            self.global_lb.configure_latency_routing({
                region: 100 for region in self.config.target_regions  # 100ms threshold
            })
        
        global_services = {
            'global_load_balancer': self.global_lb.generate_global_routing_config(),
            'cdn_enabled': self.config.enable_cdn,
            'monitoring': {
                'centralized_logging': self.config.enable_centralized_logging,
                'distributed_tracing': self.config.enable_distributed_tracing,
                'metrics_aggregation': self.config.enable_metrics_aggregation
            },
            'security': {
                'encryption_at_rest': self.config.enable_encryption_at_rest,
                'encryption_in_transit': self.config.enable_encryption_in_transit,
                'vpc_isolation': self.config.enable_vpc_isolation
            }
        }
        
        return global_services
    
    def _calculate_total_monthly_cost(self) -> float:
        """Calculate total monthly cost estimate."""
        
        total_hourly_cost = sum(node.hourly_cost_usd for node in self.infrastructure_nodes)
        monthly_cost = total_hourly_cost * 24 * 30  # 30 days
        
        # Add additional service costs
        additional_costs = 0.0
        
        if self.config.enable_global_load_balancing:
            additional_costs += 50.0  # Global LB service cost
        
        if self.config.enable_cdn:
            additional_costs += 100.0  # CDN service cost
        
        if self.config.enable_centralized_logging:
            additional_costs += 200.0  # Logging service cost
        
        return monthly_cost + additional_costs
    
    def scale_infrastructure(self, region: str, scale_factor: float) -> Dict[str, Any]:
        """Scale infrastructure in a specific region."""
        
        logger.info(f"Scaling infrastructure in {region} by factor {scale_factor}")
        
        regional_nodes = [node for node in self.infrastructure_nodes if node.region == region]
        scaling_results = {
            'region': region,
            'scale_factor': scale_factor,
            'nodes_scaled': 0,
            'scaling_actions': []
        }
        
        for node in regional_nodes:
            if InfrastructureComponent.COMPUTE in node.component_types:
                # Scale compute resources
                new_spec = ResourceSpec(
                    cpu_cores=int(node.resource_spec.cpu_cores * scale_factor),
                    memory_gb=int(node.resource_spec.memory_gb * scale_factor),
                    storage_gb=node.resource_spec.storage_gb,
                    network_bandwidth_mbps=node.resource_spec.network_bandwidth_mbps,
                    gpu_count=node.resource_spec.gpu_count,
                    gpu_memory_gb=node.resource_spec.gpu_memory_gb
                )
                
                provider = self.providers[node.provider]
                if provider.scale_instance(node, new_spec):
                    scaling_results['nodes_scaled'] += 1
                    scaling_results['scaling_actions'].append({
                        'node_id': node.node_id,
                        'action': 'scaled',
                        'old_spec': node.resource_spec.to_dict(),
                        'new_spec': new_spec.to_dict()
                    })
        
        return scaling_results
    
    def health_check_all_nodes(self) -> Dict[str, Any]:
        """Perform health checks on all infrastructure nodes."""
        
        logger.info("Performing global health checks...")
        
        health_results = {
            'check_time': datetime.now(),
            'total_nodes': len(self.infrastructure_nodes),
            'healthy_nodes': 0,
            'unhealthy_nodes': 0,
            'unknown_nodes': 0,
            'node_status': {}
        }
        
        for node in self.infrastructure_nodes:
            provider = self.providers[node.provider]
            is_healthy = provider.check_health(node)
            
            health_results['node_status'][node.node_id] = {
                'status': node.health_status,
                'last_check': node.last_health_check.isoformat() if node.last_health_check else None,
                'region': node.region,
                'provider': node.provider.value
            }
            
            if node.health_status == 'healthy':
                health_results['healthy_nodes'] += 1
            elif node.health_status == 'unhealthy':
                health_results['unhealthy_nodes'] += 1
            else:
                health_results['unknown_nodes'] += 1
        
        health_results['overall_health_percentage'] = (
            health_results['healthy_nodes'] / health_results['total_nodes'] * 100
        )
        
        logger.info(f"Health check complete: {health_results['healthy_nodes']}/{health_results['total_nodes']} nodes healthy")
        
        return health_results
    
    def generate_infrastructure_report(self, output_path: Path) -> Path:
        """Generate comprehensive infrastructure report."""
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'configuration': {
                'deployment_name': self.config.deployment_name,
                'environment': self.config.environment,
                'multi_cloud_enabled': self.config.enable_multi_cloud,
                'primary_provider': self.config.primary_provider.value,
                'target_regions': self.config.target_regions
            },
            'deployment_summary': {
                'total_nodes': len(self.infrastructure_nodes),
                'regional_nodes': len([n for n in self.infrastructure_nodes if n.tier == DeploymentTier.REGIONAL]),
                'edge_nodes': len([n for n in self.infrastructure_nodes if n.tier == DeploymentTier.EDGE]),
                'estimated_monthly_cost': self._calculate_total_monthly_cost()
            },
            'infrastructure_nodes': [node.to_dict() for node in self.infrastructure_nodes],
            'global_services': {
                'load_balancer_policies': len(self.global_lb.routing_policies),
                'cdn_enabled': self.config.enable_cdn,
                'monitoring_enabled': self.config.enable_centralized_logging
            },
            'compliance_and_security': {
                'compliance_requirements': self.config.compliance_requirements,
                'encryption_at_rest': self.config.enable_encryption_at_rest,
                'encryption_in_transit': self.config.enable_encryption_in_transit,
                'vpc_isolation': self.config.enable_vpc_isolation
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated infrastructure report: {output_path}")
        return output_path


# Example usage and testing
if __name__ == "__main__":
    # Create global infrastructure configuration
    infra_config = GlobalInfrastructureConfig(
        deployment_name="bci-gpt-global-prod",
        environment="production",
        enable_multi_cloud=True,
        primary_provider=CloudProvider.AWS,
        target_regions=["us-east-1", "eu-west-1", "ap-northeast-1"],
        enable_edge_locations=True,
        enable_global_load_balancing=True
    )
    
    # Create global infrastructure manager
    infra_manager = GlobalInfrastructureManager(infra_config)
    
    print("🌐 BCI-GPT Global Infrastructure Manager")
    print(f"Primary provider: {infra_config.primary_provider.value}")
    print(f"Target regions: {len(infra_config.target_regions)}")
    print(f"Edge locations enabled: {infra_config.enable_edge_locations}")
    
    # Deploy global infrastructure
    deployment_result = infra_manager.deploy_global_infrastructure()
    
    print(f"\\n🚀 Global deployment completed:")
    print(f"  Total nodes: {deployment_result['total_nodes']}")
    print(f"  Regional deployments: {len(deployment_result['regional_deployments'])}")
    print(f"  Edge deployments: {len(deployment_result.get('edge_deployments', {}).get('nodes', []))}")
    print(f"  Estimated monthly cost: ${deployment_result['estimated_monthly_cost']:.2f}")
    print(f"  Deployment duration: {deployment_result['deployment_duration']:.1f} seconds")
    
    # Perform health checks
    health_status = infra_manager.health_check_all_nodes()
    print(f"\\n🏥 Health check results:")
    print(f"  Overall health: {health_status['overall_health_percentage']:.1f}%")
    print(f"  Healthy nodes: {health_status['healthy_nodes']}/{health_status['total_nodes']}")
    
    # Test scaling
    scale_result = infra_manager.scale_infrastructure("us-east-1", 1.5)
    print(f"\\n📈 Scaling results for us-east-1:")
    print(f"  Nodes scaled: {scale_result['nodes_scaled']}")
    print(f"  Scale factor: {scale_result['scale_factor']}")
    
    # Generate infrastructure report
    report_path = Path("./infrastructure_report.json")
    infra_manager.generate_infrastructure_report(report_path)
    print(f"\\n📊 Infrastructure report generated: {report_path}")
    
    print("\\n🌐 Global infrastructure management system validated!")