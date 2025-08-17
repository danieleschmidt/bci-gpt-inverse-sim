"""
Autonomous SDLC System for BCI-GPT
Progressive quality gates, self-healing, and adaptive optimization.
"""

from .progressive_quality_gates import (
    ProgressiveQualityGates,
    QualityGate,
    QualityGateResult,
    QualityGateStatus,
    QualityGatePriority,
    run_quality_gates,
    start_continuous_monitoring
)

from .self_healing_system import (
    SelfHealingSystem,
    EnhancedQualityGates,
    HealingAction,
    HealingRule,
    heal_system,
    run_enhanced_quality_gates
)

from .adaptive_scaling_system import (
    AdaptiveScalingSystem,
    ResourceMetrics,
    PerformanceProfile,
    ResourceType,
    OptimizationStrategy,
    optimize_system_performance
)

from .research_framework import (
    ResearchFramework,
    ResearchOpportunity,
    ResearchArea,
    PublicationVenue,
    discover_and_validate_research,
    generate_research_report
)

from .global_deployment_system import (
    GlobalDeploymentSystem,
    DeploymentTarget,
    DeploymentRegion,
    DeploymentPlatform,
    ComplianceFramework,
    deploy_globally,
    generate_deployment_manifests
)

from .autonomous_sdlc_orchestrator import (
    AutonomousSDLCOrchestrator,
    SDLCPhase,
    ExecutionStrategy,
    SDLCMetrics,
    SDLCDecision,
    execute_autonomous_sdlc,
    run_single_sdlc_cycle
)

__all__ = [
    'ProgressiveQualityGates',
    'QualityGate', 
    'QualityGateResult',
    'QualityGateStatus',
    'QualityGatePriority',
    'run_quality_gates',
    'start_continuous_monitoring',
    'SelfHealingSystem',
    'EnhancedQualityGates',
    'HealingAction',
    'HealingRule',
    'heal_system',
    'run_enhanced_quality_gates',
    'AdaptiveScalingSystem',
    'ResourceMetrics',
    'PerformanceProfile',
    'ResourceType',
    'OptimizationStrategy',
    'optimize_system_performance',
    'ResearchFramework',
    'ResearchOpportunity',
    'ResearchArea',
    'PublicationVenue',
    'discover_and_validate_research',
    'generate_research_report',
    'GlobalDeploymentSystem',
    'DeploymentTarget',
    'DeploymentRegion',
    'DeploymentPlatform',
    'ComplianceFramework',
    'deploy_globally',
    'generate_deployment_manifests',
    'AutonomousSDLCOrchestrator',
    'SDLCPhase',
    'ExecutionStrategy',
    'SDLCMetrics',
    'SDLCDecision',
    'execute_autonomous_sdlc',
    'run_single_sdlc_cycle'
]