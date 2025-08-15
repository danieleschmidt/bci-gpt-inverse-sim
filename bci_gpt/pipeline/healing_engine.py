"""Healing Decision Engine for BCI-GPT Self-Healing System.

Intelligent decision engine that analyzes system state, prioritizes issues,
and generates optimal healing strategies using rule-based and ML approaches.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np

from ..utils.monitoring import HealthStatus
from ..utils.error_handling import BCI_GPTError


class HealingStrategy(Enum):
    """Types of healing strategies."""
    RESTART = "restart"
    FAILOVER = "failover"
    SCALE = "scale"
    OPTIMIZE = "optimize"
    DEGRADE = "degrade"
    ISOLATE = "isolate"
    BACKUP = "backup"
    MANUAL_INTERVENTION = "manual_intervention"


class IssueSeverity(Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(Enum):
    """Categories of issues."""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    DATA_QUALITY = "data_quality"
    RESOURCE = "resource"
    NETWORK = "network"
    MODEL = "model"
    SYSTEM = "system"


@dataclass
class HealingAction:
    """A specific healing action to take."""
    type: str
    component: str
    parameters: Dict[str, Any]
    expected_duration: float  # seconds
    success_probability: float  # 0-1
    risk_level: str  # "low", "medium", "high"
    prerequisites: List[str]
    rollback_action: Optional[Dict[str, Any]] = None


@dataclass
class HealingPlan:
    """Complete healing plan for an issue."""
    issue_id: str
    severity: IssueSeverity
    category: IssueCategory
    primary_actions: List[HealingAction]
    fallback_actions: List[HealingAction]
    estimated_recovery_time: float
    confidence_score: float
    created_at: datetime
    dependencies: List[str]


class HealingDecisionEngine:
    """Intelligent decision engine for generating healing strategies.
    
    Analyzes system state, issue context, and historical data to generate
    optimal healing plans with fallback strategies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Knowledge base and rules
        self.healing_rules: Dict[str, Dict[str, Any]] = {}
        self.success_history: Dict[str, List[Dict[str, Any]]] = {}
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
        # Decision factors and weights
        self.decision_weights = {
            "severity": 0.3,
            "impact": 0.25,
            "success_probability": 0.2,
            "recovery_time": 0.15,
            "risk_level": 0.1
        }
        
        # Strategy preferences
        self.strategy_preferences = {
            HealingStrategy.RESTART: {"risk": "low", "effectiveness": 0.7, "time": 30},
            HealingStrategy.FAILOVER: {"risk": "medium", "effectiveness": 0.9, "time": 10},
            HealingStrategy.SCALE: {"risk": "low", "effectiveness": 0.8, "time": 60},
            HealingStrategy.OPTIMIZE: {"risk": "low", "effectiveness": 0.6, "time": 120},
            HealingStrategy.DEGRADE: {"risk": "low", "effectiveness": 0.5, "time": 5},
            HealingStrategy.ISOLATE: {"risk": "high", "effectiveness": 0.9, "time": 15},
            HealingStrategy.BACKUP: {"risk": "medium", "effectiveness": 0.8, "time": 45},
            HealingStrategy.MANUAL_INTERVENTION: {"risk": "high", "effectiveness": 1.0, "time": 300}
        }
        
        # Component-specific healing strategies
        self.component_strategies = {
            "pipeline": [HealingStrategy.RESTART, HealingStrategy.FAILOVER, HealingStrategy.ISOLATE],
            "model": [HealingStrategy.RESTART, HealingStrategy.BACKUP, HealingStrategy.OPTIMIZE],
            "data": [HealingStrategy.FAILOVER, HealingStrategy.RESTART, HealingStrategy.BACKUP],
            "realtime": [HealingStrategy.SCALE, HealingStrategy.OPTIMIZE, HealingStrategy.DEGRADE],
            "system": [HealingStrategy.RESTART, HealingStrategy.SCALE, HealingStrategy.MANUAL_INTERVENTION]
        }
        
        # Load default healing rules
        self._initialize_healing_rules()
        
        # Learning system
        self.learning_enabled = True
        self.min_learning_samples = 10
    
    def _initialize_healing_rules(self) -> None:
        """Initialize default healing rules."""
        self.healing_rules = {
            "pipeline_critical": {
                "conditions": {"component": "pipeline", "severity": "critical"},
                "strategies": ["failover", "restart"],
                "max_attempts": 3,
                "cooldown": 60
            },
            "model_degradation": {
                "conditions": {"component": "model", "type": "model_degradation"},
                "strategies": ["optimize", "backup", "restart"],
                "max_attempts": 2,
                "cooldown": 300
            },
            "data_corruption": {
                "conditions": {"component": "data", "type": "data_corruption"},
                "strategies": ["failover", "restart", "backup"],
                "max_attempts": 3,
                "cooldown": 30
            },
            "performance_degradation": {
                "conditions": {"component": "realtime", "type": "performance_degradation"},
                "strategies": ["scale", "optimize", "degrade"],
                "max_attempts": 2,
                "cooldown": 120
            },
            "resource_exhaustion": {
                "conditions": {"category": "resource"},
                "strategies": ["scale", "optimize", "isolate"],
                "max_attempts": 2,
                "cooldown": 180
            }
        }
    
    def generate_healing_plan(self, issue: Dict[str, Any], system_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimal healing plan for an issue."""
        self.logger.info(f"Generating healing plan for issue: {issue.get('type', 'unknown')}")
        
        # Analyze issue characteristics
        issue_analysis = self._analyze_issue(issue, system_context)
        
        # Determine applicable strategies
        applicable_strategies = self._determine_applicable_strategies(issue_analysis)
        
        # Rank strategies by effectiveness
        ranked_strategies = self._rank_strategies(applicable_strategies, issue_analysis, system_context)
        
        # Generate healing actions
        healing_actions = self._generate_healing_actions(ranked_strategies, issue_analysis)
        
        # Create healing plan
        plan = self._create_healing_plan(issue, issue_analysis, healing_actions)
        
        self.logger.info(f"Generated healing plan with {len(healing_actions)} actions, confidence: {plan.confidence_score:.2f}")
        
        return self._plan_to_action_list(plan)
    
    def _analyze_issue(self, issue: Dict[str, Any], system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze issue characteristics and context."""
        analysis = {
            "component": issue.get("component", "unknown"),
            "issue_type": issue.get("type", "unknown"),
            "severity": self._determine_severity(issue, system_context),
            "category": self._categorize_issue(issue),
            "impact_scope": self._assess_impact_scope(issue, system_context),
            "urgency": self._assess_urgency(issue, system_context),
            "complexity": self._assess_complexity(issue),
            "historical_context": self._get_historical_context(issue),
            "system_state": self._analyze_system_state(system_context),
            "resource_availability": self._assess_resource_availability(system_context)
        }
        
        return analysis
    
    def _determine_severity(self, issue: Dict[str, Any], system_context: Dict[str, Any]) -> IssueSeverity:
        """Determine issue severity based on context."""
        severity_score = 0.0
        
        # Component criticality
        component = issue.get("component", "")
        if component in ["pipeline", "model"]:
            severity_score += 0.4
        elif component in ["data", "realtime"]:
            severity_score += 0.3
        else:
            severity_score += 0.2
        
        # Issue type severity
        issue_type = issue.get("type", "")
        if "critical" in issue_type or "failure" in issue_type:
            severity_score += 0.4
        elif "degradation" in issue_type or "corruption" in issue_type:
            severity_score += 0.3
        elif "performance" in issue_type or "warning" in issue_type:
            severity_score += 0.2
        
        # System health impact
        overall_health = system_context.get("overall_health_score", 1.0)
        if overall_health < 0.3:
            severity_score += 0.2
        elif overall_health < 0.6:
            severity_score += 0.1
        
        # Map score to severity level
        if severity_score >= 0.8:
            return IssueSeverity.CRITICAL
        elif severity_score >= 0.6:
            return IssueSeverity.HIGH
        elif severity_score >= 0.4:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
    
    def _categorize_issue(self, issue: Dict[str, Any]) -> IssueCategory:
        """Categorize the issue type."""
        issue_type = issue.get("type", "").lower()
        component = issue.get("component", "").lower()
        
        if "performance" in issue_type or "latency" in issue_type or "throughput" in issue_type:
            return IssueCategory.PERFORMANCE
        elif "data" in issue_type or "corruption" in issue_type or "quality" in issue_type:
            return IssueCategory.DATA_QUALITY
        elif "model" in issue_type or "accuracy" in issue_type or "inference" in issue_type:
            return IssueCategory.MODEL
        elif "memory" in issue_type or "cpu" in issue_type or "resource" in issue_type:
            return IssueCategory.RESOURCE
        elif "network" in issue_type or "connection" in issue_type:
            return IssueCategory.NETWORK
        elif "failure" in issue_type or "error" in issue_type:
            return IssueCategory.RELIABILITY
        else:
            return IssueCategory.SYSTEM
    
    def _assess_impact_scope(self, issue: Dict[str, Any], system_context: Dict[str, Any]) -> str:
        """Assess the scope of impact."""
        component = issue.get("component", "")
        
        # Pipeline issues affect entire system
        if component == "pipeline":
            return "system_wide"
        
        # Model issues affect inference quality
        elif component == "model":
            return "inference_quality"
        
        # Data issues affect input quality
        elif component == "data":
            return "data_flow"
        
        # Realtime issues affect performance
        elif component == "realtime":
            return "performance"
        
        else:
            return "component_local"
    
    def _assess_urgency(self, issue: Dict[str, Any], system_context: Dict[str, Any]) -> str:
        """Assess how urgently the issue needs to be addressed."""
        severity = issue.get("severity", "low")
        component = issue.get("component", "")
        
        # Critical issues always urgent
        if severity == "critical":
            return "immediate"
        
        # Pipeline and model issues are high priority
        elif component in ["pipeline", "model"] and severity == "high":
            return "high"
        
        # Performance issues during high load
        elif component == "realtime" and system_context.get("load_level", "normal") == "high":
            return "high"
        
        # Data issues if no backup sources
        elif component == "data" and not system_context.get("backup_available", True):
            return "high"
        
        else:
            return "normal"
    
    def _assess_complexity(self, issue: Dict[str, Any]) -> str:
        """Assess issue complexity for healing."""
        issue_type = issue.get("type", "")
        component = issue.get("component", "")
        
        # Model issues are typically complex
        if component == "model":
            return "high"
        
        # Performance optimization is complex
        elif "optimization" in issue_type or "performance" in issue_type:
            return "medium"
        
        # Simple restart/failover issues
        elif "restart" in issue_type or "failover" in issue_type:
            return "low"
        
        else:
            return "medium"
    
    def _get_historical_context(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical context for similar issues."""
        issue_key = f"{issue.get('component')}_{issue.get('type')}"
        
        success_history = self.success_history.get(issue_key, [])
        failure_patterns = self.failure_patterns.get(issue_key, [])
        
        return {
            "previous_successes": len(success_history),
            "previous_failures": len(failure_patterns),
            "most_successful_strategy": self._get_most_successful_strategy(success_history),
            "failure_rate": len(failure_patterns) / max(len(success_history) + len(failure_patterns), 1),
            "last_occurrence": self._get_last_occurrence(success_history + failure_patterns)
        }
    
    def _get_most_successful_strategy(self, success_history: List[Dict[str, Any]]) -> Optional[str]:
        """Get the most successful healing strategy from history."""
        if not success_history:
            return None
        
        strategy_counts = {}
        for record in success_history:
            strategy = record.get("strategy")
            if strategy:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        if strategy_counts:
            return max(strategy_counts, key=strategy_counts.get)
        return None
    
    def _get_last_occurrence(self, history: List[Dict[str, Any]]) -> Optional[datetime]:
        """Get the timestamp of last occurrence."""
        if not history:
            return None
        
        timestamps = []
        for record in history:
            if "timestamp" in record:
                try:
                    timestamps.append(datetime.fromisoformat(record["timestamp"]))
                except:
                    pass
        
        return max(timestamps) if timestamps else None
    
    def _analyze_system_state(self, system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current system state for decision making."""
        return {
            "overall_health": system_context.get("overall_health_score", 0.5),
            "load_level": self._determine_load_level(system_context),
            "resource_utilization": self._get_resource_utilization(system_context),
            "active_incidents": self._count_active_incidents(system_context),
            "backup_availability": self._check_backup_availability(system_context),
            "maintenance_window": self._check_maintenance_window()
        }
    
    def _determine_load_level(self, system_context: Dict[str, Any]) -> str:
        """Determine current system load level."""
        # Check real-time metrics if available
        realtime_data = system_context.get("realtime", {})
        performance_score = realtime_data.get("performance_score", 0.8)
        
        if performance_score < 0.3:
            return "critical"
        elif performance_score < 0.6:
            return "high"
        elif performance_score < 0.8:
            return "normal"
        else:
            return "low"
    
    def _get_resource_utilization(self, system_context: Dict[str, Any]) -> Dict[str, float]:
        """Get current resource utilization."""
        realtime_data = system_context.get("realtime", {})
        current_metrics = realtime_data.get("current_metrics", {})
        
        return {
            "cpu": current_metrics.get("cpu_usage", 0.0),
            "memory": current_metrics.get("memory_usage_mb", 0.0) / 1024.0,  # Convert to GB
            "gpu": current_metrics.get("gpu_usage", 0.0),
            "queue": current_metrics.get("queue_depth", 0)
        }
    
    def _count_active_incidents(self, system_context: Dict[str, Any]) -> int:
        """Count active incidents in the system."""
        # Count critical issues across components
        active_count = 0
        
        for component_name, component_data in system_context.items():
            if isinstance(component_data, dict):
                if component_data.get("status") in ["critical", "unhealthy"]:
                    active_count += 1
                
                # Check for critical issues list
                critical_issues = component_data.get("critical_issues", [])
                active_count += len(critical_issues)
        
        return active_count
    
    def _check_backup_availability(self, system_context: Dict[str, Any]) -> Dict[str, bool]:
        """Check availability of backup systems."""
        return {
            "model": system_context.get("model", {}).get("backup_models_available", 0) > 0,
            "data": len(system_context.get("data", {}).get("data_sources", {})) > 1,
            "pipeline": True,  # Assume pipeline backup is available
            "realtime": True   # Assume real-time backup is available
        }
    
    def _check_maintenance_window(self) -> bool:
        """Check if we're in a maintenance window."""
        # Simple check - assume maintenance window is 2-4 AM UTC
        current_hour = datetime.utcnow().hour
        return 2 <= current_hour <= 4
    
    def _assess_resource_availability(self, system_context: Dict[str, Any]) -> Dict[str, float]:
        """Assess available resources for healing actions."""
        utilization = self._get_resource_utilization(system_context)
        
        return {
            "cpu": max(0.0, 100.0 - utilization["cpu"]),
            "memory": max(0.0, 8.0 - utilization["memory"]),  # Assume 8GB total
            "gpu": max(0.0, 100.0 - utilization["gpu"]),
            "scaling_capacity": 0.8  # Assume 80% scaling capacity available
        }
    
    def _determine_applicable_strategies(self, issue_analysis: Dict[str, Any]) -> List[HealingStrategy]:
        """Determine which healing strategies are applicable."""
        component = issue_analysis["component"]
        severity = issue_analysis["severity"]
        category = issue_analysis["category"]
        
        # Get component-specific strategies
        applicable = self.component_strategies.get(component, [HealingStrategy.RESTART])
        
        # Add category-specific strategies
        if category == IssueCategory.PERFORMANCE:
            applicable.extend([HealingStrategy.SCALE, HealingStrategy.OPTIMIZE])
        elif category == IssueCategory.RELIABILITY:
            applicable.extend([HealingStrategy.FAILOVER, HealingStrategy.RESTART])
        elif category == IssueCategory.RESOURCE:
            applicable.extend([HealingStrategy.SCALE, HealingStrategy.ISOLATE])
        
        # Add severity-based strategies
        if severity == IssueSeverity.CRITICAL:
            applicable.extend([HealingStrategy.MANUAL_INTERVENTION])
        
        # Remove duplicates and return
        return list(set(applicable))
    
    def _rank_strategies(self, strategies: List[HealingStrategy], issue_analysis: Dict[str, Any], 
                        system_context: Dict[str, Any]) -> List[Tuple[HealingStrategy, float]]:
        """Rank healing strategies by effectiveness and suitability."""
        strategy_scores = []
        
        for strategy in strategies:
            score = self._calculate_strategy_score(strategy, issue_analysis, system_context)
            strategy_scores.append((strategy, score))
        
        # Sort by score (highest first)
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        return strategy_scores
    
    def _calculate_strategy_score(self, strategy: HealingStrategy, issue_analysis: Dict[str, Any], 
                                 system_context: Dict[str, Any]) -> float:
        """Calculate score for a healing strategy."""
        base_score = self.strategy_preferences[strategy]["effectiveness"]
        
        # Historical success rate
        historical_context = issue_analysis["historical_context"]
        most_successful = historical_context.get("most_successful_strategy")
        if most_successful == strategy.value:
            base_score *= 1.3  # 30% bonus for historically successful strategy
        
        # Adjust for severity
        severity = issue_analysis["severity"]
        if severity == IssueSeverity.CRITICAL and strategy in [HealingStrategy.FAILOVER, HealingStrategy.MANUAL_INTERVENTION]:
            base_score *= 1.2
        elif severity == IssueSeverity.LOW and strategy == HealingStrategy.MANUAL_INTERVENTION:
            base_score *= 0.5  # Penalize manual intervention for low severity
        
        # Adjust for urgency
        urgency = issue_analysis["urgency"]
        strategy_time = self.strategy_preferences[strategy]["time"]
        if urgency == "immediate" and strategy_time > 60:
            base_score *= 0.7  # Penalize slow strategies for urgent issues
        
        # Adjust for system state
        system_state = issue_analysis["system_state"]
        if system_state["load_level"] == "critical" and strategy == HealingStrategy.SCALE:
            base_score *= 1.2  # Favor scaling under critical load
        
        # Adjust for resource availability
        resource_availability = issue_analysis["resource_availability"]
        if strategy == HealingStrategy.SCALE and resource_availability["scaling_capacity"] < 0.2:
            base_score *= 0.3  # Penalize scaling if resources not available
        
        # Risk adjustment
        risk_level = self.strategy_preferences[strategy]["risk"]
        active_incidents = system_state["active_incidents"]
        if active_incidents > 2 and risk_level == "high":
            base_score *= 0.6  # Penalize high-risk strategies when already handling multiple incidents
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_healing_actions(self, ranked_strategies: List[Tuple[HealingStrategy, float]], 
                                 issue_analysis: Dict[str, Any]) -> List[HealingAction]:
        """Generate specific healing actions from strategies."""
        actions = []
        component = issue_analysis["component"]
        
        # Take top 3 strategies for primary actions
        primary_strategies = ranked_strategies[:3]
        
        for strategy, score in primary_strategies:
            action = self._create_healing_action(strategy, component, issue_analysis, score)
            if action:
                actions.append(action)
        
        # Add fallback actions from remaining strategies
        fallback_strategies = ranked_strategies[3:5]  # Up to 2 fallback actions
        for strategy, score in fallback_strategies:
            action = self._create_healing_action(strategy, component, issue_analysis, score)
            if action:
                action.type = f"fallback_{action.type}"
                actions.append(action)
        
        return actions
    
    def _create_healing_action(self, strategy: HealingStrategy, component: str, 
                              issue_analysis: Dict[str, Any], score: float) -> Optional[HealingAction]:
        """Create a specific healing action for a strategy."""
        strategy_info = self.strategy_preferences[strategy]
        
        # Determine action type based on strategy and component
        if strategy == HealingStrategy.RESTART:
            action_type = f"restart_{component}"
        elif strategy == HealingStrategy.FAILOVER:
            action_type = f"failover_{component}"
        elif strategy == HealingStrategy.SCALE:
            action_type = f"scale_{component}"
        elif strategy == HealingStrategy.OPTIMIZE:
            action_type = f"optimize_{component}"
        elif strategy == HealingStrategy.DEGRADE:
            action_type = f"degrade_{component}"
        elif strategy == HealingStrategy.ISOLATE:
            action_type = f"isolate_{component}"
        elif strategy == HealingStrategy.BACKUP:
            action_type = f"backup_{component}"
        elif strategy == HealingStrategy.MANUAL_INTERVENTION:
            action_type = "manual_intervention"
        else:
            return None
        
        # Create action parameters based on component and strategy
        parameters = self._generate_action_parameters(strategy, component, issue_analysis)
        
        # Determine prerequisites
        prerequisites = self._determine_prerequisites(strategy, component)
        
        # Create rollback action if needed
        rollback_action = self._create_rollback_action(strategy, component, parameters)
        
        return HealingAction(
            type=action_type,
            component=component,
            parameters=parameters,
            expected_duration=strategy_info["time"],
            success_probability=score,
            risk_level=strategy_info["risk"],
            prerequisites=prerequisites,
            rollback_action=rollback_action
        )
    
    def _generate_action_parameters(self, strategy: HealingStrategy, component: str, 
                                   issue_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for a healing action."""
        parameters = {}
        
        if strategy == HealingStrategy.SCALE:
            # Determine scale factor based on severity
            severity = issue_analysis["severity"]
            if severity == IssueSeverity.CRITICAL:
                parameters["scale_factor"] = 2.0
            elif severity == IssueSeverity.HIGH:
                parameters["scale_factor"] = 1.5
            else:
                parameters["scale_factor"] = 1.2
            
            parameters["max_instances"] = 10
            parameters["min_instances"] = 1
        
        elif strategy == HealingStrategy.RESTART:
            parameters["timeout"] = 60
            parameters["force_restart"] = issue_analysis["severity"] == IssueSeverity.CRITICAL
        
        elif strategy == HealingStrategy.FAILOVER:
            parameters["switch_immediately"] = issue_analysis["urgency"] == "immediate"
            parameters["validate_backup"] = True
        
        elif strategy == HealingStrategy.OPTIMIZE:
            urgency = issue_analysis["urgency"]
            if urgency == "immediate":
                parameters["optimization_level"] = "aggressive"
            else:
                parameters["optimization_level"] = "conservative"
        
        elif strategy == HealingStrategy.DEGRADE:
            # Determine degradation level based on severity
            severity = issue_analysis["severity"]
            if severity == IssueSeverity.CRITICAL:
                parameters["degradation_level"] = "minimal"
            else:
                parameters["degradation_level"] = "reduced"
        
        return parameters
    
    def _determine_prerequisites(self, strategy: HealingStrategy, component: str) -> List[str]:
        """Determine prerequisites for a healing action."""
        prerequisites = []
        
        if strategy == HealingStrategy.FAILOVER:
            prerequisites.append(f"backup_{component}_available")
        
        elif strategy == HealingStrategy.SCALE:
            prerequisites.append("scaling_capacity_available")
            prerequisites.append("resource_limits_not_exceeded")
        
        elif strategy == HealingStrategy.BACKUP:
            prerequisites.append(f"backup_{component}_validated")
        
        elif strategy == HealingStrategy.ISOLATE:
            prerequisites.append("isolation_capabilities_available")
        
        return prerequisites
    
    def _create_rollback_action(self, strategy: HealingStrategy, component: str, 
                               parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create rollback action for a healing action."""
        if strategy == HealingStrategy.SCALE:
            return {
                "type": f"scale_down_{component}",
                "parameters": {
                    "scale_factor": 1.0 / parameters.get("scale_factor", 1.0)
                }
            }
        
        elif strategy == HealingStrategy.DEGRADE:
            return {
                "type": f"restore_quality_{component}",
                "parameters": {}
            }
        
        elif strategy == HealingStrategy.FAILOVER:
            return {
                "type": f"failback_{component}",
                "parameters": {}
            }
        
        return None
    
    def _create_healing_plan(self, issue: Dict[str, Any], issue_analysis: Dict[str, Any], 
                           healing_actions: List[HealingAction]) -> HealingPlan:
        """Create complete healing plan."""
        primary_actions = [action for action in healing_actions if not action.type.startswith("fallback_")]
        fallback_actions = [action for action in healing_actions if action.type.startswith("fallback_")]
        
        # Calculate estimated recovery time
        estimated_recovery_time = sum(action.expected_duration for action in primary_actions)
        
        # Calculate confidence score based on success probabilities
        if primary_actions:
            confidence_score = np.mean([action.success_probability for action in primary_actions])
        else:
            confidence_score = 0.0
        
        # Determine dependencies
        dependencies = []
        component = issue_analysis["component"]
        if component == "model":
            dependencies.extend(["data", "pipeline"])
        elif component == "data":
            dependencies.append("pipeline")
        
        return HealingPlan(
            issue_id=issue.get("id", f"{component}_{datetime.now().timestamp()}"),
            severity=issue_analysis["severity"],
            category=issue_analysis["category"],
            primary_actions=primary_actions,
            fallback_actions=fallback_actions,
            estimated_recovery_time=estimated_recovery_time,
            confidence_score=confidence_score,
            created_at=datetime.now(),
            dependencies=dependencies
        )
    
    def _plan_to_action_list(self, plan: HealingPlan) -> List[Dict[str, Any]]:
        """Convert healing plan to action list format."""
        actions = []
        
        for action in plan.primary_actions:
            actions.append({
                "type": action.type,
                "component": action.component,
                "parameters": action.parameters,
                "expected_duration": action.expected_duration,
                "success_probability": action.success_probability,
                "risk_level": action.risk_level,
                "prerequisites": action.prerequisites,
                "rollback_action": action.rollback_action,
                "plan_id": plan.issue_id,
                "action_priority": "primary"
            })
        
        for action in plan.fallback_actions:
            actions.append({
                "type": action.type,
                "component": action.component,
                "parameters": action.parameters,
                "expected_duration": action.expected_duration,
                "success_probability": action.success_probability,
                "risk_level": action.risk_level,
                "prerequisites": action.prerequisites,
                "rollback_action": action.rollback_action,
                "plan_id": plan.issue_id,
                "action_priority": "fallback"
            })
        
        return actions
    
    def record_healing_outcome(self, issue: Dict[str, Any], action: Dict[str, Any], 
                              success: bool, duration: float, details: Dict[str, Any] = None) -> None:
        """Record outcome of a healing action for learning."""
        if not self.learning_enabled:
            return
        
        issue_key = f"{issue.get('component')}_{issue.get('type')}"
        
        outcome_record = {
            "action_type": action.get("type"),
            "success": success,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        if success:
            if issue_key not in self.success_history:
                self.success_history[issue_key] = []
            self.success_history[issue_key].append(outcome_record)
            
            # Limit history size
            if len(self.success_history[issue_key]) > 100:
                self.success_history[issue_key] = self.success_history[issue_key][-100:]
        
        else:
            if issue_key not in self.failure_patterns:
                self.failure_patterns[issue_key] = []
            self.failure_patterns[issue_key].append(outcome_record)
            
            # Limit history size
            if len(self.failure_patterns[issue_key]) > 100:
                self.failure_patterns[issue_key] = self.failure_patterns[issue_key][-100:]
        
        # Update decision weights based on learning
        self._update_decision_weights(issue_key)
    
    def _update_decision_weights(self, issue_key: str) -> None:
        """Update decision weights based on learning from outcomes."""
        success_count = len(self.success_history.get(issue_key, []))
        failure_count = len(self.failure_patterns.get(issue_key, []))
        total_count = success_count + failure_count
        
        if total_count >= self.min_learning_samples:
            success_rate = success_count / total_count
            
            # Adjust weights based on success rate
            if success_rate > 0.8:
                # High success rate - rely more on historical data
                self.decision_weights["success_probability"] = min(0.3, self.decision_weights["success_probability"] + 0.05)
            elif success_rate < 0.4:
                # Low success rate - rely more on severity and impact
                self.decision_weights["severity"] = min(0.4, self.decision_weights["severity"] + 0.05)
                self.decision_weights["impact"] = min(0.3, self.decision_weights["impact"] + 0.05)
        
        # Normalize weights
        total_weight = sum(self.decision_weights.values())
        for key in self.decision_weights:
            self.decision_weights[key] /= total_weight
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning system."""
        total_successes = sum(len(history) for history in self.success_history.values())
        total_failures = sum(len(patterns) for patterns in self.failure_patterns.values())
        total_cases = total_successes + total_failures
        
        return {
            "learning_enabled": self.learning_enabled,
            "total_cases": total_cases,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "overall_success_rate": total_successes / max(total_cases, 1),
            "issue_types_learned": len(set(list(self.success_history.keys()) + list(self.failure_patterns.keys()))),
            "decision_weights": self.decision_weights.copy(),
            "min_learning_samples": self.min_learning_samples
        }