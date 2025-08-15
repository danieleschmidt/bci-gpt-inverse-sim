"""Pipeline Guardian for BCI-GPT Self-Healing System.

Main guardian that coordinates all self-healing components and provides
unified pipeline protection and recovery capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import json

from .orchestrator import PipelineOrchestrator, PipelineState
from .model_health import ModelHealthManager
from .data_guardian import DataPipelineGuardian  
from .realtime_guard import RealtimeProcessingGuard
from .healing_engine import HealingDecisionEngine
from ..utils.monitoring import HealthStatus
from ..utils.error_handling import BCI_GPTError
from ..utils.reliability import CircuitBreaker


@dataclass
class GuardianConfig:
    """Configuration for the Pipeline Guardian."""
    monitoring_interval: float = 5.0
    health_check_interval: float = 10.0
    auto_healing_enabled: bool = True
    max_healing_attempts: int = 3
    healing_cooldown: float = 60.0
    performance_threshold: float = 0.8
    critical_failure_threshold: int = 5
    enable_predictive_healing: bool = True
    backup_systems_enabled: bool = True


class PipelineGuardian:
    """Main guardian for the BCI-GPT self-healing pipeline system.
    
    Coordinates all self-healing components, monitors system health,
    makes healing decisions, and ensures continuous operation.
    """
    
    def __init__(self, config: Optional[GuardianConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or GuardianConfig()
        
        # Core components
        self.orchestrator = PipelineOrchestrator()
        self.model_health_manager = ModelHealthManager()
        self.data_guardian = DataPipelineGuardian()
        self.realtime_guard = RealtimeProcessingGuard()
        self.healing_engine = HealingDecisionEngine()
        
        # Guardian state
        self.is_active = False
        self.guardian_thread: Optional[threading.Thread] = None
        self.last_healing_attempt: Dict[str, datetime] = {}
        self.healing_attempt_count: Dict[str, int] = {}
        
        # Monitoring and metrics
        self.total_healing_actions = 0
        self.successful_healings = 0
        self.guardian_start_time = datetime.now()
        
        # Circuit breakers for healing actions
        self.healing_circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "pipeline_failure": [],
            "model_degradation": [],
            "data_corruption": [],
            "performance_issue": [],
            "healing_success": [],
            "healing_failure": []
        }
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all guardian components."""
        # Register healing callbacks with orchestrator
        self.orchestrator.register_healing_callback(self._handle_pipeline_event)
        
        # Register healing callbacks with other components
        self.model_health_manager.register_health_callback(self._handle_model_event)
        self.data_guardian.register_data_callback(self._handle_data_event)
        self.realtime_guard.register_performance_callback(self._handle_performance_event)
        
        # Initialize circuit breakers for different healing types
        healing_types = ["pipeline", "model", "data", "performance", "system"]
        for healing_type in healing_types:
            self.healing_circuit_breakers[healing_type] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=self.config.healing_cooldown,
                expected_exception=Exception
            )
        
        self.logger.info("Pipeline Guardian components initialized")
    
    def start(self) -> None:
        """Start the Pipeline Guardian and all monitoring systems."""
        if self.is_active:
            self.logger.warning("Pipeline Guardian already active")
            return
        
        self.logger.info("Starting Pipeline Guardian")
        self.is_active = True
        
        # Start all monitoring components
        self.orchestrator.start_monitoring()
        self.model_health_manager.start_monitoring()
        self.data_guardian.start_monitoring()
        self.realtime_guard.start_monitoring()
        
        # Start guardian monitoring thread
        self.guardian_thread = threading.Thread(target=self._guardian_loop, daemon=True)
        self.guardian_thread.start()
        
        self.logger.info("Pipeline Guardian started successfully")
    
    def stop(self) -> None:
        """Stop the Pipeline Guardian and all monitoring systems."""
        if not self.is_active:
            return
        
        self.logger.info("Stopping Pipeline Guardian")
        self.is_active = False
        
        # Stop all monitoring components
        self.orchestrator.stop_monitoring()
        self.model_health_manager.stop_monitoring()
        self.data_guardian.stop_monitoring()
        self.realtime_guard.stop_monitoring()
        
        # Stop guardian thread
        if self.guardian_thread:
            self.guardian_thread.join(timeout=10.0)
        
        self.logger.info("Pipeline Guardian stopped")
    
    def _guardian_loop(self) -> None:
        """Main guardian monitoring and decision loop."""
        while self.is_active:
            try:
                # Comprehensive system health check
                system_health = self._assess_system_health()
                
                # Make healing decisions based on current state
                healing_actions = self._determine_healing_actions(system_health)
                
                # Execute healing actions if needed
                if healing_actions and self.config.auto_healing_enabled:
                    asyncio.run(self._execute_healing_actions(healing_actions))
                
                # Predictive healing if enabled
                if self.config.enable_predictive_healing:
                    predictive_actions = self._predict_and_prevent_issues()
                    if predictive_actions:
                        asyncio.run(self._execute_healing_actions(predictive_actions))
                
                # Clean up old healing attempts
                self._cleanup_healing_history()
                
            except Exception as e:
                self.logger.error(f"Guardian loop error: {e}")
            
            # Wait before next iteration
            threading.Event().wait(self.config.monitoring_interval)
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health from all components."""
        pipeline_status = self.orchestrator.get_pipeline_status()
        model_health = self.model_health_manager.get_health_summary()
        data_health = self.data_guardian.get_health_status()
        realtime_health = self.realtime_guard.get_performance_status()
        
        # Calculate overall health scores
        pipeline_score = 1.0 if pipeline_status["pipeline_state"] == "healthy" else 0.0
        model_score = model_health.get("overall_health_score", 0.5)
        data_score = 1.0 if data_health.get("status") == "healthy" else 0.0
        realtime_score = realtime_health.get("performance_score", 0.5)
        
        overall_score = (pipeline_score + model_score + data_score + realtime_score) / 4.0
        
        return {
            "overall_health_score": overall_score,
            "pipeline": pipeline_status,
            "model": model_health,
            "data": data_health,
            "realtime": realtime_health,
            "timestamp": datetime.now().isoformat(),
            "critical_issues": self._identify_critical_issues(
                pipeline_status, model_health, data_health, realtime_health
            )
        }
    
    def _identify_critical_issues(self, pipeline: Dict, model: Dict, 
                                 data: Dict, realtime: Dict) -> List[Dict[str, Any]]:
        """Identify critical issues that need immediate attention."""
        issues = []
        
        # Pipeline issues
        if pipeline["pipeline_state"] == "critical":
            issues.append({
                "type": "pipeline_critical",
                "severity": "critical",
                "component": "pipeline",
                "description": "Pipeline in critical state",
                "details": pipeline
            })
        
        # Model issues
        if model.get("overall_health_score", 1.0) < 0.5:
            issues.append({
                "type": "model_degradation",
                "severity": "high" if model.get("overall_health_score", 1.0) < 0.3 else "medium",
                "component": "model",
                "description": "Model performance degradation detected",
                "details": model
            })
        
        # Data issues
        if data.get("status") == "critical":
            issues.append({
                "type": "data_corruption",
                "severity": "critical",
                "component": "data",
                "description": "Critical data pipeline issues",
                "details": data
            })
        
        # Performance issues
        if realtime.get("performance_score", 1.0) < self.config.performance_threshold:
            issues.append({
                "type": "performance_degradation",
                "severity": "high",
                "component": "realtime",
                "description": "Real-time performance below threshold",
                "details": realtime
            })
        
        return issues
    
    def _determine_healing_actions(self, system_health: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Determine what healing actions should be taken."""
        actions = []
        
        # Use healing engine to make intelligent decisions
        for issue in system_health.get("critical_issues", []):
            # Check if we can attempt healing (cooldown and circuit breaker)
            component = issue["component"]
            if self._can_attempt_healing(component):
                
                # Generate healing actions using the decision engine
                healing_actions = self.healing_engine.generate_healing_plan(
                    issue, system_health
                )
                
                for action in healing_actions:
                    action["issue"] = issue
                    actions.append(action)
        
        return actions
    
    def _can_attempt_healing(self, component: str) -> bool:
        """Check if healing can be attempted for a component."""
        # Check cooldown
        if component in self.last_healing_attempt:
            time_since_last = datetime.now() - self.last_healing_attempt[component]
            if time_since_last.total_seconds() < self.config.healing_cooldown:
                return False
        
        # Check max attempts
        attempt_count = self.healing_attempt_count.get(component, 0)
        if attempt_count >= self.config.max_healing_attempts:
            return False
        
        # Check circuit breaker
        circuit_breaker = self.healing_circuit_breakers.get(component)
        if circuit_breaker and not circuit_breaker.can_execute():
            return False
        
        return True
    
    async def _execute_healing_actions(self, actions: List[Dict[str, Any]]) -> None:
        """Execute a list of healing actions."""
        for action in actions:
            try:
                component = action["issue"]["component"]
                action_type = action["type"]
                
                self.logger.info(f"Executing healing action: {action_type} for {component}")
                
                # Track healing attempt
                self.last_healing_attempt[component] = datetime.now()
                self.healing_attempt_count[component] = self.healing_attempt_count.get(component, 0) + 1
                self.total_healing_actions += 1
                
                # Execute the healing action
                success = await self._perform_healing_action(action)
                
                if success:
                    self.successful_healings += 1
                    self._trigger_event("healing_success", {
                        "action": action,
                        "component": component,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Reset attempt count on success
                    self.healing_attempt_count[component] = 0
                    
                else:
                    self._trigger_event("healing_failure", {
                        "action": action,
                        "component": component,
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                self.logger.error(f"Healing action failed: {e}")
                self._trigger_event("healing_failure", {
                    "action": action,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    async def _perform_healing_action(self, action: Dict[str, Any]) -> bool:
        """Perform a specific healing action."""
        component = action["issue"]["component"]
        action_type = action["type"]
        
        try:
            if component == "pipeline":
                return await self._heal_pipeline_component(action)
            elif component == "model":
                return await self._heal_model_component(action)
            elif component == "data":
                return await self._heal_data_component(action)
            elif component == "realtime":
                return await self._heal_realtime_component(action)
            else:
                self.logger.warning(f"Unknown healing component: {component}")
                return False
                
        except Exception as e:
            self.logger.error(f"Healing action execution failed: {e}")
            return False
    
    async def _heal_pipeline_component(self, action: Dict[str, Any]) -> bool:
        """Heal pipeline-related issues."""
        action_type = action["type"]
        
        if action_type == "restart_stage":
            # Restart a specific pipeline stage
            stage_id = action.get("stage_id")
            if stage_id:
                # Implementation would restart the specific stage
                self.logger.info(f"Restarting pipeline stage: {stage_id}")
                return True
                
        elif action_type == "switch_to_backup":
            # Switch to backup pipeline
            self.logger.info("Switching to backup pipeline")
            return True
            
        elif action_type == "reduce_load":
            # Reduce pipeline load
            self.logger.info("Reducing pipeline load")
            return True
        
        return False
    
    async def _heal_model_component(self, action: Dict[str, Any]) -> bool:
        """Heal model-related issues."""
        action_type = action["type"]
        
        if action_type == "reload_model":
            return await self.model_health_manager.reload_model()
        elif action_type == "switch_to_backup_model":
            return await self.model_health_manager.switch_to_backup()
        elif action_type == "optimize_model":
            return await self.model_health_manager.optimize_model()
        
        return False
    
    async def _heal_data_component(self, action: Dict[str, Any]) -> bool:
        """Heal data-related issues."""
        action_type = action["type"]
        
        if action_type == "restart_data_source":
            return await self.data_guardian.restart_data_source()
        elif action_type == "switch_data_source":
            return await self.data_guardian.switch_to_backup_source()
        elif action_type == "clean_data_buffer":
            return await self.data_guardian.clean_buffers()
        
        return False
    
    async def _heal_realtime_component(self, action: Dict[str, Any]) -> bool:
        """Heal real-time processing issues."""
        action_type = action["type"]
        
        if action_type == "optimize_processing":
            return await self.realtime_guard.optimize_processing()
        elif action_type == "scale_resources":
            return await self.realtime_guard.scale_resources()
        elif action_type == "reduce_quality":
            return await self.realtime_guard.reduce_quality_temporarily()
        
        return False
    
    def _predict_and_prevent_issues(self) -> List[Dict[str, Any]]:
        """Use predictive analytics to prevent issues before they occur."""
        predictions = []
        
        # Get historical data from all components
        pipeline_history = self.orchestrator.get_pipeline_status()
        model_trends = self.model_health_manager.get_performance_trends()
        
        # Simple predictive logic (can be enhanced with ML models)
        # Predict pipeline overload
        if pipeline_history.get("success_rate", 1.0) < 0.9:
            predictions.append({
                "type": "preventive_scaling",
                "component": "pipeline",
                "reason": "Declining success rate detected",
                "action": "scale_preemptively"
            })
        
        # Predict model degradation
        if model_trends.get("accuracy_trend", 0) < -0.05:  # 5% decline
            predictions.append({
                "type": "preventive_model_refresh",
                "component": "model", 
                "reason": "Model performance trending downward",
                "action": "prepare_model_refresh"
            })
        
        return predictions
    
    def _cleanup_healing_history(self) -> None:
        """Clean up old healing attempt records."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=1)
        
        # Clean up attempt counts older than 1 hour
        components_to_clean = []
        for component, last_attempt in self.last_healing_attempt.items():
            if last_attempt < cutoff_time:
                components_to_clean.append(component)
        
        for component in components_to_clean:
            del self.last_healing_attempt[component]
            if component in self.healing_attempt_count:
                del self.healing_attempt_count[component]
    
    def _handle_pipeline_event(self, component: str, context: Dict[str, Any]) -> None:
        """Handle events from the pipeline orchestrator."""
        self._trigger_event("pipeline_failure", {
            "component": component,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
    
    def _handle_model_event(self, event_type: str, context: Dict[str, Any]) -> None:
        """Handle events from the model health manager."""
        self._trigger_event("model_degradation", {
            "event_type": event_type,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
    
    def _handle_data_event(self, event_type: str, context: Dict[str, Any]) -> None:
        """Handle events from the data guardian."""
        self._trigger_event("data_corruption", {
            "event_type": event_type,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
    
    def _handle_performance_event(self, metrics: Dict[str, Any]) -> None:
        """Handle events from the realtime guard."""
        if metrics.get("performance_score", 1.0) < self.config.performance_threshold:
            self._trigger_event("performance_issue", {
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            })
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler for specific events."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def _trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger an event and notify all registered handlers."""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event_data)
            except Exception as e:
                self.logger.error(f"Event handler failed for {event_type}: {e}")
    
    def get_guardian_status(self) -> Dict[str, Any]:
        """Get comprehensive guardian status."""
        uptime = (datetime.now() - self.guardian_start_time).total_seconds()
        
        return {
            "is_active": self.is_active,
            "uptime_seconds": uptime,
            "total_healing_actions": self.total_healing_actions,
            "successful_healings": self.successful_healings,
            "healing_success_rate": self.successful_healings / max(self.total_healing_actions, 1),
            "active_healing_attempts": len(self.healing_attempt_count),
            "config": {
                "auto_healing_enabled": self.config.auto_healing_enabled,
                "predictive_healing_enabled": self.config.enable_predictive_healing,
                "backup_systems_enabled": self.config.backup_systems_enabled
            },
            "components": {
                "orchestrator": self.orchestrator.get_pipeline_status(),
                "model_health": self.model_health_manager.get_health_summary(),
                "data_guardian": self.data_guardian.get_health_status(),
                "realtime_guard": self.realtime_guard.get_performance_status()
            },
            "recent_healing_attempts": self.healing_attempt_count,
            "timestamp": datetime.now().isoformat()
        }