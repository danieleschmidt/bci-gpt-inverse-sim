"""Central Pipeline Orchestrator for BCI-GPT Self-Healing System.

Coordinates all processing stages, manages dependencies, and orchestrates
healing actions across the entire BCI pipeline.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from ..utils.error_handling import BCI_GPTError
from ..utils.reliability import CircuitBreakerState
from ..utils.monitoring import HealthStatus


class PipelineStage(Enum):
    """Pipeline processing stages."""
    DATA_INGESTION = "data_ingestion"
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    MODEL_INFERENCE = "model_inference"
    POST_PROCESSING = "post_processing"
    OUTPUT_DELIVERY = "output_delivery"


class PipelineState(Enum):
    """Overall pipeline states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    name: str
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    health_check_interval: float = 10.0
    critical: bool = True  # Whether failure affects overall pipeline


@dataclass
class StageStatus:
    """Status information for a pipeline stage."""
    stage: str
    state: str
    health: HealthStatus
    last_execution: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    average_duration: float = 0.0
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED


class PipelineOrchestrator:
    """Central orchestrator for the BCI-GPT processing pipeline.
    
    Manages stage execution, dependencies, health monitoring, and coordinates
    self-healing actions across all pipeline components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Core state
        self.stages: Dict[str, StageConfig] = {}
        self.stage_handlers: Dict[str, Callable] = {}
        self.stage_status: Dict[str, StageStatus] = {}
        self.pipeline_state = PipelineState.INITIALIZING
        
        # Execution control
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.active_executions: Dict[str, Future] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Monitoring and healing
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.healing_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Metrics
        self.total_executions = 0
        self.total_failures = 0
        self.pipeline_start_time = datetime.now()
        
        self._initialize_default_stages()
    
    def _initialize_default_stages(self) -> None:
        """Initialize default BCI pipeline stages."""
        default_stages = {
            PipelineStage.DATA_INGESTION.value: StageConfig(
                name="Data Ingestion",
                dependencies=[],
                timeout=10.0,
                critical=True
            ),
            PipelineStage.PREPROCESSING.value: StageConfig(
                name="EEG Preprocessing", 
                dependencies=[PipelineStage.DATA_INGESTION.value],
                timeout=15.0,
                critical=True
            ),
            PipelineStage.FEATURE_EXTRACTION.value: StageConfig(
                name="Feature Extraction",
                dependencies=[PipelineStage.PREPROCESSING.value],
                timeout=20.0,
                critical=True
            ),
            PipelineStage.MODEL_INFERENCE.value: StageConfig(
                name="Model Inference",
                dependencies=[PipelineStage.FEATURE_EXTRACTION.value],
                timeout=25.0,
                critical=True
            ),
            PipelineStage.POST_PROCESSING.value: StageConfig(
                name="Post Processing",
                dependencies=[PipelineStage.MODEL_INFERENCE.value],
                timeout=10.0,
                critical=False
            ),
            PipelineStage.OUTPUT_DELIVERY.value: StageConfig(
                name="Output Delivery",
                dependencies=[PipelineStage.POST_PROCESSING.value],
                timeout=5.0,
                critical=False
            )
        }
        
        for stage_id, config in default_stages.items():
            self.register_stage(stage_id, config)
    
    def register_stage(self, stage_id: str, config: StageConfig) -> None:
        """Register a pipeline stage with configuration."""
        self.stages[stage_id] = config
        self.stage_status[stage_id] = StageStatus(
            stage=stage_id,
            state="initialized",
            health=HealthStatus.UNKNOWN
        )
        
        # Build dependency graph
        self.dependency_graph[stage_id] = set(config.dependencies)
        
        self.logger.info(f"Registered pipeline stage: {stage_id}")
    
    def register_stage_handler(self, stage_id: str, handler: Callable) -> None:
        """Register a handler function for a specific stage."""
        if stage_id not in self.stages:
            raise BCI_GPTError(f"Stage {stage_id} not registered")
        
        self.stage_handlers[stage_id] = handler
        self.logger.info(f"Registered handler for stage: {stage_id}")
    
    def register_healing_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """Register a callback for healing events."""
        self.healing_callbacks.append(callback)
    
    async def execute_pipeline(self, input_data: Any) -> Any:
        """Execute the complete pipeline with input data."""
        self.total_executions += 1
        execution_id = f"exec_{self.total_executions}_{datetime.now().timestamp()}"
        
        self.logger.info(f"Starting pipeline execution: {execution_id}")
        
        try:
            # Execute stages in dependency order
            execution_results = {}
            
            for stage_id in self._get_execution_order():
                try:
                    # Check if dependencies are satisfied
                    if not self._are_dependencies_satisfied(stage_id, execution_results):
                        raise BCI_GPTError(f"Dependencies not satisfied for stage {stage_id}")
                    
                    # Execute stage
                    stage_input = self._prepare_stage_input(stage_id, input_data, execution_results)
                    result = await self._execute_stage(stage_id, stage_input)
                    execution_results[stage_id] = result
                    
                    self._update_stage_success(stage_id)
                    
                except Exception as e:
                    self._update_stage_failure(stage_id, e)
                    
                    # Check if stage is critical
                    if self.stages[stage_id].critical:
                        self.logger.error(f"Critical stage {stage_id} failed: {e}")
                        await self._trigger_healing(stage_id, {"error": str(e), "execution_id": execution_id})
                        raise
                    else:
                        self.logger.warning(f"Non-critical stage {stage_id} failed: {e}")
                        execution_results[stage_id] = None
            
            self._update_pipeline_state()
            return execution_results
            
        except Exception as e:
            self.total_failures += 1
            self.logger.error(f"Pipeline execution {execution_id} failed: {e}")
            await self._trigger_healing("pipeline", {"error": str(e), "execution_id": execution_id})
            raise
    
    def _get_execution_order(self) -> List[str]:
        """Get the correct execution order based on dependencies."""
        # Topological sort
        in_degree = {stage: len(deps) for stage, deps in self.dependency_graph.items()}
        queue = [stage for stage, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees of dependent stages
            for stage, deps in self.dependency_graph.items():
                if current in deps:
                    in_degree[stage] -= 1
                    if in_degree[stage] == 0:
                        queue.append(stage)
        
        if len(result) != len(self.stages):
            raise BCI_GPTError("Circular dependency detected in pipeline stages")
        
        return result
    
    def _are_dependencies_satisfied(self, stage_id: str, execution_results: Dict[str, Any]) -> bool:
        """Check if all dependencies for a stage are satisfied."""
        dependencies = self.dependency_graph.get(stage_id, set())
        return all(dep in execution_results for dep in dependencies)
    
    def _prepare_stage_input(self, stage_id: str, original_input: Any, execution_results: Dict[str, Any]) -> Any:
        """Prepare input data for a specific stage."""
        # For now, pass the original input and results
        # This can be enhanced with stage-specific input preparation
        return {
            "original_input": original_input,
            "execution_results": execution_results,
            "stage_id": stage_id
        }
    
    async def _execute_stage(self, stage_id: str, stage_input: Any) -> Any:
        """Execute a specific pipeline stage."""
        if stage_id not in self.stage_handlers:
            raise BCI_GPTError(f"No handler registered for stage {stage_id}")
        
        config = self.stages[stage_id]
        handler = self.stage_handlers[stage_id]
        
        self.logger.debug(f"Executing stage: {stage_id}")
        
        start_time = datetime.now()
        
        try:
            # Execute with timeout
            future = self.executor.submit(handler, stage_input)
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: future.result(timeout=config.timeout)
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            self._update_stage_metrics(stage_id, duration, True)
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._update_stage_metrics(stage_id, duration, False)
            raise
    
    def _update_stage_success(self, stage_id: str) -> None:
        """Update stage status after successful execution."""
        status = self.stage_status[stage_id]
        status.last_success = datetime.now()
        status.last_execution = datetime.now()
        status.success_count += 1
        status.state = "healthy"
        status.health = HealthStatus.HEALTHY
    
    def _update_stage_failure(self, stage_id: str, error: Exception) -> None:
        """Update stage status after failed execution."""
        status = self.stage_status[stage_id]
        status.last_failure = datetime.now()
        status.last_execution = datetime.now()
        status.failure_count += 1
        status.state = "failed"
        status.health = HealthStatus.UNHEALTHY
    
    def _update_stage_metrics(self, stage_id: str, duration: float, success: bool) -> None:
        """Update stage performance metrics."""
        status = self.stage_status[stage_id]
        
        # Update average duration using exponential moving average
        if status.average_duration == 0:
            status.average_duration = duration
        else:
            status.average_duration = 0.9 * status.average_duration + 0.1 * duration
    
    def _update_pipeline_state(self) -> None:
        """Update overall pipeline state based on stage statuses."""
        unhealthy_critical = 0
        total_critical = 0
        
        for stage_id, config in self.stages.items():
            if config.critical:
                total_critical += 1
                status = self.stage_status[stage_id]
                if status.health != HealthStatus.HEALTHY:
                    unhealthy_critical += 1
        
        if unhealthy_critical == 0:
            self.pipeline_state = PipelineState.HEALTHY
        elif unhealthy_critical < total_critical * 0.5:
            self.pipeline_state = PipelineState.DEGRADED
        else:
            self.pipeline_state = PipelineState.CRITICAL
    
    async def _trigger_healing(self, component: str, context: Dict[str, Any]) -> None:
        """Trigger healing actions for a failed component."""
        self.logger.info(f"Triggering healing for component: {component}")
        
        # Notify all registered healing callbacks
        for callback in self.healing_callbacks:
            try:
                callback(component, context)
            except Exception as e:
                self.logger.error(f"Healing callback failed: {e}")
    
    def start_monitoring(self) -> None:
        """Start continuous pipeline monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Pipeline monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop pipeline monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Pipeline monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_stage_health()
                self._update_pipeline_state()
                asyncio.run(self._handle_unhealthy_stages())
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
            
            # Sleep between checks
            threading.Event().wait(5.0)
    
    def _check_stage_health(self) -> None:
        """Check health of all pipeline stages."""
        current_time = datetime.now()
        
        for stage_id, status in self.stage_status.items():
            # Check if stage hasn't executed recently
            if status.last_execution:
                time_since_execution = (current_time - status.last_execution).total_seconds()
                config = self.stages[stage_id]
                
                # Mark as unhealthy if no execution within 2x the expected interval
                if time_since_execution > config.timeout * 2:
                    status.health = HealthStatus.UNHEALTHY
                    status.state = "stale"
    
    async def _handle_unhealthy_stages(self) -> None:
        """Handle unhealthy stages."""
        for stage_id, status in self.stage_status.items():
            if status.health == HealthStatus.UNHEALTHY:
                await self._trigger_healing(stage_id, {
                    "reason": "health_check_failed",
                    "last_execution": status.last_execution,
                    "failure_count": status.failure_count
                })
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        uptime = (datetime.now() - self.pipeline_start_time).total_seconds()
        
        return {
            "pipeline_state": self.pipeline_state.value,
            "uptime_seconds": uptime,
            "total_executions": self.total_executions,
            "total_failures": self.total_failures,
            "success_rate": (self.total_executions - self.total_failures) / max(self.total_executions, 1),
            "stages": {
                stage_id: {
                    "state": status.state,
                    "health": status.health.value,
                    "success_count": status.success_count,
                    "failure_count": status.failure_count,
                    "average_duration": status.average_duration,
                    "last_execution": status.last_execution.isoformat() if status.last_execution else None,
                    "last_success": status.last_success.isoformat() if status.last_success else None,
                    "last_failure": status.last_failure.isoformat() if status.last_failure else None
                }
                for stage_id, status in self.stage_status.items()
            }
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        self.logger.info("Shutting down pipeline orchestrator")
        
        self.stop_monitoring()
        
        # Cancel active executions
        for execution_id, future in self.active_executions.items():
            future.cancel()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        self.logger.info("Pipeline orchestrator shutdown complete")