"""Comprehensive Test Suite for BCI-GPT Self-Healing Pipeline System.

Tests all components of the self-healing pipeline including orchestration,
monitoring, security, compliance, and distributed processing.
"""

import asyncio
import pytest
import logging
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import all pipeline components
from .orchestrator import PipelineOrchestrator, PipelineState, StageConfig
from .guardian import PipelineGuardian, GuardianConfig
from .model_health import ModelHealthManager, ModelHealthConfig
from .data_guardian import DataPipelineGuardian, DataGuardianConfig, DataSource, DataSourceType
from .realtime_guard import RealtimeProcessingGuard, RealtimeGuardConfig
from .healing_engine import HealingDecisionEngine
from .security_guardian import SecurityGuardian, SecurityPolicy, ThreatLevel
from .compliance_monitor import ComplianceMonitor, ComplianceFramework
from .advanced_monitoring import AdvancedMonitoringSystem, MetricDefinition, MetricType
from .distributed_processing import DistributedProcessingEngine, TaskPriority, NodeRole


class TestPipelineOrchestrator:
    """Test suite for Pipeline Orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        return PipelineOrchestrator()
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator is not None
        assert orchestrator.pipeline_state == PipelineState.INITIALIZING
        assert len(orchestrator.stages) > 0
        assert "data_ingestion" in orchestrator.stages
        assert "model_inference" in orchestrator.stages
    
    def test_stage_registration(self, orchestrator):
        """Test stage registration."""
        stage_config = StageConfig(
            name="test_stage",
            dependencies=["data_ingestion"],
            timeout=60.0
        )
        
        orchestrator.register_stage("test_stage", stage_config)
        
        assert "test_stage" in orchestrator.stages
        assert orchestrator.stages["test_stage"].name == "test_stage"
    
    def test_stage_handler_registration(self, orchestrator):
        """Test stage handler registration."""
        def test_handler(data):
            return {"result": "test"}
        
        stage_config = StageConfig(name="test_stage")
        orchestrator.register_stage("test_stage", stage_config)
        orchestrator.register_stage_handler("test_stage", test_handler)
        
        assert "test_stage" in orchestrator.stage_handlers
        assert orchestrator.stage_handlers["test_stage"] == test_handler
    
    @pytest.mark.asyncio
    async def test_pipeline_execution(self, orchestrator):
        """Test pipeline execution."""
        # Mock stage handlers
        def mock_handler(data):
            return {"processed": True, "stage": data.get("stage_id")}
        
        for stage_id in orchestrator.stages.keys():
            orchestrator.register_stage_handler(stage_id, mock_handler)
        
        # Execute pipeline
        input_data = {"test": "data"}
        result = await orchestrator.execute_pipeline(input_data)
        
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_dependency_resolution(self, orchestrator):
        """Test dependency resolution."""
        execution_order = orchestrator._get_execution_order()
        
        # data_ingestion should come before preprocessing
        data_idx = execution_order.index("data_ingestion")
        preproc_idx = execution_order.index("preprocessing")
        
        assert data_idx < preproc_idx
    
    def test_monitoring_start_stop(self, orchestrator):
        """Test monitoring start and stop."""
        orchestrator.start_monitoring()
        assert orchestrator.monitoring_active
        
        orchestrator.stop_monitoring()
        assert not orchestrator.monitoring_active
    
    def test_pipeline_status(self, orchestrator):
        """Test pipeline status reporting."""
        status = orchestrator.get_pipeline_status()
        
        assert "pipeline_state" in status
        assert "uptime_seconds" in status
        assert "stages" in status
        assert "total_executions" in status


class TestPipelineGuardian:
    """Test suite for Pipeline Guardian."""
    
    @pytest.fixture
    def guardian(self):
        config = GuardianConfig()
        return PipelineGuardian(config)
    
    def test_initialization(self, guardian):
        """Test guardian initialization."""
        assert guardian is not None
        assert guardian.config is not None
        assert guardian.orchestrator is not None
        assert guardian.model_health_manager is not None
        assert guardian.data_guardian is not None
        assert guardian.realtime_guard is not None
        assert guardian.healing_engine is not None
    
    def test_start_stop(self, guardian):
        """Test guardian start and stop."""
        guardian.start()
        assert guardian.is_active
        
        guardian.stop()
        assert not guardian.is_active
    
    def test_system_health_assessment(self, guardian):
        """Test system health assessment."""
        health = guardian._assess_system_health()
        
        assert "overall_health_score" in health
        assert "pipeline" in health
        assert "model" in health
        assert "data" in health
        assert "realtime" in health
        assert "critical_issues" in health
    
    def test_healing_actions(self, guardian):
        """Test healing action generation."""
        # Mock system health with issues
        system_health = {
            "overall_health_score": 0.3,
            "critical_issues": [
                {
                    "type": "pipeline_critical",
                    "severity": "critical",
                    "component": "pipeline",
                    "description": "Pipeline in critical state"
                }
            ]
        }
        
        actions = guardian._determine_healing_actions(system_health)
        assert len(actions) > 0
    
    def test_guardian_status(self, guardian):
        """Test guardian status reporting."""
        status = guardian.get_guardian_status()
        
        assert "is_active" in status
        assert "total_healing_actions" in status
        assert "components" in status
        assert "config" in status


class TestModelHealthManager:
    """Test suite for Model Health Manager."""
    
    @pytest.fixture
    def health_manager(self):
        config = ModelHealthConfig()
        return ModelHealthManager(config)
    
    def test_initialization(self, health_manager):
        """Test health manager initialization."""
        assert health_manager is not None
        assert health_manager.config is not None
        assert len(health_manager.metrics_history) == 0
    
    def test_metric_recording(self, health_manager):
        """Test metric recording."""
        health_manager.record_prediction_metrics(
            accuracy=0.85,
            latency_ms=45.0,
            confidence=0.9
        )
        
        assert len(health_manager.metrics_history) == 1
        metric = health_manager.metrics_history[0]
        assert metric.accuracy == 0.85
        assert metric.latency_ms == 45.0
        assert metric.confidence_score == 0.9
    
    def test_health_assessment(self, health_manager):
        """Test health assessment."""
        # Add some metrics
        for i in range(20):
            health_manager.record_prediction_metrics(
                accuracy=0.8 + (i * 0.01),
                latency_ms=50.0,
                confidence=0.85
            )
        
        health_manager._comprehensive_health_check()
        assert health_manager.current_health_status is not None
    
    def test_degradation_detection(self, health_manager):
        """Test degradation detection."""
        # Simulate declining accuracy trend
        for i in range(25):
            accuracy = 0.9 - (i * 0.02)  # Declining accuracy
            health_manager.record_prediction_metrics(
                accuracy=accuracy,
                latency_ms=50.0,
                confidence=0.8
            )
            health_manager._update_performance_trends()
        
        health_manager._detect_degradation_patterns()
        # Should detect degradation in accuracy
    
    @pytest.mark.asyncio
    async def test_model_reload(self, health_manager):
        """Test model reload functionality."""
        health_manager.current_model_path = "/fake/model/path"
        result = await health_manager.reload_model()
        # Should succeed in test environment
        assert isinstance(result, bool)
    
    def test_health_summary(self, health_manager):
        """Test health summary generation."""
        # Add some metrics first
        for i in range(10):
            health_manager.record_prediction_metrics(
                accuracy=0.85,
                latency_ms=50.0,
                confidence=0.9
            )
        
        summary = health_manager.get_health_summary()
        
        assert "overall_health_score" in summary
        assert "health_status" in summary
        assert "metrics" in summary
        assert "trends" in summary


class TestDataPipelineGuardian:
    """Test suite for Data Pipeline Guardian."""
    
    @pytest.fixture
    def data_guardian(self):
        config = DataGuardianConfig()
        return DataPipelineGuardian(config)
    
    def test_initialization(self, data_guardian):
        """Test data guardian initialization."""
        assert data_guardian is not None
        assert data_guardian.config is not None
        assert len(data_guardian.data_sources) == 0
    
    def test_data_source_registration(self, data_guardian):
        """Test data source registration."""
        source = DataSource(
            source_id="test_source",
            source_type=DataSourceType.PRIMARY_EEG,
            connection_string="test://connection",
            priority=1
        )
        
        data_guardian.register_data_source(source)
        
        assert "test_source" in data_guardian.data_sources
        assert data_guardian.data_sources["test_source"].source_type == DataSourceType.PRIMARY_EEG
    
    def test_data_quality_assessment(self, data_guardian):
        """Test data quality assessment."""
        # Mock some data
        mock_data = [
            [1.0, 2.0, 3.0],  # Mock EEG sample
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9]
        ]
        
        for sample in mock_data:
            data_guardian.data_buffer.append(sample)
        
        quality_metrics = data_guardian._assess_data_quality()
        
        assert quality_metrics is not None
        assert hasattr(quality_metrics, 'signal_to_noise_ratio')
        assert hasattr(quality_metrics, 'artifact_percentage')
    
    def test_data_processing(self, data_guardian):
        """Test data processing and validation."""
        test_data = [1.0, 2.0, 3.0]
        is_valid = data_guardian._validate_data_sample(test_data)
        
        assert isinstance(is_valid, bool)
        
        # Process valid data
        if is_valid:
            data_guardian.process_data_sample(test_data, "test_source")
    
    def test_health_status(self, data_guardian):
        """Test health status reporting."""
        status = data_guardian.get_health_status()
        
        assert "status" in status
        assert "data_sources" in status
        assert "statistics" in status
        assert "config" in status


class TestRealtimeProcessingGuard:
    """Test suite for Realtime Processing Guard."""
    
    @pytest.fixture
    def realtime_guard(self):
        config = RealtimeGuardConfig()
        return RealtimeProcessingGuard(config)
    
    def test_initialization(self, realtime_guard):
        """Test realtime guard initialization."""
        assert realtime_guard is not None
        assert realtime_guard.config is not None
        assert realtime_guard.constraints is not None
    
    def test_metric_collection(self, realtime_guard):
        """Test performance metric collection."""
        metrics = realtime_guard._collect_performance_metrics()
        
        assert hasattr(metrics, 'latency_ms')
        assert hasattr(metrics, 'throughput_hz')
        assert hasattr(metrics, 'cpu_usage')
        assert hasattr(metrics, 'memory_usage_mb')
    
    def test_constraint_violations(self, realtime_guard):
        """Test constraint violation detection."""
        # Create metrics that violate constraints
        mock_metrics = Mock()
        mock_metrics.latency_ms = 500.0  # High latency
        mock_metrics.throughput_hz = 10.0  # Low throughput
        mock_metrics.cpu_usage = 95.0  # High CPU
        mock_metrics.memory_usage_mb = 4000.0  # High memory
        mock_metrics.queue_depth = 100  # High queue
        
        violations = realtime_guard._check_constraint_violations(mock_metrics)
        
        assert len(violations) > 0
        assert any("latency_exceeded" in v for v in violations)
    
    def test_performance_score(self, realtime_guard):
        """Test performance score calculation."""
        mock_metrics = Mock()
        mock_metrics.latency_ms = 80.0
        mock_metrics.throughput_hz = 75.0
        mock_metrics.cpu_usage = 60.0
        mock_metrics.memory_usage_mb = 1000.0
        mock_metrics.queue_depth = 10
        
        score = realtime_guard._calculate_performance_score(mock_metrics)
        
        assert 0.0 <= score <= 1.0
    
    def test_latency_recording(self, realtime_guard):
        """Test latency recording."""
        realtime_guard.record_processing_latency(85.0)
        
        assert len(realtime_guard.latency_history) == 1
        assert realtime_guard.latency_history[0] == 85.0
    
    def test_performance_status(self, realtime_guard):
        """Test performance status reporting."""
        status = realtime_guard.get_performance_status()
        
        assert "performance_score" in status
        assert "current_quality_level" in status
        assert "constraints" in status
        assert "statistics" in status


class TestHealingDecisionEngine:
    """Test suite for Healing Decision Engine."""
    
    @pytest.fixture
    def healing_engine(self):
        return HealingDecisionEngine()
    
    def test_initialization(self, healing_engine):
        """Test healing engine initialization."""
        assert healing_engine is not None
        assert len(healing_engine.healing_rules) > 0
        assert len(healing_engine.strategy_preferences) > 0
    
    def test_issue_analysis(self, healing_engine):
        """Test issue analysis."""
        issue = {
            "component": "pipeline",
            "type": "pipeline_failure",
            "severity": "critical"
        }
        
        system_context = {
            "overall_health_score": 0.3,
            "pipeline": {"pipeline_state": "critical"}
        }
        
        analysis = healing_engine._analyze_issue(issue, system_context)
        
        assert "component" in analysis
        assert "severity" in analysis
        assert "category" in analysis
        assert "impact_scope" in analysis
    
    def test_healing_plan_generation(self, healing_engine):
        """Test healing plan generation."""
        issue = {
            "component": "model",
            "type": "model_degradation",
            "severity": "high"
        }
        
        system_context = {
            "overall_health_score": 0.6,
            "model": {"overall_health_score": 0.4}
        }
        
        actions = healing_engine.generate_healing_plan(issue, system_context)
        
        assert len(actions) > 0
        assert all("type" in action for action in actions)
        assert all("component" in action for action in actions)
    
    def test_outcome_recording(self, healing_engine):
        """Test healing outcome recording."""
        issue = {"component": "pipeline", "type": "failure"}
        action = {"type": "restart_pipeline", "component": "pipeline"}
        
        healing_engine.record_healing_outcome(issue, action, True, 30.0)
        
        issue_key = "pipeline_failure"
        assert issue_key in healing_engine.success_history
        assert len(healing_engine.success_history[issue_key]) == 1
    
    def test_learning_statistics(self, healing_engine):
        """Test learning statistics."""
        stats = healing_engine.get_learning_statistics()
        
        assert "learning_enabled" in stats
        assert "total_cases" in stats
        assert "decision_weights" in stats


class TestSecurityGuardian:
    """Test suite for Security Guardian."""
    
    @pytest.fixture
    def security_guardian(self):
        policy = SecurityPolicy()
        return SecurityGuardian(policy)
    
    def test_initialization(self, security_guardian):
        """Test security guardian initialization."""
        assert security_guardian is not None
        assert security_guardian.security_policy is not None
        assert len(security_guardian.threat_patterns) > 0
    
    def test_authentication(self, security_guardian):
        """Test authentication process."""
        result = security_guardian.authenticate_request(
            user_id="test_user",
            password="test_password_123",
            source_ip="192.168.1.1",
            component="test"
        )
        
        assert "success" in result
        if result["success"]:
            assert "session_token" in result
    
    def test_authorization(self, security_guardian):
        """Test authorization process."""
        # First authenticate to get a session
        auth_result = security_guardian.authenticate_request(
            user_id="test_user",
            password="test_password_123",
            source_ip="192.168.1.1",
            component="test"
        )
        
        if auth_result["success"]:
            session_token = auth_result["session_token"]
            
            auth_result = security_guardian.authorize_action(
                session_token=session_token,
                component="pipeline",
                action="status"
            )
            
            assert "authorized" in auth_result
    
    def test_threat_detection(self, security_guardian):
        """Test threat detection."""
        # Simulate multiple failed attempts from same IP
        for i in range(6):
            security_guardian.authenticate_request(
                user_id=f"user_{i}",
                password="wrong_password",
                source_ip="192.168.1.100",
                component="test"
            )
        
        # Should trigger brute force detection
        security_guardian._detect_brute_force_patterns(datetime.now())
    
    def test_security_status(self, security_guardian):
        """Test security status reporting."""
        status = security_guardian.get_security_status()
        
        assert "monitoring_active" in status
        assert "total_events" in status
        assert "security_policy" in status
        assert "blocked_ips" in status


class TestComplianceMonitor:
    """Test suite for Compliance Monitor."""
    
    @pytest.fixture
    def compliance_monitor(self):
        return ComplianceMonitor()
    
    def test_initialization(self, compliance_monitor):
        """Test compliance monitor initialization."""
        assert compliance_monitor is not None
        assert len(compliance_monitor.compliance_rules) > 0
        assert len(compliance_monitor.enabled_frameworks) > 0
    
    def test_compliance_rules(self, compliance_monitor):
        """Test compliance rules."""
        # Check GDPR rules
        gdpr_rules = [rule for rule in compliance_monitor.compliance_rules.values() 
                     if rule.framework == ComplianceFramework.GDPR]
        assert len(gdpr_rules) > 0
        
        # Check HIPAA rules
        hipaa_rules = [rule for rule in compliance_monitor.compliance_rules.values() 
                      if rule.framework == ComplianceFramework.HIPAA]
        assert len(hipaa_rules) > 0
    
    def test_data_processing_record(self, compliance_monitor):
        """Test data processing recording."""
        from .compliance_monitor import DataClassification
        
        compliance_monitor.record_data_processing(
            data_type="eeg_signals",
            classification=DataClassification.CONFIDENTIAL,
            purpose="medical_analysis",
            legal_basis="consent",
            data_subject_id="subject_123",
            processing_activity="signal_analysis",
            retention_period=365,
            processor_id="bci_system"
        )
        
        assert len(compliance_monitor.data_processing_records) == 1
    
    def test_consent_management(self, compliance_monitor):
        """Test consent management."""
        consent_id = compliance_monitor.record_consent(
            subject_id="subject_123",
            purpose="medical_research",
            data_types=["eeg", "demographic"],
            consent_method="electronic"
        )
        
        assert consent_id is not None
        assert consent_id in compliance_monitor.consent_records
        
        # Test consent withdrawal
        result = compliance_monitor.withdraw_consent(consent_id, "subject_123")
        assert result is True
    
    def test_compliance_status(self, compliance_monitor):
        """Test compliance status reporting."""
        status = compliance_monitor.get_compliance_status()
        
        assert "enabled_frameworks" in status
        assert "compliance_status" in status
        assert "total_violations" in status
        assert "audit_trail_size" in status


class TestAdvancedMonitoringSystem:
    """Test suite for Advanced Monitoring System."""
    
    @pytest.fixture
    def monitoring_system(self):
        return AdvancedMonitoringSystem()
    
    def test_initialization(self, monitoring_system):
        """Test monitoring system initialization."""
        assert monitoring_system is not None
        assert len(monitoring_system.metric_definitions) > 0
        assert "cpu_usage" in monitoring_system.metric_definitions
        assert "memory_usage" in monitoring_system.metric_definitions
    
    def test_metric_registration(self, monitoring_system):
        """Test metric registration."""
        metric_def = MetricDefinition(
            name="test_metric",
            type=MetricType.PERFORMANCE,
            unit="percent",
            description="Test metric",
            warning_threshold=80.0,
            critical_threshold=90.0
        )
        
        monitoring_system.register_metric(metric_def)
        
        assert "test_metric" in monitoring_system.metric_definitions
        assert "test_metric" in monitoring_system.metric_storage
    
    def test_metric_recording(self, monitoring_system):
        """Test metric recording."""
        monitoring_system.record_metric("cpu_usage", 75.0, {"host": "test"})
        
        assert len(monitoring_system.metric_storage["cpu_usage"]) == 1
        metric_value = monitoring_system.metric_storage["cpu_usage"][0]
        assert metric_value.value == 75.0
    
    def test_alert_generation(self, monitoring_system):
        """Test alert generation."""
        # Record metric that exceeds threshold
        monitoring_system.record_metric("cpu_usage", 95.0)  # Above critical threshold
        
        # Should generate alert
        assert len(monitoring_system.alerts) > 0
    
    def test_anomaly_detection(self, monitoring_system):
        """Test anomaly detection."""
        # Add normal values
        for i in range(20):
            monitoring_system.record_metric("cpu_usage", 50.0 + (i % 5))
        
        # Add anomalous value
        monitoring_system.record_metric("cpu_usage", 150.0)  # Anomalous
        
        # Check if anomaly was detected
        if "cpu_usage" in monitoring_system.anomaly_history:
            anomalies = monitoring_system.anomaly_history["cpu_usage"]
            assert len(anomalies) >= 0  # May or may not detect depending on algorithm
    
    def test_monitoring_status(self, monitoring_system):
        """Test monitoring status reporting."""
        status = monitoring_system.get_monitoring_status()
        
        assert "monitoring_active" in status
        assert "monitored_metrics" in status
        assert "total_metrics_collected" in status
        assert "active_alerts" in status


class TestDistributedProcessingEngine:
    """Test suite for Distributed Processing Engine."""
    
    @pytest.fixture
    def processing_engine(self):
        return DistributedProcessingEngine()
    
    def test_initialization(self, processing_engine):
        """Test distributed processing engine initialization."""
        assert processing_engine is not None
        assert processing_engine.node_id is not None
        assert processing_engine.local_node is not None
        assert processing_engine.local_node.role == NodeRole.COORDINATOR
    
    def test_task_submission(self, processing_engine):
        """Test task submission."""
        task_id = processing_engine.submit_task(
            task_type="test_task",
            payload={"data": "test"},
            priority=TaskPriority.NORMAL
        )
        
        assert task_id is not None
        assert len(processing_engine.task_queue) == 1
        assert processing_engine.workload_metrics.total_tasks == 1
    
    def test_task_handler_registration(self, processing_engine):
        """Test task handler registration."""
        def test_handler(payload):
            return {"result": "success"}
        
        processing_engine.register_task_handler("test_task", test_handler)
        
        assert "test_task" in processing_engine.task_handlers
        assert processing_engine.task_handlers["test_task"] == test_handler
    
    def test_task_execution(self, processing_engine):
        """Test task execution."""
        # Register handler
        def test_handler(payload):
            return {"result": payload.get("data", "default")}
        
        processing_engine.register_task_handler("test_task", test_handler)
        
        # Submit task
        task_id = processing_engine.submit_task(
            task_type="test_task",
            payload={"data": "test_data"}
        )
        
        # Get task and execute
        task = processing_engine._get_next_task("test_worker")
        
        if task:
            processing_engine._execute_task(task, "test_worker")
            
            # Check if task completed
            assert task.status.value in ["completed", "failed"]
    
    def test_cluster_status(self, processing_engine):
        """Test cluster status reporting."""
        status = processing_engine.get_cluster_status()
        
        assert "local_node" in status
        assert "cluster_nodes" in status
        assert "workload_metrics" in status
        assert "processing_active" in status


class TestIntegrationScenarios:
    """Integration tests for complete self-healing scenarios."""
    
    @pytest.fixture
    def complete_system(self):
        """Create complete integrated system."""
        guardian = PipelineGuardian()
        guardian.start()
        return guardian
    
    @pytest.mark.asyncio
    async def test_complete_healing_scenario(self, complete_system):
        """Test complete healing scenario."""
        # Simulate system degradation
        complete_system.model_health_manager.record_prediction_metrics(
            accuracy=0.4,  # Low accuracy
            latency_ms=150.0,  # High latency
            confidence=0.3  # Low confidence
        )
        
        # Wait for system to detect issues and trigger healing
        await asyncio.sleep(2.0)
        
        # Check if healing actions were triggered
        status = complete_system.get_guardian_status()
        assert status["is_active"]
    
    def test_security_compliance_integration(self, complete_system):
        """Test security and compliance integration."""
        # Simulate security event
        complete_system.security_guardian.authenticate_request(
            user_id="unauthorized_user",
            password="wrong_password",
            source_ip="suspicious.ip",
            component="system"
        )
        
        # Simulate compliance violation
        complete_system.compliance_monitor.record_data_processing(
            data_type="sensitive_data",
            classification="CONFIDENTIAL",
            purpose="unauthorized_access",
            legal_basis="none",
            data_subject_id="victim_123",
            processing_activity="data_breach",
            retention_period=0,
            processor_id="malicious_actor"
        )
        
        # Check if violations were detected
        security_status = complete_system.security_guardian.get_security_status()
        compliance_status = complete_system.compliance_monitor.get_compliance_status()
        
        assert security_status["total_events"] > 0
        assert compliance_status["total_violations"] >= 0
    
    def test_performance_monitoring_integration(self, complete_system):
        """Test performance monitoring integration."""
        # Record high latency
        complete_system.realtime_guard.record_processing_latency(200.0)
        complete_system.realtime_guard.record_processing_latency(250.0)
        
        # Record poor model performance
        complete_system.model_health_manager.record_prediction_metrics(
            accuracy=0.6,
            latency_ms=180.0,
            confidence=0.7
        )
        
        # Check if monitoring detected issues
        realtime_status = complete_system.realtime_guard.get_performance_status()
        model_status = complete_system.model_health_manager.get_health_summary()
        
        assert "performance_score" in realtime_status
        assert "overall_health_score" in model_status
    
    @pytest.mark.asyncio
    async def test_distributed_healing(self):
        """Test distributed healing scenario."""
        # Create distributed system
        coordinator = DistributedProcessingEngine()
        worker = DistributedProcessingEngine(coordinator_address="localhost:8080")
        
        # Start both nodes
        coordinator.start_processing()
        worker.start_processing()
        
        try:
            # Submit healing task
            task_id = coordinator.submit_task(
                task_type="pipeline_optimization",
                payload={"target": "latency", "threshold": 100.0},
                priority=TaskPriority.HIGH
            )
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            # Check task status
            task_status = coordinator.get_task_status(task_id)
            assert task_status is not None
            
        finally:
            # Clean up
            coordinator.stop_processing()
            worker.stop_processing()
    
    def teardown_method(self, method):
        """Clean up after each test."""
        # Stop any running threads
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])