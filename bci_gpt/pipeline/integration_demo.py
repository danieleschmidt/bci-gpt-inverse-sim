"""Integration Demo for BCI-GPT Self-Healing Pipeline System.

Demonstrates the complete self-healing pipeline system with all components
working together to provide autonomous operation and recovery.
"""

import asyncio
import logging
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import all pipeline components
from .orchestrator import PipelineOrchestrator, StageConfig
from .guardian import PipelineGuardian, GuardianConfig
from .model_health import ModelHealthManager, ModelHealthConfig
from .data_guardian import DataPipelineGuardian, DataGuardianConfig, DataSource, DataSourceType
from .realtime_guard import RealtimeProcessingGuard, RealtimeGuardConfig
from .healing_engine import HealingDecisionEngine
from .security_guardian import SecurityGuardian, SecurityPolicy
from .compliance_monitor import ComplianceMonitor, ComplianceFramework, DataClassification
from .advanced_monitoring import AdvancedMonitoringSystem, MetricDefinition, MetricType
from .distributed_processing import DistributedProcessingEngine, TaskPriority


class BCIGPTSelfHealingDemo:
    """Complete BCI-GPT Self-Healing System Integration Demo."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.guardian = None
        self.distributed_engine = None
        self.monitoring_system = None
        
        self.demo_running = False
        self.start_time = datetime.now()
    
    async def initialize_system(self) -> None:
        """Initialize the complete self-healing system."""
        self.logger.info("🚀 Initializing BCI-GPT Self-Healing Pipeline System...")
        
        try:
            # Initialize Pipeline Guardian (integrates all healing components)
            config = GuardianConfig(
                monitoring_interval=2.0,
                auto_healing_enabled=True,
                enable_predictive_healing=True,
                backup_systems_enabled=True
            )
            self.guardian = PipelineGuardian(config)
            
            # Initialize Distributed Processing Engine
            self.distributed_engine = DistributedProcessingEngine()
            
            # Initialize Advanced Monitoring System
            self.monitoring_system = AdvancedMonitoringSystem()
            
            # Register custom task handlers
            self._register_task_handlers()
            
            # Configure data sources
            self._configure_data_sources()
            
            # Set up compliance and security
            self._configure_compliance_security()
            
            # Start all systems
            await self._start_all_systems()
            
            self.logger.info("✅ System initialization complete!")
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def _register_task_handlers(self) -> None:
        """Register custom task handlers for distributed processing."""
        
        def eeg_processing_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            """Handle EEG processing tasks."""
            self.logger.info(f"🧠 Processing EEG data: {len(payload.get('samples', []))} samples")
            
            # Simulate EEG processing
            time.sleep(0.3)
            
            return {
                "processed_samples": len(payload.get('samples', [])),
                "quality_score": 0.92,
                "artifacts_removed": 2,
                "processing_time": 0.3
            }
        
        def model_inference_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            """Handle model inference tasks."""
            self.logger.info("🤖 Running model inference")
            
            # Simulate model inference
            time.sleep(0.2)
            
            return {
                "predictions": [0.85, 0.12, 0.03],
                "confidence": 0.87,
                "inference_time": 0.2
            }
        
        def optimization_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            """Handle optimization tasks."""
            self.logger.info("⚡ Optimizing system performance")
            
            # Simulate optimization
            time.sleep(0.5)
            
            return {
                "optimization_applied": True,
                "performance_improvement": "15%",
                "new_config": {"batch_size": 32, "workers": 4}
            }
        
        # Register handlers
        self.distributed_engine.register_task_handler("eeg_processing", eeg_processing_handler)
        self.distributed_engine.register_task_handler("model_inference", model_inference_handler)
        self.distributed_engine.register_task_handler("optimization", optimization_handler)
    
    def _configure_data_sources(self) -> None:
        """Configure data sources for the system."""
        # Primary EEG source
        primary_source = DataSource(
            source_id="primary_eeg",
            source_type=DataSourceType.PRIMARY_EEG,
            connection_string="eeg://device/primary",
            priority=1
        )
        
        # Backup EEG source
        backup_source = DataSource(
            source_id="backup_eeg",
            source_type=DataSourceType.BACKUP_EEG,
            connection_string="eeg://device/backup",
            priority=2
        )
        
        # Register sources
        self.guardian.data_guardian.register_data_source(primary_source)
        self.guardian.data_guardian.register_data_source(backup_source)
        self.guardian.data_guardian.set_active_source("primary_eeg")
    
    def _configure_compliance_security(self) -> None:
        """Configure compliance and security settings."""
        # Record consent for demo
        consent_id = self.guardian.compliance_monitor.record_consent(
            subject_id="demo_subject_001",
            purpose="medical_research",
            data_types=["eeg", "demographic"],
            consent_method="electronic",
            informed=True
        )
        
        # Record data processing activity
        self.guardian.compliance_monitor.record_data_processing(
            data_type="eeg_signals",
            classification=DataClassification.CONFIDENTIAL,
            purpose="brain_computer_interface",
            legal_basis="consent",
            data_subject_id="demo_subject_001",
            processing_activity="signal_analysis",
            retention_period=365,
            processor_id="bci_gpt_system",
            consent_id=consent_id
        )
    
    async def _start_all_systems(self) -> None:
        """Start all system components."""
        self.logger.info("🔄 Starting all system components...")
        
        # Start guardian (includes orchestrator, health managers, etc.)
        self.guardian.start()
        
        # Start distributed processing
        self.distributed_engine.start_processing()
        
        # Start advanced monitoring
        self.monitoring_system.start_monitoring()
        
        # Allow systems to initialize
        await asyncio.sleep(2.0)
        
        self.logger.info("✅ All systems started successfully!")
    
    async def run_demo_scenarios(self) -> None:
        """Run various demo scenarios to show self-healing capabilities."""
        self.logger.info("🎬 Starting demo scenarios...")
        
        self.demo_running = True
        
        try:
            # Scenario 1: Normal Operation
            await self._demo_normal_operation()
            
            # Scenario 2: Model Degradation and Recovery
            await self._demo_model_degradation_recovery()
            
            # Scenario 3: Data Quality Issues
            await self._demo_data_quality_issues()
            
            # Scenario 4: Performance Bottlenecks
            await self._demo_performance_bottlenecks()
            
            # Scenario 5: Security Incident Response
            await self._demo_security_incident()
            
            # Scenario 6: Distributed Processing
            await self._demo_distributed_processing()
            
            # Scenario 7: Compliance Monitoring
            await self._demo_compliance_monitoring()
            
            self.logger.info("🎉 All demo scenarios completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Demo scenario failed: {e}")
        finally:
            self.demo_running = False
    
    async def _demo_normal_operation(self) -> None:
        """Demo normal system operation."""
        self.logger.info("\n📊 SCENARIO 1: Normal Operation")
        self.logger.info("-" * 50)
        
        # Simulate normal model performance
        for i in range(5):
            self.guardian.model_health_manager.record_prediction_metrics(
                accuracy=0.87 + (i * 0.01),
                latency_ms=45.0 + (i * 2),
                confidence=0.89 + (i * 0.005)
            )
            
            # Record monitoring metrics
            self.monitoring_system.record_metric("cpu_usage", 60.0 + (i * 2))
            self.monitoring_system.record_metric("memory_usage", 70.0 + (i * 1.5))
            self.monitoring_system.record_metric("pipeline_latency", 80.0 + (i * 3))
            
            await asyncio.sleep(0.5)
        
        # Show system status
        status = self.guardian.get_guardian_status()
        self.logger.info(f"✅ System Health Score: {status['components']['model_health']['overall_health_score']:.2f}")
        self.logger.info(f"✅ Active Healing Attempts: {status['active_healing_attempts']}")
        
        await asyncio.sleep(2.0)
    
    async def _demo_model_degradation_recovery(self) -> None:
        """Demo model degradation detection and recovery."""
        self.logger.info("\n🚨 SCENARIO 2: Model Degradation and Recovery")
        self.logger.info("-" * 50)
        
        # Simulate model degradation
        self.logger.info("🔄 Simulating model performance degradation...")
        
        for i in range(8):
            # Declining accuracy and increasing latency
            accuracy = 0.87 - (i * 0.08)  # Down to ~0.3
            latency = 45.0 + (i * 15)     # Up to ~150ms
            confidence = 0.89 - (i * 0.07)  # Down to ~0.3
            
            self.guardian.model_health_manager.record_prediction_metrics(
                accuracy=accuracy,
                latency_ms=latency,
                confidence=confidence
            )
            
            self.logger.info(f"  📉 Model metrics - Accuracy: {accuracy:.2f}, Latency: {latency:.1f}ms")
            await asyncio.sleep(0.5)
        
        # Wait for healing system to detect and respond
        self.logger.info("🤖 Healing system detecting degradation...")
        await asyncio.sleep(3.0)
        
        # Check if healing was triggered
        status = self.guardian.get_guardian_status()
        self.logger.info(f"🔧 Total Healing Actions: {status['total_healing_actions']}")
        
        # Simulate recovery after healing
        self.logger.info("💫 Simulating recovery after healing actions...")
        for i in range(5):
            accuracy = 0.4 + (i * 0.1)   # Recovery to ~0.8
            latency = 140.0 - (i * 15)   # Down to ~65ms
            confidence = 0.4 + (i * 0.09)  # Up to ~0.76
            
            self.guardian.model_health_manager.record_prediction_metrics(
                accuracy=accuracy,
                latency_ms=latency,
                confidence=confidence
            )
            
            await asyncio.sleep(0.5)
        
        model_status = self.guardian.model_health_manager.get_health_summary()
        self.logger.info(f"✅ Recovery Complete - Health Score: {model_status['overall_health_score']:.2f}")
        
        await asyncio.sleep(2.0)
    
    async def _demo_data_quality_issues(self) -> None:
        """Demo data quality monitoring and remediation."""
        self.logger.info("\n📡 SCENARIO 3: Data Quality Issues")
        self.logger.info("-" * 50)
        
        # Simulate data quality degradation
        self.logger.info("📊 Simulating data quality issues...")
        
        # Process some corrupt data samples
        corrupt_samples = [
            [999.0, -999.0, 999.0],  # Extreme values
            [0.0, 0.0, 0.0],         # Flat signal
            [float('inf'), 1.0, 2.0], # Invalid values
        ]
        
        for i, sample in enumerate(corrupt_samples):
            try:
                self.guardian.data_guardian.process_data_sample(sample, "primary_eeg")
                self.logger.info(f"  📈 Processed sample {i+1}: quality issues detected")
            except Exception as e:
                self.logger.warning(f"  ⚠️  Sample {i+1} rejected: {e}")
            
            await asyncio.sleep(0.5)
        
        # Check data health status
        data_status = self.guardian.data_guardian.get_health_status()
        self.logger.info(f"📊 Data Health Status: {data_status['status']}")
        
        # Simulate failover to backup source
        if data_status['status'] != 'healthy':
            self.logger.info("🔄 Initiating failover to backup data source...")
            await self.guardian.data_guardian.switch_to_backup_source()
            
            # Wait and check status
            await asyncio.sleep(2.0)
            new_status = self.guardian.data_guardian.get_health_status()
            self.logger.info(f"✅ Failover complete - Status: {new_status['status']}")
        
        await asyncio.sleep(2.0)
    
    async def _demo_performance_bottlenecks(self) -> None:
        """Demo performance monitoring and optimization."""
        self.logger.info("\n⚡ SCENARIO 4: Performance Bottlenecks")
        self.logger.info("-" * 50)
        
        # Simulate performance degradation
        self.logger.info("📈 Simulating performance bottlenecks...")
        
        for i in range(6):
            # Increasing latency and resource usage
            latency = 80.0 + (i * 30)    # Up to ~230ms
            cpu_usage = 60.0 + (i * 8)   # Up to ~100%
            memory_usage = 70.0 + (i * 6) # Up to ~100%
            
            self.guardian.realtime_guard.record_processing_latency(latency)
            self.monitoring_system.record_metric("cpu_usage", cpu_usage)
            self.monitoring_system.record_metric("memory_usage", memory_usage)
            
            self.logger.info(f"  📊 Performance - Latency: {latency:.1f}ms, CPU: {cpu_usage:.1f}%")
            await asyncio.sleep(0.5)
        
        # Wait for performance optimization
        self.logger.info("🔧 Performance optimization triggered...")
        await asyncio.sleep(2.0)
        
        # Check performance status
        perf_status = self.guardian.realtime_guard.get_performance_status()
        self.logger.info(f"⚡ Performance Score: {perf_status['performance_score']:.2f}")
        self.logger.info(f"🎛️  Quality Level: {perf_status['current_quality_level']}")
        
        await asyncio.sleep(2.0)
    
    async def _demo_security_incident(self) -> None:
        """Demo security incident detection and response."""
        self.logger.info("\n🔒 SCENARIO 5: Security Incident Response")
        self.logger.info("-" * 50)
        
        # Simulate brute force attack
        self.logger.info("🚨 Simulating brute force attack...")
        
        for i in range(6):
            result = self.guardian.security_guardian.authenticate_request(
                user_id=f"attacker_{i}",
                password="wrong_password",
                source_ip="192.168.1.100",
                component="system"
            )
            
            self.logger.info(f"  🔐 Auth attempt {i+1}: {result['reason'] if not result['success'] else 'success'}")
            await asyncio.sleep(0.3)
        
        # Check security status
        security_status = self.guardian.security_guardian.get_security_status()
        self.logger.info(f"🛡️  Security Events: {security_status['total_events']}")
        self.logger.info(f"🚫 Blocked IPs: {security_status['blocked_ips']}")
        self.logger.info(f"🔒 Threats Blocked: {security_status['total_threats_blocked']}")
        
        await asyncio.sleep(2.0)
    
    async def _demo_distributed_processing(self) -> None:
        """Demo distributed processing and load balancing."""
        self.logger.info("\n🌐 SCENARIO 6: Distributed Processing")
        self.logger.info("-" * 50)
        
        # Submit various tasks
        tasks = [
            ("eeg_processing", {"samples": list(range(1000))}),
            ("model_inference", {"input_data": [1, 2, 3, 4, 5]}),
            ("optimization", {"target": "latency"}),
            ("eeg_processing", {"samples": list(range(500))}),
            ("model_inference", {"input_data": [6, 7, 8, 9, 10]})
        ]
        
        submitted_tasks = []
        
        for task_type, payload in tasks:
            task_id = self.distributed_engine.submit_task(
                task_type=task_type,
                payload=payload,
                priority=TaskPriority.NORMAL
            )
            submitted_tasks.append(task_id)
            self.logger.info(f"📋 Submitted {task_type} task: {task_id[:8]}...")
            
            await asyncio.sleep(0.2)
        
        # Wait for tasks to process
        self.logger.info("⏳ Processing tasks...")
        await asyncio.sleep(3.0)
        
        # Check task statuses
        completed_tasks = 0
        for task_id in submitted_tasks:
            status = self.distributed_engine.get_task_status(task_id)
            if status and status['status'] == 'completed':
                completed_tasks += 1
        
        self.logger.info(f"✅ Completed {completed_tasks}/{len(submitted_tasks)} tasks")
        
        # Show cluster status
        cluster_status = self.distributed_engine.get_cluster_status()
        metrics = cluster_status['workload_metrics']
        self.logger.info(f"📊 Cluster Metrics - Total: {metrics['total_tasks']}, Success Rate: {(metrics['completed_tasks']/(metrics['completed_tasks']+metrics['failed_tasks']) if metrics['completed_tasks']+metrics['failed_tasks'] > 0 else 0):.1%}")
        
        await asyncio.sleep(2.0)
    
    async def _demo_compliance_monitoring(self) -> None:
        """Demo compliance monitoring and violation detection."""
        self.logger.info("\n📋 SCENARIO 7: Compliance Monitoring")
        self.logger.info("-" * 50)
        
        # Show current compliance status
        compliance_status = self.guardian.compliance_monitor.get_compliance_status()
        self.logger.info(f"📊 Enabled Frameworks: {compliance_status['enabled_frameworks']}")
        self.logger.info(f"✅ Total Compliance Checks: {compliance_status['total_checks']}")
        self.logger.info(f"⚠️  Total Violations: {compliance_status['total_violations']}")
        
        # Show recent violations if any
        violations = self.guardian.compliance_monitor.get_violations(limit=5)
        if violations:
            self.logger.info("🚨 Recent Compliance Violations:")
            for violation in violations:
                self.logger.info(f"  - {violation['framework']}: {violation['description']}")
        else:
            self.logger.info("✅ No recent compliance violations")
        
        # Show audit trail
        audit_entries = self.guardian.compliance_monitor.get_audit_trail(limit=3)
        self.logger.info("📝 Recent Audit Entries:")
        for entry in audit_entries:
            self.logger.info(f"  - {entry['event_type']}: {entry['details'].get('description', 'N/A')}")
        
        await asyncio.sleep(2.0)
    
    async def show_final_status(self) -> None:
        """Show final system status."""
        self.logger.info("\n📈 FINAL SYSTEM STATUS")
        self.logger.info("=" * 60)
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Guardian status
        guardian_status = self.guardian.get_guardian_status()
        self.logger.info(f"🛡️  Guardian Uptime: {uptime:.1f}s")
        self.logger.info(f"🔧 Total Healing Actions: {guardian_status['total_healing_actions']}")
        self.logger.info(f"✅ Successful Healings: {guardian_status['successful_healings']}")
        self.logger.info(f"📊 Healing Success Rate: {guardian_status['healing_success_rate']:.1%}")
        
        # Component statuses
        components = guardian_status['components']
        self.logger.info("\n📊 Component Health:")
        self.logger.info(f"  🧠 Model Health Score: {components['model_health']['overall_health_score']:.2f}")
        self.logger.info(f"  📡 Data Status: {components['data_guardian']['status']}")
        self.logger.info(f"  ⚡ Performance Score: {components['realtime_guard']['performance_score']:.2f}")
        
        # Security and compliance
        security_status = self.guardian.security_guardian.get_security_status()
        compliance_status = self.guardian.compliance_monitor.get_compliance_status()
        
        self.logger.info("\n🔒 Security & Compliance:")
        self.logger.info(f"  🛡️  Security Events: {security_status['total_events']}")
        self.logger.info(f"  🚫 Threats Blocked: {security_status['total_threats_blocked']}")
        self.logger.info(f"  📋 Compliance Checks: {compliance_status['total_checks']}")
        
        # Distributed processing
        cluster_status = self.distributed_engine.get_cluster_status()
        metrics = cluster_status['workload_metrics']
        self.logger.info("\n🌐 Distributed Processing:")
        self.logger.info(f"  📋 Total Tasks: {metrics['total_tasks']}")
        self.logger.info(f"  ✅ Completed: {metrics['completed_tasks']}")
        self.logger.info(f"  ❌ Failed: {metrics['failed_tasks']}")
        
        self.logger.info("\n🎉 BCI-GPT Self-Healing System Demo Complete!")
    
    async def shutdown_system(self) -> None:
        """Gracefully shutdown the system."""
        self.logger.info("\n🔄 Shutting down system...")
        
        try:
            # Stop monitoring
            if self.monitoring_system:
                self.monitoring_system.stop_monitoring()
            
            # Stop distributed processing
            if self.distributed_engine:
                self.distributed_engine.stop_processing()
            
            # Stop guardian
            if self.guardian:
                self.guardian.stop()
            
            self.logger.info("✅ System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")


async def main():
    """Main demo function."""
    print("🧠 BCI-GPT Self-Healing Pipeline System Demo")
    print("=" * 60)
    print()
    
    demo = BCIGPTSelfHealingDemo()
    
    try:
        # Initialize system
        await demo.initialize_system()
        
        # Run demo scenarios
        await demo.run_demo_scenarios()
        
        # Show final status
        await demo.show_final_status()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        logging.error(f"❌ Demo failed: {e}")
    finally:
        # Shutdown
        await demo.shutdown_system()
        print("\n👋 Thank you for exploring BCI-GPT Self-Healing System!")


if __name__ == "__main__":
    # Run the demo
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Failed to run demo: {e}")
        sys.exit(1)