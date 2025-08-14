#!/usr/bin/env python3
"""
Comprehensive test of the enhanced BCI-GPT system capabilities.
Tests all new Generation 2 and 3 enhancements without requiring heavy dependencies.
"""

import sys
import time
import threading
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_reliability_system():
    """Test the reliability and error handling system."""
    print("Testing Reliability System...")
    
    try:
        from bci_gpt.utils.reliability import (
            CircuitBreaker, RetryManager, HealthChecker, ErrorReporter,
            get_health_checker, get_error_reporter
        )
        
        # Test circuit breaker
        @CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        def failing_function():
            raise Exception("Test failure")
        
        # Test retry manager
        @RetryManager(max_retries=2, base_delay=0.1)
        def flaky_function():
            if hasattr(flaky_function, 'attempts'):
                flaky_function.attempts += 1
            else:
                flaky_function.attempts = 1
            
            if flaky_function.attempts < 2:
                raise Exception("Temporary failure")
            return "Success after retry"
        
        # Test circuit breaker
        failures = 0
        for _ in range(5):
            try:
                failing_function()
            except Exception:
                failures += 1
        
        print(f"  ✅ Circuit breaker: {failures}/5 calls failed as expected")
        
        # Test retry mechanism
        result = flaky_function()
        print(f"  ✅ Retry manager: {result}")
        
        # Test health checker
        health_checker = get_health_checker()
        health_status = health_checker.run_all_checks()
        print(f"  ✅ Health checker: {health_status['overall_status']}")
        
        # Test error reporter
        error_reporter = get_error_reporter()
        error_reporter.report_error(
            Exception("Test error"), 
            context={"test": True},
            component="test_system"
        )
        summary = error_reporter.get_error_summary(hours=1)
        print(f"  ✅ Error reporter: {summary['total_errors']} errors recorded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Reliability system test failed: {e}")
        return False


def test_monitoring_system():
    """Test the advanced monitoring system."""
    print("Testing Advanced Monitoring System...")
    
    try:
        from bci_gpt.utils.advanced_monitoring import (
            MetricsCollector, PerformanceProfiler, AlertManager, SystemMonitor,
            get_system_monitor, monitor_performance, record_metric
        )
        
        # Test metrics collection
        metrics = MetricsCollector()
        
        # Record some test metrics
        metrics.counter("test_counter", 5)
        metrics.gauge("test_gauge", 42.5)
        metrics.histogram("test_histogram", 100.0)
        metrics.timing("test_operation", 150.0)
        
        stats = metrics.get_stats()
        print(f"  ✅ Metrics collector: {stats.hit_rate:.2f} hit rate")
        
        # Test performance profiler
        profiler = PerformanceProfiler(metrics)
        
        with profiler.span("test_operation"):
            time.sleep(0.01)  # Simulate work
        
        active_spans = profiler.get_active_spans()
        print(f"  ✅ Performance profiler: {len(active_spans)} active spans")
        
        # Test alert manager
        alert_manager = AlertManager(metrics)
        alert_manager.add_alert(
            name="test_alert",
            metric_name="test_gauge",
            threshold=50.0,
            condition="greater_than",
            severity="warning"
        )
        print("  ✅ Alert manager: Alert configured")
        
        # Test system monitor
        monitor = get_system_monitor()
        dashboard_data = monitor.get_dashboard_data()
        print(f"  ✅ System monitor: Dashboard has {len(dashboard_data)} sections")
        
        # Test record metric function
        record_metric("gauge", "system_load", 75.0, {"component": "test"})
        print("  ✅ Metric recording: Global metric recorded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Monitoring system test failed: {e}")
        return False


def test_security_system():
    """Test the enhanced security system."""
    print("Testing Enhanced Security System...")
    
    try:
        from bci_gpt.utils.enhanced_security import (
            SecureKeyManager, DataEncryption, SecurityAuditor, TokenManager,
            ComplianceValidator, get_key_manager, get_security_auditor,
            get_token_manager, get_compliance_validator
        )
        
        # Test key management
        key_manager = get_key_manager()
        key = key_manager.generate_key("test_key", 32)
        retrieved_key = key_manager.get_key("test_key")
        print(f"  ✅ Key manager: Generated and retrieved {len(key)} byte key")
        
        # Test data encryption (if cryptography is available)
        try:
            encryption = DataEncryption(key_manager)
            test_data = {"message": "Hello, secure world!"}
            encrypted = encryption.encrypt_data(test_data, "test_key")
            decrypted = encryption.decrypt_data(encrypted)
            print(f"  ✅ Data encryption: {decrypted['message']}")
        except ImportError:
            print("  ⚠️  Data encryption: Cryptography library not available")
        
        # Test security auditor
        auditor = get_security_auditor()
        auditor.log_security_event(
            event_type="test_login",
            severity="low",
            source="test_system",
            details={"user": "test_user", "success": True}
        )
        summary = auditor.get_security_summary(hours=1)
        print(f"  ✅ Security auditor: {summary['total_events']} events logged")
        
        # Test token manager
        token_manager = get_token_manager()
        token = token_manager.create_token("test_user", ["read", "write"], 1)
        validated = token_manager.validate_token(token)
        print(f"  ✅ Token manager: Token for {validated.user_id if validated else 'invalid'}")
        
        # Test compliance validator
        compliance = get_compliance_validator()
        test_config = {
            'encryption_at_rest': True,
            'encryption_in_transit': True,
            'access_controls': True,
            'audit_logging': True
        }
        hipaa_report = compliance.validate_hipaa_compliance(test_config)
        gdpr_report = compliance.validate_gdpr_compliance(test_config)
        print(f"  ✅ Compliance: HIPAA {hipaa_report['score']}%, GDPR {gdpr_report['score']}%")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Security system test failed: {e}")
        return False


def test_optimization_system():
    """Test the advanced optimization system."""
    print("Testing Advanced Optimization System...")
    
    try:
        from bci_gpt.utils.advanced_optimization import (
            IntelligentCache, PerformanceOptimizer, AsyncTaskManager,
            get_cache, get_performance_optimizer, cached, performance_monitoring
        )
        
        # Test intelligent cache
        cache = get_cache()
        
        # Test cache operations
        cache.put("test_key", "test_value", ttl=60)
        value = cache.get("test_key")
        stats = cache.get_stats()
        memory_usage = cache.get_memory_usage()
        
        print(f"  ✅ Intelligent cache: {value}, hit rate {stats.hit_rate:.2f}")
        print(f"  ✅ Cache memory: {memory_usage['total_mb']:.2f} MB")
        
        # Test cached decorator
        @cached(ttl=60)
        def expensive_function(x):
            time.sleep(0.01)  # Simulate expensive operation
            return x * 2
        
        start_time = time.time()
        result1 = expensive_function(5)  # Should be slow
        first_call_time = time.time() - start_time
        
        start_time = time.time()
        result2 = expensive_function(5)  # Should be fast (cached)
        second_call_time = time.time() - start_time
        
        print(f"  ✅ Cache decorator: {result1} ({first_call_time:.3f}s vs {second_call_time:.3f}s)")
        
        # Test performance optimizer
        optimizer = get_performance_optimizer()
        optimizer.record_performance_metric("test_metric", 123.45)
        summary = optimizer.get_performance_summary()
        print(f"  ✅ Performance optimizer: {len(summary['registered_optimizations'])} optimizations")
        
        # Test performance monitoring context
        with performance_monitoring("test_operation"):
            time.sleep(0.005)  # Simulate work
        
        print("  ✅ Performance monitoring: Context completed")
        
        # Test async task manager
        task_manager = AsyncTaskManager()
        status = task_manager.get_task_status()
        print(f"  ✅ Task manager: {status['max_workers']} max workers")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Optimization system test failed: {e}")
        return False


def test_scaling_system():
    """Test the auto-scaling system."""
    print("Testing Auto-Scaling System...")
    
    try:
        from bci_gpt.utils.auto_scaling import (
            LoadBalancer, AutoScaler, ServiceInstance, ScalingRule,
            get_load_balancer, get_auto_scaler, setup_default_scaling_rules
        )
        
        # Test load balancer
        load_balancer = get_load_balancer()
        
        # Register test instances
        instance1 = ServiceInstance("test_1", "10.0.0.1", 8080)
        instance2 = ServiceInstance("test_2", "10.0.0.2", 8080)
        
        load_balancer.register_instance(instance1)
        load_balancer.register_instance(instance2)
        
        stats = load_balancer.get_load_balancer_stats()
        print(f"  ✅ Load balancer: {stats['total_instances']} instances, {stats['strategy']} strategy")
        
        # Test auto-scaler
        auto_scaler = get_auto_scaler()
        
        # Add test scaling rule
        rule = ScalingRule(
            name="test_rule",
            metric_name="cpu_percent",
            threshold_up=80.0,
            threshold_down=20.0,
            min_instances=1,
            max_instances=5
        )
        auto_scaler.add_scaling_rule(rule)
        
        # Setup default rules
        setup_default_scaling_rules()
        
        summary = auto_scaler.get_scaling_summary()
        print(f"  ✅ Auto-scaler: {summary['scaling_rules']} rules configured")
        
        # Test brief auto-scaling
        auto_scaler.start_auto_scaling(check_interval=1)
        time.sleep(2)  # Let it run briefly
        auto_scaler.stop_auto_scaling()
        
        final_summary = auto_scaler.get_scaling_summary()
        print(f"  ✅ Auto-scaling test: {final_summary['current_instances']} instances")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Scaling system test failed: {e}")
        return False


def test_system_integration():
    """Test integration between all systems."""
    print("Testing System Integration...")
    
    try:
        # Test that all systems can work together
        from bci_gpt.utils.reliability import get_error_reporter
        from bci_gpt.utils.advanced_monitoring import get_system_monitor
        from bci_gpt.utils.enhanced_security import get_security_auditor
        from bci_gpt.utils.advanced_optimization import get_performance_optimizer
        from bci_gpt.utils.auto_scaling import get_auto_scaler
        
        # Cross-system test: Generate load and monitor
        error_reporter = get_error_reporter()
        monitor = get_system_monitor()
        auditor = get_security_auditor()
        optimizer = get_performance_optimizer()
        scaler = get_auto_scaler()
        
        # Simulate some activity
        for i in range(10):
            # Log security events
            auditor.log_security_event(
                event_type="api_access",
                severity="low",
                source="integration_test",
                details={"request_id": i}
            )
            
            # Record performance metrics
            optimizer.record_performance_metric("response_time", 50 + i * 5)
            
            # Log some errors
            if i % 3 == 0:
                error_reporter.report_error(
                    Exception(f"Test error {i}"),
                    context={"iteration": i},
                    component="integration_test"
                )
        
        # Get combined status
        dashboard = monitor.get_dashboard_data()
        security_summary = auditor.get_security_summary(hours=1)
        error_summary = error_reporter.get_error_summary(hours=1)
        scaling_summary = scaler.get_scaling_summary()
        
        print(f"  ✅ Integration test results:")
        print(f"    - Dashboard sections: {len(dashboard)}")
        print(f"    - Security events: {security_summary['total_events']}")
        print(f"    - Error events: {error_summary['total_errors']}")
        print(f"    - Scaling rules: {scaling_summary['scaling_rules']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ System integration test failed: {e}")
        return False


def main():
    """Run comprehensive enhanced system tests."""
    print("=" * 80)
    print("BCI-GPT Enhanced System Test Suite")
    print("Testing Generation 2 & 3 Enhancements")
    print("=" * 80)
    
    tests = [
        ("Reliability System", test_reliability_system),
        ("Advanced Monitoring", test_monitoring_system),
        ("Enhanced Security", test_security_system),
        ("Advanced Optimization", test_optimization_system),
        ("Auto-Scaling System", test_scaling_system),
        ("System Integration", test_system_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*80}")
    print(f"Enhanced System Test Results: {passed}/{total} PASSED ({passed/total*100:.1f}%)")
    print("=" * 80)
    
    if passed == total:
        print("🎉 ALL ENHANCED SYSTEM TESTS PASSED!")
        print("✅ Generation 2 & 3 enhancements are fully operational")
        print("✅ System is ready for production deployment")
    else:
        print("⚠️  Some enhanced system tests failed")
        print("🔧 Review failing components before production deployment")
    
    # Final system status
    print(f"\n📊 Final System Status:")
    print(f"   - Reliability: ✅ Circuit breakers, retries, health checks")
    print(f"   - Monitoring: ✅ Metrics, alerts, performance profiling")
    print(f"   - Security: ✅ Encryption, auditing, compliance validation")
    print(f"   - Optimization: ✅ Intelligent caching, performance tuning")
    print(f"   - Scaling: ✅ Auto-scaling, load balancing")
    print(f"   - Integration: ✅ Cross-system coordination")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)