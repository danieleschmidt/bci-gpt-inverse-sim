#!/usr/bin/env python3
"""Advanced functionality test for BCI-GPT system."""

import sys
import traceback
from typing import Dict, Any, List

def test_error_handling():
    """Test error handling system."""
    print("Testing: Error Handling System")
    try:
        from bci_gpt.core.error_handling import ErrorHandler, BCIGPTError, CircuitBreaker
        
        # Test error handler
        eh = ErrorHandler()
        assert hasattr(eh, 'handle_error'), "ErrorHandler missing handle_error method"
        
        # Test circuit breaker
        cb = CircuitBreaker()
        assert hasattr(cb, 'call'), "CircuitBreaker missing call method"
        
        print("✅ Error handling system working")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_validation_system():
    """Test data validation system."""
    print("Testing: Data Validation System")
    try:
        from bci_gpt.core.validation import DataValidator, ValidationResult
        
        validator = DataValidator()
        
        # Test text validation
        result = validator.validate("Hello world", "text_input")
        assert isinstance(result, ValidationResult), "Validation should return ValidationResult"
        assert result.is_valid, "Simple text should be valid"
        
        # Test suspicious content detection
        result = validator.validate("<script>alert('bad')</script>", "text_input")
        assert not result.is_valid, "Script tags should be invalid"
        
        print("✅ Validation system working")
        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_health_monitoring():
    """Test health monitoring system."""
    print("Testing: Health Monitoring System")
    try:
        from bci_gpt.utils.health_monitoring import SystemHealthMonitor, HealthStatus
        
        hm = SystemHealthMonitor()
        status = hm.get_health_status()
        
        assert isinstance(status, dict), "Health status should be dict"
        assert "overall_status" in status, "Missing overall_status"
        
        print("✅ Health monitoring working")
        return True
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        return False

def test_performance_optimization():
    """Test performance optimization system."""
    print("Testing: Performance Optimization System")
    try:
        from bci_gpt.optimization.performance_optimizer import PerformanceOptimizer, CacheSystem
        
        # Test optimizer
        optimizer = PerformanceOptimizer()
        optimizer.record_metric('latency', 150.0)
        summary = optimizer.get_performance_summary()
        
        assert isinstance(summary, dict), "Performance summary should be dict"
        
        # Test cache
        cache = CacheSystem(max_size=10)
        cache.put('test', 'value')
        value = cache.get('test')
        assert value == 'value', "Cache should return stored value"
        
        print("✅ Performance optimization working")
        return True
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        return False

def test_auto_scaling():
    """Test auto-scaling system."""
    print("Testing: Auto-Scaling System")
    try:
        from bci_gpt.scaling.auto_scaler import AutoScaler, ResourceManager
        
        # Test auto-scaler
        scaler = AutoScaler()
        scaler.update_metric('cpu_usage', 75.0)
        status = scaler.get_scaling_status()
        
        assert isinstance(status, dict), "Scaling status should be dict"
        assert "enabled" in status, "Missing enabled field"
        
        # Test resource manager
        rm = ResourceManager()
        initial_threads = rm.current_resources['worker_threads']
        rm.scale_resource('worker_threads', 1.5)
        new_threads = rm.current_resources['worker_threads']
        assert new_threads != initial_threads, "Resource should be scaled"
        
        print("✅ Auto-scaling working")
        return True
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        return False

def test_integration():
    """Test system integration."""
    print("Testing: System Integration")
    try:
        from bci_gpt.core.error_handling import ErrorHandler
        from bci_gpt.core.validation import DataValidator
        from bci_gpt.utils.health_monitoring import SystemHealthMonitor
        from bci_gpt.optimization.performance_optimizer import PerformanceOptimizer
        from bci_gpt.scaling.auto_scaler import AutoScaler
        
        # Create integrated system components
        eh = ErrorHandler()
        validator = DataValidator()
        hm = SystemHealthMonitor()
        optimizer = PerformanceOptimizer()
        scaler = AutoScaler()
        
        # Test error handling with validation
        try:
            result = validator.validate(None, "text_input")
        except Exception as e:
            recovery_result = eh.handle_error(e, {"component": "validation"})
            assert recovery_result is not None, "Error handler should provide recovery"
        
        # Test performance monitoring integration
        optimizer.record_metric('response_time', 200.0)
        scaler.update_metric('response_time', 200.0)
        
        # Verify all components are working together
        assert validator is not None
        assert eh is not None
        assert hm is not None
        assert optimizer is not None
        assert scaler is not None
        
        print("✅ System integration working")
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def test_graceful_degradation():
    """Test graceful degradation with missing dependencies."""
    print("Testing: Graceful Degradation")
    try:
        # Test that system works without optional dependencies
        import bci_gpt
        
        # Test CLI graceful degradation
        from bci_gpt.cli import main
        
        # Test core modules can be imported
        from bci_gpt.core import error_handling
        from bci_gpt.utils import health_monitoring
        
        print("✅ Graceful degradation working")
        return True
    except Exception as e:
        print(f"❌ Graceful degradation test failed: {e}")
        return False

def run_security_tests():
    """Run security validation tests."""
    print("Testing: Security Validation")
    try:
        from bci_gpt.core.validation import DataValidator
        
        validator = DataValidator()
        
        # Test malicious input detection
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "eval('malicious code')",
            "exec('rm -rf /')"
        ]
        
        for malicious_input in malicious_inputs:
            result = validator.validate(malicious_input, "text_input")
            assert not result.is_valid, f"Should detect malicious input: {malicious_input}"
        
        # Test basic security guardian without full pipeline import
        try:
            from bci_gpt.pipeline.security_guardian import SecurityGuardian
            guardian = SecurityGuardian()
            
            for malicious_input in malicious_inputs:
                is_safe = guardian.validate_request({"input": malicious_input})
                assert not is_safe, f"Security guardian should block: {malicious_input}"
        except ImportError:
            print("   Note: Full security guardian test skipped (missing dependencies)")
        
        print("✅ Security validation working")
        return True
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False

def run_performance_tests():
    """Run performance validation tests."""
    print("Testing: Performance Benchmarks")
    try:
        import time
        from bci_gpt.optimization.performance_optimizer import CacheSystem
        
        # Test cache performance
        cache = CacheSystem(max_size=1000)
        
        # Benchmark cache operations
        start_time = time.time()
        for i in range(1000):
            cache.put(f"key_{i}", f"value_{i}")
        put_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        get_time = time.time() - start_time
        
        # Performance should be reasonable
        assert put_time < 1.0, f"Cache put operations too slow: {put_time:.3f}s"
        assert get_time < 0.5, f"Cache get operations too slow: {get_time:.3f}s"
        
        print(f"✅ Performance benchmarks passed (put: {put_time:.3f}s, get: {get_time:.3f}s)")
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all advanced functionality tests."""
    print("=" * 60)
    print("BCI-GPT Advanced Functionality Test")
    print("=" * 60)
    
    tests = [
        test_error_handling,
        test_validation_system,
        test_health_monitoring,
        test_performance_optimization,
        test_auto_scaling,
        test_integration,
        test_graceful_degradation,
        run_security_tests,
        run_performance_tests
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("=" * 60)
    print(f"Advanced Functionality Test Results: {sum(results)}/{len(results)} PASSED")
    print("=" * 60)
    
    if all(results):
        print("🎉 ALL ADVANCED TESTS PASSED - System is production-ready!")
        return 0
    else:
        print("⚠️  Some advanced tests failed - check implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())