#!/usr/bin/env python3
"""Test the robustness and reliability systems"""

import time
import sys
from pathlib import Path

def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("🔧 Testing Circuit Breaker System...")
    
    try:
        from bci_gpt.robustness.circuit_breaker import (
            CircuitBreaker, CircuitBreakerConfig, circuit_breaker
        )
        
        # Test basic circuit breaker
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=5.0)
        breaker = CircuitBreaker("test_breaker", config)
        
        # Test successful execution
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        print("✅ Circuit breaker successful execution")
        
        # Test decorator
        @circuit_breaker("test_decorator", failure_threshold=2)
        def decorated_func():
            return "decorated_success"
        
        result = decorated_func()
        assert result == "decorated_success"
        print("✅ Circuit breaker decorator")
        
        # Test statistics
        stats = breaker.get_stats()
        assert stats['total_requests'] > 0
        print("✅ Circuit breaker statistics")
        
        return True
        
    except Exception as e:
        print(f"❌ Circuit breaker test failed: {e}")
        return False

def test_retry_manager():
    """Test retry manager functionality"""
    print("\n🔄 Testing Retry Manager System...")
    
    try:
        from bci_gpt.robustness.retry_manager import (
            RetryManager, RetryConfig, RetryStrategy, retry
        )
        
        # Test basic retry manager
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        manager = RetryManager("test_retry", config)
        
        # Test successful execution
        def success_func():
            return "retry_success"
        
        result = manager.execute(success_func)
        assert result == "retry_success"
        print("✅ Retry manager successful execution")
        
        # Test decorator
        @retry("test_decorator_retry", max_attempts=2)
        def decorated_retry_func():
            return "decorated_retry_success"
        
        result = decorated_retry_func()
        assert result == "decorated_retry_success"
        print("✅ Retry manager decorator")
        
        # Test statistics
        stats = manager.get_stats()
        assert stats['total_attempts'] > 0
        print("✅ Retry manager statistics")
        
        return True
        
    except Exception as e:
        print(f"❌ Retry manager test failed: {e}")
        return False

def test_health_checker():
    """Test health checker functionality"""
    print("\n🏥 Testing Health Checker System...")
    
    try:
        from bci_gpt.robustness.health_checker import (
            HealthChecker, SystemResourceCheck, EEGProcessorCheck,
            ModelInferenceCheck, DataPipelineCheck, HealthStatus,
            create_default_health_checker
        )
        
        # Create health checker
        checker = HealthChecker()
        
        # Add checks
        checker.add_check(SystemResourceCheck())
        checker.add_check(EEGProcessorCheck())
        checker.add_check(ModelInferenceCheck())
        checker.add_check(DataPipelineCheck())
        print("✅ Health checks added")
        
        # Run checks
        results = checker.run_checks()
        assert len(results) == 4
        print("✅ Health checks executed")
        
        # Get system status
        status = checker.get_system_status()
        assert 'overall_status' in status
        assert 'checks' in status
        assert 'summary' in status
        print("✅ System status retrieved")
        
        # Test default health checker
        default_checker = create_default_health_checker()
        default_results = default_checker.run_checks()
        assert len(default_results) >= 4
        print("✅ Default health checker")
        
        return True
        
    except Exception as e:
        print(f"❌ Health checker test failed: {e}")
        return False

def test_integration():
    """Test integration between robustness components"""
    print("\n🔗 Testing Robustness Integration...")
    
    try:
        from bci_gpt.robustness.circuit_breaker import circuit_breaker
        from bci_gpt.robustness.retry_manager import retry
        from bci_gpt.robustness.health_checker import HealthChecker, SystemResourceCheck
        
        # Test combined circuit breaker and retry
        @circuit_breaker("integrated_test", failure_threshold=3)
        @retry("integrated_retry", max_attempts=2)
        def integrated_func():
            return "integrated_success"
        
        result = integrated_func()
        assert result == "integrated_success"
        print("✅ Circuit breaker + retry integration")
        
        # Test health monitoring with robustness features
        checker = HealthChecker()
        checker.add_check(SystemResourceCheck())
        
        # Simulate monitoring
        results = checker.run_checks()
        system_status = checker.get_system_status()
        
        assert len(results) > 0
        assert system_status['overall_status'] in ['healthy', 'degraded', 'unhealthy', 'unknown']
        print("✅ Health monitoring integration")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all robustness tests"""
    print("🛡️  BCI-GPT Robustness System Test")
    print("=" * 50)
    
    tests = [
        test_circuit_breaker,
        test_retry_manager,
        test_health_checker,
        test_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 ROBUSTNESS SYSTEM TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("🎉 GENERATION 2 (MAKE IT ROBUST): ✅ PASSED")
        print("   System reliability and error handling operational")
        return True
    elif success_rate >= 70:
        print("⚠️  GENERATION 2 (MAKE IT ROBUST): ⚠️  PARTIAL")
        print("   System has basic robustness but needs improvements")
        return False
    else:
        print("❌ GENERATION 2 (MAKE IT ROBUST): ❌ FAILED")
        print("   System robustness features not operational")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)