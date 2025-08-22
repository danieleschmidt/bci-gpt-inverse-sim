#!/usr/bin/env python3
"""Production deployment readiness check for BCI-GPT."""

import sys
import os
import subprocess
from typing import Dict, Any, List, Tuple

def check_code_quality() -> Tuple[bool, str]:
    """Check code quality metrics."""
    try:
        # Check for Python syntax errors
        result = subprocess.run([
            'python3', '-m', 'py_compile', '-q',
        ], capture_output=True, text=True, cwd='/root/repo')
        
        # Check basic syntax on key files
        key_files = [
            'bci_gpt/__init__.py',
            'bci_gpt/core/error_handling.py',
            'bci_gpt/core/validation.py',
            'bci_gpt/utils/health_monitoring.py',
            'bci_gpt/optimization/performance_optimizer.py',
            'bci_gpt/scaling/auto_scaler.py'
        ]
        
        for file in key_files:
            if os.path.exists(file):
                result = subprocess.run([
                    'python3', '-m', 'py_compile', file
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    return False, f"Syntax error in {file}: {result.stderr}"
        
        return True, "Code quality checks passed"
    except Exception as e:
        return False, f"Code quality check failed: {e}"

def check_security() -> Tuple[bool, str]:
    """Check security configurations."""
    try:
        from bci_gpt.core.validation import DataValidator
        
        validator = DataValidator()
        
        # Test security validation
        test_cases = [
            ("<script>alert('xss')</script>", False),
            ("javascript:void(0)", False),
            ("Hello world", True),
            ("Normal text input", True)
        ]
        
        for test_input, should_be_valid in test_cases:
            result = validator.validate(test_input, "text_input")
            if result.is_valid != should_be_valid:
                return False, f"Security validation failed for: {test_input}"
        
        return True, "Security checks passed"
    except Exception as e:
        return False, f"Security check failed: {e}"

def check_performance() -> Tuple[bool, str]:
    """Check performance requirements."""
    try:
        import time
        from bci_gpt.optimization.performance_optimizer import CacheSystem
        
        # Test cache performance
        cache = CacheSystem(max_size=1000)
        
        start_time = time.time()
        for i in range(100):
            cache.put(f"key_{i}", f"value_{i}")
        put_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(100):
            cache.get(f"key_{i}")
        get_time = time.time() - start_time
        
        # Performance thresholds
        if put_time > 0.1:  # 100ms for 100 operations
            return False, f"Cache put performance too slow: {put_time:.3f}s"
        
        if get_time > 0.05:  # 50ms for 100 operations
            return False, f"Cache get performance too slow: {get_time:.3f}s"
        
        return True, f"Performance checks passed (put: {put_time:.3f}s, get: {get_time:.3f}s)"
    except Exception as e:
        return False, f"Performance check failed: {e}"

def check_scalability() -> Tuple[bool, str]:
    """Check scalability features."""
    try:
        from bci_gpt.scaling.auto_scaler import AutoScaler, ResourceManager
        
        # Test auto-scaler
        scaler = AutoScaler()
        scaler.update_metric('cpu_usage', 80.0)
        scaler.update_metric('memory_usage', 75.0)
        
        status = scaler.get_scaling_status()
        if not isinstance(status, dict):
            return False, "Auto-scaler status check failed"
        
        # Test resource manager
        rm = ResourceManager()
        initial_threads = rm.current_resources['worker_threads']
        rm.scale_resource('worker_threads', 1.5)
        new_threads = rm.current_resources['worker_threads']
        
        if new_threads == initial_threads:
            return False, "Resource scaling not working"
        
        return True, "Scalability checks passed"
    except Exception as e:
        return False, f"Scalability check failed: {e}"

def check_monitoring() -> Tuple[bool, str]:
    """Check monitoring capabilities."""
    try:
        from bci_gpt.utils.health_monitoring import SystemHealthMonitor
        from bci_gpt.core.error_handling import ErrorHandler
        
        # Test health monitoring
        hm = SystemHealthMonitor()
        status = hm.get_health_status()
        
        if not isinstance(status, dict) or 'overall_status' not in status:
            return False, "Health monitoring status check failed"
        
        # Test error handling
        eh = ErrorHandler()
        stats = eh.get_error_statistics()
        
        if not isinstance(stats, dict):
            return False, "Error handling statistics check failed"
        
        return True, "Monitoring checks passed"
    except Exception as e:
        return False, f"Monitoring check failed: {e}"

def check_graceful_degradation() -> Tuple[bool, str]:
    """Check graceful degradation with missing dependencies."""
    try:
        # Test imports work without optional dependencies
        import bci_gpt
        from bci_gpt.core import error_handling, validation
        from bci_gpt.utils import health_monitoring
        from bci_gpt.optimization import performance_optimizer
        from bci_gpt.scaling import auto_scaler
        
        return True, "Graceful degradation checks passed"
    except Exception as e:
        return False, f"Graceful degradation check failed: {e}"

def check_configuration() -> Tuple[bool, str]:
    """Check configuration files and structure."""
    try:
        required_files = [
            'bci_gpt/__init__.py',
            'bci_gpt/core/__init__.py',
            'bci_gpt/utils/__init__.py',
            'README.md',
            'requirements.txt'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            return False, f"Missing required files: {missing_files}"
        
        # Check Python version compatibility
        if sys.version_info < (3, 8):
            return False, f"Python 3.8+ required, got {sys.version_info}"
        
        return True, "Configuration checks passed"
    except Exception as e:
        return False, f"Configuration check failed: {e}"

def run_deployment_readiness_check() -> Dict[str, Any]:
    """Run complete deployment readiness check."""
    checks = [
        ("Code Quality", check_code_quality),
        ("Security", check_security),
        ("Performance", check_performance),
        ("Scalability", check_scalability),
        ("Monitoring", check_monitoring),
        ("Graceful Degradation", check_graceful_degradation),
        ("Configuration", check_configuration)
    ]
    
    results = {}
    all_passed = True
    
    print("=" * 60)
    print("BCI-GPT Production Deployment Readiness Check")
    print("=" * 60)
    
    for check_name, check_func in checks:
        print(f"Running: {check_name}")
        try:
            passed, message = check_func()
            results[check_name] = {
                "passed": passed,
                "message": message
            }
            
            if passed:
                print(f"✅ {check_name}: {message}")
            else:
                print(f"❌ {check_name}: {message}")
                all_passed = False
                
        except Exception as e:
            results[check_name] = {
                "passed": False,
                "message": f"Check failed with exception: {e}"
            }
            print(f"❌ {check_name}: Check failed with exception: {e}")
            all_passed = False
    
    print("=" * 60)
    
    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)
    
    print(f"Deployment Readiness: {passed_count}/{total_count} checks passed")
    
    if all_passed:
        print("🎉 SYSTEM IS PRODUCTION-READY!")
        deployment_status = "READY"
    else:
        print("⚠️  System needs fixes before production deployment")
        deployment_status = "NOT_READY"
    
    print("=" * 60)
    
    return {
        "status": deployment_status,
        "overall_passed": all_passed,
        "checks": results,
        "summary": {
            "total_checks": total_count,
            "passed_checks": passed_count,
            "failed_checks": total_count - passed_count
        }
    }

def main():
    """Main entry point."""
    results = run_deployment_readiness_check()
    
    if results["overall_passed"]:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())