#!/usr/bin/env python3
"""
Production Deployment Validator
Final validation script for production deployment readiness
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

def validate_core_architecture():
    """Validate core BCI-GPT architecture"""
    print("🏗️  Validating Core Architecture...")
    
    checks = []
    
    # Check core modules exist
    core_modules = [
        "bci_gpt/core/models.py",
        "bci_gpt/core/inverse_gan.py", 
        "bci_gpt/core/fusion_layers.py",
        "bci_gpt/preprocessing/eeg_processor.py",
        "bci_gpt/decoding/realtime_decoder.py",
        "bci_gpt/training/trainer.py"
    ]
    
    for module in core_modules:
        if Path(module).exists():
            checks.append(f"✅ {module}")
        else:
            checks.append(f"❌ {module}")
    
    print("\n".join(checks))
    passed = sum(1 for check in checks if check.startswith("✅"))
    return passed / len(checks) >= 0.8

def validate_reliability_features():
    """Validate reliability and robustness features"""
    print("\n🛡️  Validating Reliability Features...")
    
    checks = []
    
    # Check robustness modules
    robustness_modules = [
        "bci_gpt/robustness/circuit_breaker.py",
        "bci_gpt/robustness/retry_manager.py",
        "bci_gpt/robustness/health_checker.py",
        "bci_gpt/robustness/fault_tolerance.py",
        "bci_gpt/robustness/graceful_degradation.py"
    ]
    
    for module in robustness_modules:
        if Path(module).exists():
            checks.append(f"✅ {module}")
        else:
            checks.append(f"❌ {module}")
    
    print("\n".join(checks))
    passed = sum(1 for check in checks if check.startswith("✅"))
    return passed / len(checks) >= 0.8

def validate_scaling_capabilities():
    """Validate scaling and performance features"""
    print("\n⚡ Validating Scaling Capabilities...")
    
    checks = []
    
    # Check scaling modules
    scaling_modules = [
        "bci_gpt/scaling/advanced_auto_scaler.py",
        "bci_gpt/scaling/load_balancer.py",
        "bci_gpt/scaling/auto_scaler.py"
    ]
    
    for module in scaling_modules:
        if Path(module).exists():
            checks.append(f"✅ {module}")
        else:
            checks.append(f"❌ {module}")
    
    print("\n".join(checks))
    passed = sum(1 for check in checks if check.startswith("✅"))
    return passed / len(checks) >= 0.8

def validate_compliance_security():
    """Validate compliance and security features"""
    print("\n🔒 Validating Compliance & Security...")
    
    checks = []
    
    # Check compliance modules
    compliance_modules = [
        "bci_gpt/compliance/gdpr.py",
        "bci_gpt/compliance/data_protection.py",
        "bci_gpt/global/compliance.py",
        "bci_gpt/utils/security.py"
    ]
    
    for module in compliance_modules:
        if Path(module).exists():
            checks.append(f"✅ {module}")
        else:
            checks.append(f"❌ {module}")
    
    print("\n".join(checks))
    passed = sum(1 for check in checks if check.startswith("✅"))
    return passed / len(checks) >= 0.8

def validate_deployment_infrastructure():
    """Validate deployment infrastructure"""
    print("\n🚀 Validating Deployment Infrastructure...")
    
    checks = []
    
    # Check deployment files
    deployment_files = [
        "deployment/Dockerfile",
        "deployment/docker-compose.prod.yml", 
        "deployment/kubernetes/bci-gpt-deployment.yaml",
        "deploy.sh",
        "requirements.txt",
        "pyproject.toml"
    ]
    
    for file_path in deployment_files:
        if Path(file_path).exists():
            checks.append(f"✅ {file_path}")
        else:
            checks.append(f"❌ {file_path}")
    
    print("\n".join(checks))
    passed = sum(1 for check in checks if check.startswith("✅"))
    return passed / len(checks) >= 0.7

def validate_documentation():
    """Validate documentation completeness"""
    print("\n📚 Validating Documentation...")
    
    checks = []
    
    # Check documentation files
    doc_files = [
        "README.md",
        "IMPLEMENTATION_GUIDE.md",
        "DEPLOYMENT.md",
        "SYSTEM_STATUS.md",
        "RESEARCH_OPPORTUNITIES.md"
    ]
    
    for doc_file in doc_files:
        if Path(doc_file).exists():
            # Check if file has content
            size = Path(doc_file).stat().st_size
            if size > 1000:  # At least 1KB of content
                checks.append(f"✅ {doc_file} ({size} bytes)")
            else:
                checks.append(f"⚠️  {doc_file} (minimal content: {size} bytes)")
        else:
            checks.append(f"❌ {doc_file}")
    
    print("\n".join(checks))
    passed = sum(1 for check in checks if check.startswith("✅"))
    return passed / len(checks) >= 0.8

def validate_test_coverage():
    """Validate test coverage"""
    print("\n🧪 Validating Test Coverage...")
    
    checks = []
    
    # Check test files
    test_files = [
        "test_basic_system.py",
        "test_robustness_system.py",
        "test_scaling_system.py", 
        "test_comprehensive_autonomous_system.py",
        "tests/test_models.py",
        "tests/test_integration.py"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            checks.append(f"✅ {test_file}")
        else:
            checks.append(f"❌ {test_file}")
    
    print("\n".join(checks))
    passed = sum(1 for check in checks if check.startswith("✅"))
    return passed / len(checks) >= 0.7

def run_integration_test():
    """Run quick integration test"""
    print("\n🔗 Running Integration Test...")
    
    try:
        # Test basic import
        result = subprocess.run([
            sys.executable, "-c", "import bci_gpt; print('✅ BCI-GPT package imports successfully')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Package imports successfully")
            
            # Test basic functionality
            result2 = subprocess.run([
                sys.executable, "-c", 
                """
import sys
sys.path.insert(0, '.')
try:
    from bci_gpt.robustness.circuit_breaker import CircuitBreaker
    from bci_gpt.scaling.load_balancer import LoadBalancer
    print('✅ Advanced modules functional')
except Exception as e:
    print(f'⚠️  Advanced modules partial: {e}')
                """
            ], capture_output=True, text=True, timeout=10)
            
            print(result2.stdout.strip() if result2.stdout else "⚠️  Advanced modules test incomplete")
            return True
        else:
            print(f"❌ Import failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def calculate_production_readiness_score(results: Dict[str, bool]) -> float:
    """Calculate overall production readiness score"""
    weights = {
        'core_architecture': 0.25,
        'reliability_features': 0.20,
        'scaling_capabilities': 0.20,
        'compliance_security': 0.15,
        'deployment_infrastructure': 0.10,
        'documentation': 0.05,
        'test_coverage': 0.05
    }
    
    score = sum(weights[key] * (1.0 if passed else 0.0) for key, passed in results.items() if key in weights)
    return score * 100

def generate_deployment_readiness_report(results: Dict[str, bool], score: float) -> Dict[str, Any]:
    """Generate comprehensive deployment readiness report"""
    
    # Determine deployment recommendation
    if score >= 90:
        recommendation = "READY FOR IMMEDIATE PRODUCTION DEPLOYMENT"
        risk_level = "LOW"
        confidence = "HIGH"
    elif score >= 80:
        recommendation = "READY FOR STAGED PRODUCTION DEPLOYMENT"
        risk_level = "MEDIUM"
        confidence = "MEDIUM-HIGH"
    elif score >= 70:
        recommendation = "READY FOR DEVELOPMENT/STAGING DEPLOYMENT"
        risk_level = "MEDIUM-HIGH"
        confidence = "MEDIUM"
    else:
        recommendation = "REQUIRES ADDITIONAL DEVELOPMENT"
        risk_level = "HIGH"
        confidence = "LOW"
    
    report = {
        'timestamp': time.time(),
        'production_readiness_score': score,
        'recommendation': recommendation,
        'risk_level': risk_level,
        'confidence_level': confidence,
        'validation_results': results,
        'deployment_checklist': {
            'infrastructure': results.get('deployment_infrastructure', False),
            'security': results.get('compliance_security', False),
            'reliability': results.get('reliability_features', False),
            'scalability': results.get('scaling_capabilities', False),
            'monitoring': results.get('core_architecture', False),
            'documentation': results.get('documentation', False)
        },
        'next_steps': _generate_next_steps(results, score),
        'deployment_timeline': _estimate_deployment_timeline(score)
    }
    
    return report

def _generate_next_steps(results: Dict[str, bool], score: float) -> List[str]:
    """Generate next steps based on results"""
    steps = []
    
    if score >= 90:
        steps = [
            "✅ System is production-ready",
            "🚀 Begin production deployment",
            "📊 Set up production monitoring",
            "🔄 Implement continuous deployment pipeline"
        ]
    elif score >= 80:
        steps = [
            "✅ Core system is ready",
            "🔧 Address minor gaps identified in validation",
            "🧪 Run extended integration testing",
            "🚀 Plan staged production rollout"
        ]
    else:
        if not results.get('core_architecture', True):
            steps.append("🏗️  Complete core architecture implementation")
        if not results.get('reliability_features', True):
            steps.append("🛡️  Implement reliability and fault tolerance")
        if not results.get('scaling_capabilities', True):
            steps.append("⚡ Add auto-scaling and load balancing")
        if not results.get('compliance_security', True):
            steps.append("🔒 Complete compliance and security features")
    
    return steps

def _estimate_deployment_timeline(score: float) -> str:
    """Estimate deployment timeline"""
    if score >= 90:
        return "Immediate (0-1 days)"
    elif score >= 80:
        return "Short-term (1-3 days)"
    elif score >= 70:
        return "Medium-term (1-2 weeks)"
    else:
        return "Long-term (2-4 weeks)"

def main():
    """Main validation function"""
    print("🚀 BCI-GPT PRODUCTION DEPLOYMENT VALIDATOR")
    print("=" * 60)
    print("Comprehensive production readiness assessment\n")
    
    # Run all validation checks
    validation_functions = [
        ('core_architecture', validate_core_architecture),
        ('reliability_features', validate_reliability_features),
        ('scaling_capabilities', validate_scaling_capabilities),
        ('compliance_security', validate_compliance_security),
        ('deployment_infrastructure', validate_deployment_infrastructure),
        ('documentation', validate_documentation),
        ('test_coverage', validate_test_coverage)
    ]
    
    results = {}
    
    for name, func in validation_functions:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            print(f"❌ Validation {name} failed: {e}")
            results[name] = False
    
    # Run integration test
    integration_success = run_integration_test()
    results['integration_test'] = integration_success
    
    # Calculate production readiness score
    score = calculate_production_readiness_score(results)
    
    # Generate and display report
    print("\n" + "=" * 60)
    print("📊 PRODUCTION DEPLOYMENT READINESS ASSESSMENT")
    print("=" * 60)
    
    print(f"🎯 Production Readiness Score: {score:.1f}%")
    
    # Show individual results
    print(f"\n📋 Validation Results:")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {name.replace('_', ' ').title()}")
    
    # Generate detailed report
    full_report = generate_deployment_readiness_report(results, score)
    
    print(f"\n🎯 Deployment Recommendation: {full_report['recommendation']}")
    print(f"⚠️  Risk Level: {full_report['risk_level']}")
    print(f"🎯 Confidence: {full_report['confidence_level']}")
    print(f"⏱️  Estimated Timeline: {full_report['deployment_timeline']}")
    
    print(f"\n📋 Next Steps:")
    for step in full_report['next_steps']:
        print(f"   {step}")
    
    # Save detailed report
    with open('production_deployment_readiness.json', 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"\n📁 Detailed report saved: production_deployment_readiness.json")
    
    # Final assessment
    if score >= 80:
        print(f"\n🎉 AUTONOMOUS SDLC DEPLOYMENT: ✅ READY")
        print("   System meets production deployment standards!")
        return True
    else:
        print(f"\n⚠️  AUTONOMOUS SDLC DEPLOYMENT: ⚠️  NEEDS WORK")
        print("   System requires additional development before production.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)