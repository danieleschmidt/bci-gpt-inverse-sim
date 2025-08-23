#!/usr/bin/env python3
"""Basic system validation test - lightweight without heavy dependencies"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def test_core_imports():
    """Test that core modules can be imported without heavy dependencies"""
    print("🧠 Testing Core Module Imports...")
    
    # Test basic imports
    try:
        import bci_gpt
        print("✅ bci_gpt package imports successfully")
    except ImportError as e:
        print(f"❌ bci_gpt import failed: {e}")
        return False
    
    # Test specific modules that should work without heavy deps
    modules_to_test = [
        'bci_gpt.utils.config_manager',
        'bci_gpt.utils.error_handling',
        'bci_gpt.compliance.gdpr',
        'bci_gpt.compliance.data_protection',
        'bci_gpt.i18n.translator',
        'bci_gpt.global.compliance',
        'bci_gpt.optimization.caching'
    ]
    
    success_count = 0
    for module_name in modules_to_test:
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, 
                str(Path(module_name.replace('.', '/') + '.py'))
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"✅ {module_name}")
                success_count += 1
        except Exception as e:
            print(f"⚠️  {module_name}: {e}")
    
    print(f"📊 Import Success Rate: {success_count}/{len(modules_to_test)} ({success_count/len(modules_to_test)*100:.1f}%)")
    return success_count > len(modules_to_test) * 0.7  # 70% success threshold

def test_basic_functionality():
    """Test basic functionality without heavy ML dependencies"""
    print("\n🔧 Testing Basic Functionality...")
    
    try:
        # Test configuration management
        from bci_gpt.utils.config_manager import ConfigManager
        config = ConfigManager()
        print("✅ ConfigManager instantiation")
        
        # Test error handling
        from bci_gpt.utils.error_handling import BCIError, ValidationError
        error = BCIError("Test error")
        print("✅ Error handling classes")
        
        # Test compliance
        from bci_gpt.compliance.gdpr import GDPRCompliance
        gdpr = GDPRCompliance()
        print("✅ GDPR compliance module")
        
        # Test data protection
        from bci_gpt.compliance.data_protection import DataProtection
        dp = DataProtection()
        print("✅ Data protection module")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_cli_functionality():
    """Test CLI functionality"""
    print("\n💻 Testing CLI Functionality...")
    
    try:
        # Test CLI import
        from bci_gpt import cli
        print("✅ CLI module imports")
        
        # Test if CLI help works
        result = subprocess.run([
            sys.executable, "-m", "bci_gpt.cli", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ CLI --help works")
            return True
        else:
            print(f"⚠️  CLI --help returned {result.returncode}")
            return False
            
    except Exception as e:
        print(f"⚠️  CLI functionality test: {e}")
        return False

def test_project_structure():
    """Validate project structure"""
    print("\n📁 Testing Project Structure...")
    
    required_dirs = [
        'bci_gpt/core',
        'bci_gpt/preprocessing', 
        'bci_gpt/decoding',
        'bci_gpt/training',
        'bci_gpt/inverse',
        'bci_gpt/utils',
        'bci_gpt/compliance',
        'bci_gpt/global',
        'bci_gpt/i18n',
        'deployment',
        'tests'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"⚠️  Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories present")
        return True

def test_configuration_files():
    """Test configuration files"""
    print("\n⚙️  Testing Configuration Files...")
    
    required_files = [
        'pyproject.toml',
        'requirements.txt', 
        'pytest.ini',
        'README.md',
        'SYSTEM_STATUS.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        return False
    else:
        print("✅ All required configuration files present")
        return True

def main():
    """Run all basic tests"""
    print("🚀 BCI-GPT Basic System Validation")
    print("=" * 50)
    
    tests = [
        test_project_structure,
        test_configuration_files,
        test_core_imports,
        test_basic_functionality,
        test_cli_functionality
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
    print("📊 BASIC SYSTEM VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 GENERATION 1 (MAKE IT WORK): ✅ PASSED")
        print("   System core functionality is operational")
        return True
    elif success_rate >= 60:
        print("⚠️  GENERATION 1 (MAKE IT WORK): ⚠️  PARTIAL")
        print("   System has basic functionality but needs fixes")
        return False
    else:
        print("❌ GENERATION 1 (MAKE IT WORK): ❌ FAILED") 
        print("   System core functionality is not operational")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)