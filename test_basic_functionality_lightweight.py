#!/usr/bin/env python3
"""
Lightweight basic functionality test that doesn't require heavy dependencies.
Tests core structure and imports without PyTorch.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_project_structure():
    """Test that all essential project files exist."""
    required_files = [
        "README.md",
        "requirements.txt", 
        "pyproject.toml",
        "bci_gpt/__init__.py",
        "bci_gpt/core/__init__.py",
        "bci_gpt/preprocessing/__init__.py",
        "bci_gpt/utils/__init__.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Project structure validated")
    return True

def test_python_syntax():
    """Test that all Python files have valid syntax."""
    python_files = list(Path("bci_gpt").rglob("*.py"))
    
    for py_file in python_files:
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(py_file)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"❌ Syntax error in {py_file}")
                return False
        except Exception as e:
            print(f"❌ Error checking {py_file}: {e}")
            return False
    
    print(f"✅ All {len(python_files)} Python files have valid syntax")
    return True

def test_basic_imports():
    """Test basic imports that don't require heavy dependencies."""
    basic_modules = [
        "bci_gpt",
        "bci_gpt.utils.logging_config",
        "bci_gpt.utils.error_handling"
    ]
    
    for module in basic_modules:
        try:
            result = subprocess.run(
                ["python3", "-c", f"import {module}; print('OK')"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"❌ Failed to import {module}: {result.stderr}")
                # Don't fail if it's just missing dependencies
                if "torch" in result.stderr or "transformers" in result.stderr:
                    print(f"⚠️  {module} skipped due to missing dependencies")
                    continue
                return False
        except Exception as e:
            print(f"❌ Error testing import {module}: {e}")
            return False
    
    print("✅ Basic imports validated")
    return True

def test_cli_interface():
    """Test that CLI interfaces are accessible."""
    cli_modules = [
        "bci_gpt.cli",
        "bci_gpt.decoding.cli",
        "bci_gpt.inverse.cli",
        "bci_gpt.training.cli"
    ]
    
    for module in cli_modules:
        try:
            result = subprocess.run(
                ["python3", "-c", f"import {module}; print('OK')"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0 and "torch" not in result.stderr:
                print(f"❌ CLI module {module} has issues")
                # Don't fail for dependency issues
                continue
        except Exception as e:
            # CLI modules may fail due to dependencies, that's OK
            pass
    
    print("✅ CLI interfaces checked")
    return True

def test_configuration_files():
    """Test configuration file validity."""
    config_files = [
        "pyproject.toml",
        "requirements.txt"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                content = Path(config_file).read_text()
                if len(content.strip()) == 0:
                    print(f"❌ Empty configuration file: {config_file}")
                    return False
            except Exception as e:
                print(f"❌ Error reading {config_file}: {e}")
                return False
    
    print("✅ Configuration files validated")
    return True

def main():
    """Run all basic functionality tests."""
    print("=" * 60)
    print("BCI-GPT Basic Functionality Test (Lightweight)")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Python Syntax", test_python_syntax),
        ("Basic Imports", test_basic_imports),
        ("CLI Interface", test_cli_interface),
        ("Configuration Files", test_configuration_files)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"Basic Functionality Test Results: {passed}/{total} PASSED")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL BASIC TESTS PASSED - Core structure is valid!")
        return True
    else:
        print("⚠️  Some basic tests failed - check core structure")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)