#!/usr/bin/env python3
"""Lightweight BCI-GPT system validation without heavy dependencies."""

import sys
import os
import traceback
from pathlib import Path

# Test results tracker
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "errors": []
}

def test_basic_imports():
    """Test basic package imports without PyTorch."""
    test_results["total"] += 1
    try:
        import bci_gpt
        assert hasattr(bci_gpt, '__version__')
        print(f"✅ BCI-GPT version: {bci_gpt.__version__}")
        test_results["passed"] += 1
        return True
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Basic imports failed: {e}")
        print(f"❌ Basic imports failed: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    test_results["total"] += 1
    try:
        from bci_gpt.utils.logging_config import setup_logging, get_logger
        from bci_gpt.utils.error_handling import BCI_GPTError
        
        # Test logging
        logger = get_logger(__name__)
        logger.info("Test logging successful")
        
        # Test custom error
        try:
            raise BCI_GPTError("Test error")
        except BCI_GPTError:
            pass  # Expected
            
        print("✅ Utilities working correctly")
        test_results["passed"] += 1
        return True
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Utilities test failed: {e}")
        print(f"❌ Utilities test failed: {e}")
        return False

def test_preprocessing_lightweight():
    """Test preprocessing without MNE."""
    test_results["total"] += 1
    try:
        # Test that modules can be imported even if MNE is missing
        from bci_gpt.preprocessing import EEGProcessor, SignalQuality
        
        # These should work even without actual EEG data
        processor = EEGProcessor()
        assert processor is not None
        
        print("✅ Preprocessing modules importable")
        test_results["passed"] += 1
        return True
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Preprocessing test failed: {e}")
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    test_results["total"] += 1
    try:
        from bci_gpt.utils.config_manager import get_config_manager
        
        config_manager = get_config_manager()
        assert config_manager is not None
        
        print("✅ Configuration system working")
        test_results["passed"] += 1
        return True
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Configuration test failed: {e}")
        print(f"❌ Configuration test failed: {e}")
        return False

def test_file_structure():
    """Test that key files exist."""
    test_results["total"] += 1
    try:
        repo_root = Path(__file__).parent
        
        # Check key files
        key_files = [
            "bci_gpt/__init__.py",
            "bci_gpt/core/models.py",
            "bci_gpt/core/inverse_gan.py",
            "bci_gpt/preprocessing/eeg_processor.py",
            "bci_gpt/training/trainer.py",
            "pyproject.toml",
            "requirements.txt"
        ]
        
        missing_files = []
        for file_path in key_files:
            if not (repo_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            raise AssertionError(f"Missing files: {missing_files}")
        
        print("✅ All key files present")
        test_results["passed"] += 1
        return True
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"File structure test failed: {e}")
        print(f"❌ File structure test failed: {e}")
        return False

def test_cli_interfaces():
    """Test CLI interfaces can be imported."""
    test_results["total"] += 1
    try:
        # Test main CLI
        from bci_gpt import cli
        
        # Test sub-CLIs
        from bci_gpt.training import cli as training_cli
        from bci_gpt.decoding import cli as decoding_cli
        from bci_gpt.inverse import cli as inverse_cli
        
        print("✅ CLI interfaces importable")
        test_results["passed"] += 1
        return True
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"CLI test failed: {e}")
        print(f"❌ CLI test failed: {e}")
        return False

def run_lightweight_validation():
    """Run all lightweight validation tests."""
    print("🧠 BCI-GPT Lightweight System Validation")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_basic_imports,
        test_utilities,
        test_preprocessing_lightweight,
        test_configuration,
        test_file_structure,
        test_cli_interfaces
    ]
    
    for test_func in tests:
        print(f"\n📋 Running {test_func.__name__}...")
        test_func()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print(f"Total tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']} ✅")
    print(f"Failed: {test_results['failed']} ❌")
    
    if test_results["errors"]:
        print("\n🔍 ERRORS:")
        for error in test_results["errors"]:
            print(f"  - {error}")
    
    success_rate = (test_results['passed'] / test_results['total']) * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🚀 SYSTEM STATUS: OPERATIONAL")
        return True
    else:
        print("⚠️  SYSTEM STATUS: NEEDS ATTENTION")
        return False

if __name__ == "__main__":
    success = run_lightweight_validation()
    sys.exit(0 if success else 1)