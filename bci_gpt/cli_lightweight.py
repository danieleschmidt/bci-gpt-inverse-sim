#!/usr/bin/env python3
"""Lightweight CLI interface for BCI-GPT without heavy dependencies."""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, List

from .utils.logging_config import get_logger
from .utils.config_manager import get_config_manager


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for BCI-GPT CLI."""
    parser = argparse.ArgumentParser(
        prog="bci-gpt",
        description="BCI-GPT: Brain-Computer Interface GPT Inverse Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="BCI-GPT 0.1.0"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show system status and configuration"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate system configuration and dependencies"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management"
    )
    config_subparsers = config_parser.add_subparsers(dest="config_action")
    
    config_show = config_subparsers.add_parser("show", help="Show configuration")
    config_set = config_subparsers.add_parser("set", help="Set configuration value")
    config_set.add_argument("key", help="Configuration key (e.g., model.hidden_dim)")
    config_set.add_argument("value", help="Configuration value")
    
    return parser


def cmd_status(args) -> int:
    """Show system status."""
    logger = get_logger(__name__)
    
    print("🧠 BCI-GPT System Status")
    print("=" * 40)
    
    # System information
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Package information
    try:
        import bci_gpt
        print(f"BCI-GPT Version: {bci_gpt.__version__}")
    except ImportError:
        print("❌ BCI-GPT package not found")
        return 1
    
    # Check dependencies
    dependencies = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "sklearn": "Scikit-learn",
        "mne": "MNE-Python",
        "transformers": "HuggingFace Transformers"
    }
    
    print("\n📦 Dependencies:")
    for dep, name in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (optional)")
    
    # Configuration status
    print("\n⚙️  Configuration:")
    try:
        config = get_config_manager()
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1
    
    logger.info("System status check completed")
    return 0


def cmd_validate(args) -> int:
    """Validate system configuration and dependencies."""
    logger = get_logger(__name__)
    
    print("🔍 BCI-GPT System Validation")
    print("=" * 40)
    
    errors = []
    warnings = []
    
    # Basic imports
    try:
        import bci_gpt
        print("✅ BCI-GPT package import")
    except ImportError as e:
        errors.append(f"BCI-GPT import failed: {e}")
        print("❌ BCI-GPT package import")
    
    # Configuration
    try:
        config = get_config_manager()
        print("✅ Configuration system")
    except Exception as e:
        errors.append(f"Configuration error: {e}")
        print("❌ Configuration system")
    
    # Utilities
    try:
        from bci_gpt.utils.error_handling import BCI_GPTError
        print("✅ Error handling utilities")
    except ImportError as e:
        warnings.append(f"Error handling import: {e}")
        print("⚠️  Error handling utilities")
    
    # Preprocessing (optional)
    try:
        from bci_gpt.preprocessing import EEGProcessor
        print("✅ Preprocessing modules")
    except ImportError as e:
        warnings.append(f"Preprocessing import: {e}")
        print("⚠️  Preprocessing modules")
    
    # Summary
    print("\n📊 Validation Summary:")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    
    if errors:
        print("\n🔴 Errors:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("\n🟡 Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors:
        print("\n🚀 System validation passed!")
        logger.info("System validation completed successfully")
        return 0
    else:
        print("\n❌ System validation failed!")
        logger.error("System validation failed with errors")
        return 1


def cmd_config(args) -> int:
    """Configuration management."""
    if not args.config_action:
        print("Error: No config action specified. Use 'show' or 'set'.")
        return 1
    
    try:
        config = get_config_manager()
    except Exception as e:
        print(f"Error: Failed to load configuration: {e}")
        return 1
    
    if args.config_action == "show":
        print("⚙️  BCI-GPT Configuration:")
        print("=" * 30)
        
        # Show configuration sections
        if hasattr(config, 'config_data'):
            import json
            print(json.dumps(config.config_data, indent=2))
        else:
            print("Configuration data not available")
    
    elif args.config_action == "set":
        try:
            # Parse value (try JSON first, then string)
            try:
                import json
                value = json.loads(args.value)
            except (json.JSONDecodeError, ValueError):
                value = args.value
            
            config.set(args.key, value)
            print(f"✅ Set {args.key} = {value}")
            
            # Save configuration
            if hasattr(config, 'save'):
                config.save()
                print("💾 Configuration saved")
        except Exception as e:
            print(f"❌ Error setting configuration: {e}")
            return 1
    
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    if args.verbose:
        os.environ['BCI_GPT_LOG_LEVEL'] = 'DEBUG'
    
    # Execute command
    if args.command == "status":
        return cmd_status(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "config":
        return cmd_config(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())