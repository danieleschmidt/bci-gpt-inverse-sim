"""
Self-Healing Quality Validation System v4.0
Automatic detection, diagnosis, and resolution of quality issues.
"""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import json
import re
import sys

logger = logging.getLogger(__name__)


class HealingAction(Enum):
    """Types of healing actions."""
    FIX_SYNTAX = "fix_syntax"
    INSTALL_DEPS = "install_dependencies"
    FORMAT_CODE = "format_code"
    FIX_IMPORTS = "fix_imports"
    CREATE_MISSING = "create_missing"
    UPDATE_CONFIG = "update_config"
    REPAIR_TESTS = "repair_tests"


@dataclass
class HealingRule:
    """Self-healing rule definition."""
    name: str
    pattern: str
    action: HealingAction
    command: str
    description: str
    priority: int = 1
    conditions: List[str] = None


class SelfHealingSystem:
    """
    Autonomous self-healing system that automatically detects
    and resolves common development and quality issues.
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.healing_rules = self._setup_healing_rules()
        self.healing_history: List[Dict[str, Any]] = []
        
    def _setup_healing_rules(self) -> List[HealingRule]:
        """Setup comprehensive healing rules for the BCI-GPT system."""
        return [
            # Python syntax and import fixes
            HealingRule(
                name="fix_missing_init",
                pattern=r"No module named.*bci_gpt",
                action=HealingAction.CREATE_MISSING,
                command="find bci_gpt -type d -exec touch {}/__init__.py \\;",
                description="Create missing __init__.py files",
                priority=1
            ),
            
            HealingRule(
                name="install_requirements",
                pattern=r"ModuleNotFoundError|ImportError",
                action=HealingAction.INSTALL_DEPS,
                command="python3 -m pip install -e . && python3 -m pip install -r requirements.txt",
                description="Install missing dependencies",
                priority=1
            ),
            
            HealingRule(
                name="format_code_black",
                pattern=r"would reformat|formatting",
                action=HealingAction.FORMAT_CODE,
                command="python3 -m black bci_gpt/ --quiet",
                description="Auto-format code with Black",
                priority=2
            ),
            
            HealingRule(
                name="sort_imports",
                pattern=r"import.*incorrectly sorted",
                action=HealingAction.FORMAT_CODE,
                command="python3 -m isort bci_gpt/ --quiet",
                description="Sort imports with isort",
                priority=2
            ),
            
            HealingRule(
                name="fix_syntax_errors",
                pattern=r"SyntaxError|invalid syntax",
                action=HealingAction.FIX_SYNTAX,
                command="python3 -m py_compile",
                description="Attempt basic syntax fixes",
                priority=1
            ),
            
            # Test-related healing
            HealingRule(
                name="create_test_directories",
                pattern=r"tests.*not found",
                action=HealingAction.CREATE_MISSING,
                command="mkdir -p tests && touch tests/__init__.py",
                description="Create missing test directories",
                priority=2
            ),
            
            HealingRule(
                name="install_pytest",
                pattern=r"pytest.*not found",
                action=HealingAction.INSTALL_DEPS,
                command="python3 -m pip install pytest pytest-cov",
                description="Install testing dependencies",
                priority=1
            ),
            
            # Configuration fixes
            HealingRule(
                name="create_setup_py",
                pattern=r"setup.py.*not found",
                action=HealingAction.CREATE_MISSING,
                command="echo 'from setuptools import setup, find_packages; setup(name=\"bci-gpt\", packages=find_packages())' > setup.py",
                description="Create minimal setup.py",
                priority=2
            ),
            
            # Quality tool installation
            HealingRule(
                name="install_quality_tools",
                pattern=r"black.*not found|isort.*not found|mypy.*not found",
                action=HealingAction.INSTALL_DEPS,
                command="python3 -m pip install black isort mypy",
                description="Install code quality tools",
                priority=1
            ),
        ]
    
    async def diagnose_issue(self, error_output: str, command: str) -> Optional[HealingRule]:
        """Diagnose issue from error output and suggest healing action."""
        for rule in sorted(self.healing_rules, key=lambda x: x.priority):
            if re.search(rule.pattern, error_output, re.IGNORECASE):
                logger.info(f"Diagnosed issue: {rule.description}")
                return rule
        return None
    
    async def apply_healing(self, rule: HealingRule) -> bool:
        """Apply a healing rule to fix an issue."""
        try:
            logger.info(f"Applying healing action: {rule.description}")
            
            # Execute healing command
            process = await asyncio.create_subprocess_shell(
                rule.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await process.communicate()
            success = process.returncode == 0
            
            # Record healing attempt
            self.healing_history.append({
                "timestamp": time.time(),
                "rule_name": rule.name,
                "command": rule.command,
                "success": success,
                "stdout": stdout.decode()[:500],
                "stderr": stderr.decode()[:500]
            })
            
            if success:
                logger.info(f"✅ Successfully applied: {rule.description}")
            else:
                logger.warning(f"❌ Failed to apply: {rule.description}")
                logger.debug(f"Error: {stderr.decode()}")
            
            return success
            
        except Exception as e:
            logger.error(f"Exception during healing: {e}")
            return False
    
    async def heal_system(self, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Perform comprehensive system healing with multiple iterations.
        """
        healing_report = {
            "iterations": 0,
            "fixes_applied": [],
            "remaining_issues": [],
            "success": False,
            "timestamp": time.time()
        }
        
        for iteration in range(max_iterations):
            healing_report["iterations"] = iteration + 1
            logger.info(f"🔧 Healing iteration {iteration + 1}/{max_iterations}")
            
            # Attempt to fix critical system issues first
            critical_fixes = await self._apply_critical_fixes()
            healing_report["fixes_applied"].extend(critical_fixes)
            
            # Run a lightweight validation check
            validation_result = await self._run_validation_check()
            
            if validation_result["success"]:
                healing_report["success"] = True
                logger.info("✅ System healing successful!")
                break
            else:
                # Diagnose and fix specific issues
                for issue in validation_result["issues"]:
                    rule = await self.diagnose_issue(issue["error"], issue["command"])
                    if rule:
                        success = await self.apply_healing(rule)
                        healing_report["fixes_applied"].append({
                            "rule": rule.name,
                            "description": rule.description,
                            "success": success
                        })
                
                # Wait between iterations
                if iteration < max_iterations - 1:
                    await asyncio.sleep(2)
        
        # Save healing report
        await self._save_healing_report(healing_report)
        
        return healing_report
    
    async def _apply_critical_fixes(self) -> List[Dict[str, Any]]:
        """Apply critical fixes that are always safe."""
        critical_fixes = []
        
        # Ensure all directories have __init__.py files
        init_files = [
            "bci_gpt/__init__.py",
            "bci_gpt/autonomous/__init__.py",
            "bci_gpt/core/__init__.py",
            "bci_gpt/utils/__init__.py",
            "tests/__init__.py"
        ]
        
        for init_file in init_files:
            file_path = self.project_root / init_file
            if not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()
                critical_fixes.append({
                    "action": "create_init_file",
                    "file": str(file_path),
                    "success": True
                })
        
        # Create essential directories
        essential_dirs = [
            "tests",
            "quality_reports",
            "logs"
        ]
        
        for dir_name in essential_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                critical_fixes.append({
                    "action": "create_directory",
                    "directory": str(dir_path),
                    "success": True
                })
        
        return critical_fixes
    
    async def _run_validation_check(self) -> Dict[str, Any]:
        """Run lightweight validation to identify issues."""
        validation_issues = []
        
        # Check Python syntax
        try:
            process = await asyncio.create_subprocess_shell(
                "python3 -m py_compile bci_gpt/__init__.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                validation_issues.append({
                    "command": "syntax_check",
                    "error": stderr.decode(),
                    "severity": "critical"
                })
        except Exception as e:
            validation_issues.append({
                "command": "syntax_check",
                "error": str(e),
                "severity": "critical"
            })
        
        # Check import structure
        try:
            process = await asyncio.create_subprocess_shell(
                "python3 -c \"import bci_gpt; print('Import successful')\"",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                validation_issues.append({
                    "command": "import_check",
                    "error": stderr.decode(),
                    "severity": "high"
                })
        except Exception as e:
            validation_issues.append({
                "command": "import_check",
                "error": str(e),
                "severity": "high"
            })
        
        return {
            "success": len(validation_issues) == 0,
            "issues": validation_issues,
            "timestamp": time.time()
        }
    
    async def _save_healing_report(self, report: Dict[str, Any]):
        """Save healing report to file."""
        report_path = self.project_root / "quality_reports" / "self_healing_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Healing report saved to: {report_path}")
    
    def get_healing_summary(self) -> Dict[str, Any]:
        """Get summary of healing activities."""
        if not self.healing_history:
            return {"status": "no_healing_performed", "total_actions": 0}
        
        successful_actions = sum(1 for h in self.healing_history if h["success"])
        total_actions = len(self.healing_history)
        
        return {
            "status": "healing_active",
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": successful_actions / total_actions if total_actions > 0 else 0.0,
            "last_healing": max(h["timestamp"] for h in self.healing_history),
            "actions_performed": [h["rule_name"] for h in self.healing_history if h["success"]]
        }


# Integration with Progressive Quality Gates
class EnhancedQualityGates:
    """Enhanced quality gates with self-healing capabilities."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.healing_system = SelfHealingSystem(project_root)
        
    async def run_with_healing(self, max_healing_iterations: int = 3) -> Dict[str, Any]:
        """Run quality gates with automatic healing."""
        from .progressive_quality_gates import ProgressiveQualityGates
        
        # First attempt healing
        logger.info("🏥 Starting system healing before quality gates...")
        healing_report = await self.healing_system.heal_system(max_healing_iterations)
        
        # Run quality gates
        logger.info("🚀 Running enhanced quality gates...")
        gates = ProgressiveQualityGates()
        results = await gates.execute_all_gates(parallel=True)
        summary = gates.get_summary()
        
        # If quality gates still fail, attempt targeted healing
        if summary['pass_rate'] < 0.8:
            logger.info("🔧 Quality gates below threshold, applying targeted healing...")
            
            # Analyze specific failures and apply targeted fixes
            for gate_name, result in results.items():
                if result.status.value == "failed":
                    rule = await self.healing_system.diagnose_issue(
                        result.message, 
                        gates.gates[gate_name].command
                    )
                    if rule:
                        await self.healing_system.apply_healing(rule)
            
            # Re-run failed gates
            logger.info("🔄 Re-running quality gates after healing...")
            results = await gates.execute_all_gates(parallel=True)
            summary = gates.get_summary()
        
        # Combine results
        enhanced_summary = {
            **summary,
            "healing_applied": healing_report["success"],
            "healing_iterations": healing_report["iterations"],
            "fixes_applied": len(healing_report["fixes_applied"]),
            "self_healing_summary": self.healing_system.get_healing_summary()
        }
        
        return enhanced_summary


# Standalone healing function
async def heal_system(project_root: Path = None, max_iterations: int = 3) -> Dict[str, Any]:
    """Standalone system healing function."""
    healer = SelfHealingSystem(project_root)
    return await healer.heal_system(max_iterations)


async def run_enhanced_quality_gates(project_root: Path = None) -> Dict[str, Any]:
    """Run quality gates with automatic healing."""
    enhanced = EnhancedQualityGates(project_root)
    return await enhanced.run_with_healing()