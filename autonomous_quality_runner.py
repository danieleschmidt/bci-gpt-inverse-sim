#!/usr/bin/env python3
"""
Autonomous Quality Gates Runner
Execute progressive quality gates with real-time monitoring.
"""

import asyncio
import sys
import json
from pathlib import Path
from bci_gpt.autonomous import run_quality_gates, ProgressiveQualityGates


async def main():
    """Main execution function."""
    print("🚀 Starting Autonomous Progressive Quality Gates...")
    print("=" * 60)
    
    try:
        # Initialize quality gates system
        gates = ProgressiveQualityGates()
        
        # Execute all quality gates
        print("Executing quality gates...")
        results = await gates.execute_all_gates(parallel=True)
        
        # Display results
        print("\n📊 QUALITY GATE RESULTS:")
        print("-" * 40)
        
        for gate_name, result in results.items():
            status_emoji = "✅" if result.status.value == "passed" else "❌"
            print(f"{status_emoji} {gate_name:20} | Score: {result.score:.2f} | "
                  f"Time: {result.execution_time:.1f}s | Status: {result.status.value}")
        
        # Get summary
        summary = gates.get_summary()
        
        print("\n📈 SUMMARY:")
        print("-" * 40)
        print(f"Overall Status: {summary['status'].upper()}")
        print(f"Overall Score: {summary['overall_score']:.3f}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"Critical Pass Rate: {summary['critical_pass_rate']:.1%}")
        print(f"Gates Passed: {summary['gates_passed']}/{summary['gates_total']}")
        print(f"Total Execution Time: {summary['execution_time']:.1f}s")
        
        # Save detailed report
        report_path = Path("quality_reports/autonomous_quality_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        detailed_report = {
            "summary": summary,
            "results": {
                name: {
                    "status": result.status.value,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "message": result.message,
                    "retry_count": result.retry_count
                }
                for name, result in results.items()
            },
            "recommendations": _generate_recommendations(results, summary)
        }
        
        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"\n📝 Detailed report saved to: {report_path}")
        
        # Auto-fix if needed
        if summary['pass_rate'] < 0.9:
            print("\n🔧 Attempting auto-fixes...")
            gates.auto_fix_issues()
        
        # Return appropriate exit code
        if summary['critical_pass_rate'] >= 0.9 and summary['pass_rate'] >= 0.8:
            print("\n🎉 QUALITY GATES PASSED! System ready for production.")
            return 0
        else:
            print("\n⚠️  Some quality gates failed. Review and fix issues.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error executing quality gates: {e}")
        return 1


def _generate_recommendations(results, summary):
    """Generate actionable recommendations based on results."""
    recommendations = []
    
    if summary['pass_rate'] < 0.8:
        recommendations.append("Overall pass rate below 80% - review failed gates and implement fixes")
    
    if summary['critical_pass_rate'] < 0.9:
        recommendations.append("Critical gates failing - address immediately before deployment")
    
    for gate_name, result in results.items():
        if result.status.value == "failed":
            if gate_name == "unit_tests":
                recommendations.append("Unit tests failing - fix test cases and implementation bugs")
            elif gate_name == "code_style":
                recommendations.append("Code style issues - run 'black bci_gpt/' and 'isort bci_gpt/'")
            elif gate_name == "type_checking":
                recommendations.append("Type checking errors - add missing type annotations")
            elif gate_name == "security_scan":
                recommendations.append("Security issues detected - review dependencies and code")
    
    if summary['execution_time'] > 600:
        recommendations.append("Quality gates taking too long - optimize test execution")
    
    return recommendations


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)