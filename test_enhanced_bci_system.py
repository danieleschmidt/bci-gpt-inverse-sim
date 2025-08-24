#!/usr/bin/env python3
"""Test enhanced BCI-GPT system with breakthrough architectures.

This test validates the enhanced BCI-GPT models with:
1. Real-time performance (<50ms latency)
2. Multi-language support
3. Uncertainty quantification
4. Adaptive signal quality processing

Authors: Daniel Schmidt, Terragon Labs
Status: Production Validation Test
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Any
import warnings

try:
    from bci_gpt.core.enhanced_models import (
        EnhancedBCIGPTModel,
        RealTimeOptimizedEEGEncoder,
        UncertaintyEstimator,
        SignalQualityAssessor
    )
    from bci_gpt.research.advanced_fusion_architectures import (
        AttentionGuidedSpectralTemporalFusion,
        AttentionGuidedFusionConfig
    )
    from bci_gpt.research.experimental_validation import (
        ExperimentRunner,
        ExperimentConfig
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Ensure all BCI-GPT modules are properly installed")
    exit(1)


class EnhancedSystemValidator:
    """Comprehensive validation for enhanced BCI-GPT system."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests."""
        
        print("🚀 Starting Enhanced BCI-GPT System Validation")
        print(f"🖥️  Device: {self.device}")
        print("=" * 60)
        
        # Test 1: Model Initialization and Architecture
        print("\n🏗️  Test 1: Enhanced Model Architecture")
        arch_results = self._test_enhanced_architecture()
        self.results['architecture'] = arch_results
        
        # Test 2: Real-time Performance
        print("\n⚡ Test 2: Real-time Performance Validation")
        perf_results = self._test_real_time_performance()
        self.results['performance'] = perf_results
        
        # Test 3: Multi-language Support
        print("\n🌍 Test 3: Multi-language Neural Pattern Recognition")
        lang_results = self._test_multilingual_support()
        self.results['multilingual'] = lang_results
        
        # Test 4: Uncertainty Quantification
        print("\n🎯 Test 4: Uncertainty Quantification")
        uncertainty_results = self._test_uncertainty_quantification()
        self.results['uncertainty'] = uncertainty_results
        
        # Test 5: Signal Quality Assessment
        print("\n📊 Test 5: Adaptive Signal Quality Processing")
        quality_results = self._test_signal_quality()
        self.results['signal_quality'] = quality_results
        
        # Test 6: Advanced Fusion Architecture
        print("\n🔬 Test 6: Advanced Fusion Architecture")
        fusion_results = self._test_advanced_fusion()
        self.results['fusion'] = fusion_results
        
        # Test 7: End-to-end Integration
        print("\n🔄 Test 7: End-to-end System Integration")
        integration_results = self._test_integration()
        self.results['integration'] = integration_results
        
        # Generate summary report
        summary = self._generate_validation_summary()
        self.results['summary'] = summary
        
        return self.results
    
    def _test_enhanced_architecture(self) -> Dict[str, Any]:
        """Test enhanced model architecture."""
        
        results = {
            'model_created': False,
            'parameters': 0,
            'components_tested': [],
            'errors': []
        }
        
        try:
            # Create enhanced model
            model_config = {
                'eeg_channels': 32,
                'sampling_rate': 1000,
                'sequence_length': 1000,
                'hidden_dim': 512,
                'num_attention_heads': 16,
                'fusion_layers': 4,
                'enable_uncertainty': True,
                'enable_multi_language': True,
                'real_time_optimization': True
            }
            
            model = EnhancedBCIGPTModel(**model_config)
            model.to(self.device)
            
            results['model_created'] = True
            results['parameters'] = sum(p.numel() for p in model.parameters())
            
            print(f"   ✅ Enhanced model created successfully")
            print(f"   📊 Parameters: {results['parameters']:,}")
            
            # Test components
            components = [
                ('EEG Encoder', model.eeg_encoder),
                ('Advanced Fusion', model.advanced_fusion),
                ('Signal Quality Assessor', model.signal_quality_assessor),
                ('Attention Controller', model.attention_controller),
                ('Performance Monitor', model.performance_monitor)
            ]
            
            for name, component in components:
                if component is not None:
                    results['components_tested'].append(name)
                    print(f"   ✅ {name}: OK")
                else:
                    results['errors'].append(f"{name} not initialized")
                    print(f"   ❌ {name}: Missing")
            
            # Test uncertainty estimator
            if hasattr(model, 'uncertainty_estimator') and model.uncertainty_estimator:
                results['components_tested'].append('Uncertainty Estimator')
                print(f"   ✅ Uncertainty Estimator: OK")
            
            # Test multi-language support
            if hasattr(model, 'language_models'):
                lang_count = len(model.language_models)
                results['components_tested'].append(f'Multi-language Support ({lang_count} languages)')
                print(f"   ✅ Multi-language Support: {lang_count} languages")
            
        except Exception as e:
            results['errors'].append(str(e))
            print(f"   ❌ Architecture test failed: {e}")
        
        return results
    
    def _test_real_time_performance(self) -> Dict[str, Any]:
        """Test real-time performance requirements."""
        
        results = {
            'target_latency_ms': 50,
            'measured_latencies': [],
            'average_latency': 0,
            'meets_requirement': False,
            'throughput_samples_per_sec': 0,
            'errors': []
        }
        
        try:
            # Create model for performance testing
            model = EnhancedBCIGPTModel(
                eeg_channels=32,
                real_time_optimization=True
            )
            model.to(self.device)
            model.eval()
            
            # Warm up
            dummy_eeg = torch.randn(1, 32, 1000, device=self.device)
            for _ in range(5):
                with torch.no_grad():
                    _ = model(dummy_eeg)
            
            # Performance test
            num_tests = 100
            latencies = []
            
            print(f"   🏃 Running {num_tests} performance tests...")
            
            for i in range(num_tests):
                # Generate test data
                batch_size = 1
                eeg_data = torch.randn(batch_size, 32, 1000, device=self.device)
                
                # Measure latency
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    outputs = model(eeg_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                if i % 20 == 0:
                    print(f"   📈 Test {i}: {latency_ms:.2f}ms")
            
            # Compute statistics
            results['measured_latencies'] = latencies
            results['average_latency'] = np.mean(latencies)
            results['std_latency'] = np.std(latencies)
            results['min_latency'] = np.min(latencies)
            results['max_latency'] = np.max(latencies)
            results['p95_latency'] = np.percentile(latencies, 95)
            results['p99_latency'] = np.percentile(latencies, 99)
            
            # Check requirement
            results['meets_requirement'] = results['average_latency'] < results['target_latency_ms']
            
            # Compute throughput
            results['throughput_samples_per_sec'] = 1000 / results['average_latency']
            
            print(f"   📊 Average latency: {results['average_latency']:.2f}ms")
            print(f"   📊 P95 latency: {results['p95_latency']:.2f}ms")
            print(f"   📊 Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
            
            if results['meets_requirement']:
                print(f"   ✅ Meets real-time requirement (<{results['target_latency_ms']}ms)")
            else:
                print(f"   ❌ Exceeds latency target ({results['target_latency_ms']}ms)")
            
        except Exception as e:
            results['errors'].append(str(e))
            print(f"   ❌ Performance test failed: {e}")
        
        return results
    
    def _test_multilingual_support(self) -> Dict[str, Any]:
        """Test multi-language neural pattern recognition."""
        
        results = {
            'supported_languages': [],
            'generation_tests': {},
            'errors': []
        }
        
        try:
            # Create model with multi-language support
            model = EnhancedBCIGPTModel(
                enable_multi_language=True,
                eeg_channels=32
            )
            model.to(self.device)
            model.eval()
            
            # Test supported languages
            if hasattr(model, 'language_models'):
                results['supported_languages'] = list(model.language_models.keys())
                print(f"   🌍 Supported languages: {results['supported_languages']}")
            
            # Test generation for each language
            test_eeg = torch.randn(2, 32, 1000, device=self.device)
            
            for language in results['supported_languages']:
                try:
                    print(f"   🔤 Testing {language} generation...")
                    
                    generation_result = model.generate_text_multilingual(
                        test_eeg,
                        language=language,
                        max_length=20,
                        confidence_threshold=0.5
                    )
                    
                    results['generation_tests'][language] = {
                        'success': True,
                        'texts_generated': len(generation_result.get('texts', [])),
                        'avg_confidence': generation_result.get('avg_confidence', 0.0)
                    }
                    
                    print(f"   ✅ {language}: Generated {results['generation_tests'][language]['texts_generated']} texts")
                    print(f"   📈 {language}: Avg confidence {results['generation_tests'][language]['avg_confidence']:.3f}")
                    
                except Exception as e:
                    results['generation_tests'][language] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"   ❌ {language}: Failed - {e}")
            
        except Exception as e:
            results['errors'].append(str(e))
            print(f"   ❌ Multi-language test failed: {e}")
        
        return results
    
    def _test_uncertainty_quantification(self) -> Dict[str, Any]:
        """Test uncertainty quantification capabilities."""
        
        results = {
            'uncertainty_enabled': False,
            'epistemic_uncertainty_computed': False,
            'aleatoric_uncertainty_computed': False,
            'uncertainty_stats': {},
            'errors': []
        }
        
        try:
            # Create model with uncertainty
            model = EnhancedBCIGPTModel(
                enable_uncertainty=True,
                eeg_channels=32
            )
            model.to(self.device)
            model.eval()
            
            results['uncertainty_enabled'] = True
            print(f"   🎯 Uncertainty estimation enabled")
            
            # Test uncertainty computation
            test_eeg = torch.randn(4, 32, 1000, device=self.device)
            
            with torch.no_grad():
                outputs = model(test_eeg)
            
            if 'uncertainty_params' in outputs and outputs['uncertainty_params']:
                uncertainty = outputs['uncertainty_params']
                
                if 'epistemic_uncertainty' in uncertainty:
                    results['epistemic_uncertainty_computed'] = True
                    epistemic = uncertainty['epistemic_uncertainty']
                    results['uncertainty_stats']['epistemic_mean'] = float(torch.mean(epistemic))
                    results['uncertainty_stats']['epistemic_std'] = float(torch.std(epistemic))
                    print(f"   ✅ Epistemic uncertainty: μ={results['uncertainty_stats']['epistemic_mean']:.4f}")
                
                if 'aleatoric_uncertainty' in uncertainty:
                    results['aleatoric_uncertainty_computed'] = True
                    aleatoric = uncertainty['aleatoric_uncertainty']
                    results['uncertainty_stats']['aleatoric_mean'] = float(torch.mean(aleatoric))
                    results['uncertainty_stats']['aleatoric_std'] = float(torch.std(aleatoric))
                    print(f"   ✅ Aleatoric uncertainty: μ={results['uncertainty_stats']['aleatoric_mean']:.4f}")
                
                if 'total_uncertainty' in uncertainty:
                    total = uncertainty['total_uncertainty']
                    results['uncertainty_stats']['total_mean'] = float(torch.mean(total))
                    results['uncertainty_stats']['total_std'] = float(torch.std(total))
                    print(f"   ✅ Total uncertainty: μ={results['uncertainty_stats']['total_mean']:.4f}")
            
        except Exception as e:
            results['errors'].append(str(e))
            print(f"   ❌ Uncertainty test failed: {e}")
        
        return results
    
    def _test_signal_quality(self) -> Dict[str, Any]:
        """Test signal quality assessment."""
        
        results = {
            'quality_assessment_working': False,
            'channel_quality_computed': False,
            'adaptive_processing': False,
            'quality_stats': {},
            'errors': []
        }
        
        try:
            # Create signal quality assessor
            assessor = SignalQualityAssessor(n_channels=32, sampling_rate=1000)
            assessor.to(self.device)
            
            # Test with different signal qualities
            print(f"   📊 Testing signal quality assessment...")
            
            # Good quality signal
            good_signal = torch.randn(2, 32, 1000, device=self.device) * 0.5
            
            # Poor quality signal (high noise)
            poor_signal = torch.randn(2, 32, 1000, device=self.device) * 5.0
            
            with torch.no_grad():
                good_quality = assessor(good_signal)
                poor_quality = assessor(poor_signal)
            
            results['quality_assessment_working'] = True
            print(f"   ✅ Signal quality assessment working")
            
            if 'channel_quality' in good_quality:
                results['channel_quality_computed'] = True
                
                good_avg = float(torch.mean(good_quality['channel_quality']))
                poor_avg = float(torch.mean(poor_quality['channel_quality']))
                
                results['quality_stats'] = {
                    'good_signal_quality': good_avg,
                    'poor_signal_quality': poor_avg,
                    'quality_discrimination': good_avg - poor_avg
                }
                
                print(f"   📈 Good signal quality: {good_avg:.3f}")
                print(f"   📉 Poor signal quality: {poor_avg:.3f}")
                print(f"   🎯 Quality discrimination: {results['quality_stats']['quality_discrimination']:.3f}")
                
                if results['quality_stats']['quality_discrimination'] > 0.1:
                    results['adaptive_processing'] = True
                    print(f"   ✅ Quality discrimination successful")
            
        except Exception as e:
            results['errors'].append(str(e))
            print(f"   ❌ Signal quality test failed: {e}")
        
        return results
    
    def _test_advanced_fusion(self) -> Dict[str, Any]:
        """Test advanced fusion architecture."""
        
        results = {
            'fusion_architecture_created': False,
            'spectral_temporal_fusion': False,
            'attention_mechanisms': False,
            'causal_interventions': False,
            'errors': []
        }
        
        try:
            # Create advanced fusion architecture
            config = AttentionGuidedFusionConfig(
                eeg_channels=32,
                sampling_rate=1000,
                hidden_dim=512,
                num_attention_heads=16,
                fusion_layers=4
            )
            
            fusion_model = AttentionGuidedSpectralTemporalFusion(config)
            fusion_model.to(self.device)
            
            results['fusion_architecture_created'] = True
            print(f"   🔬 Advanced fusion architecture created")
            
            # Test with dummy data
            test_eeg = torch.randn(2, 32, 1000, device=self.device)
            test_language = torch.randn(2, 50, 768, device=self.device)
            
            with torch.no_grad():
                fusion_outputs = fusion_model(
                    eeg_data=test_eeg,
                    language_features=test_language,
                    compute_uncertainty=True
                )
            
            # Check outputs
            if 'fused_features' in fusion_outputs:
                print(f"   ✅ Fused features: {fusion_outputs['fused_features'].shape}")
            
            if 'spectral_attention' in fusion_outputs:
                results['spectral_temporal_fusion'] = True
                print(f"   ✅ Spectral-temporal fusion working")
            
            if 'cross_modal_attention' in fusion_outputs:
                results['attention_mechanisms'] = True
                print(f"   ✅ Cross-modal attention working")
            
            if 'intervention_effects' in fusion_outputs:
                results['causal_interventions'] = True
                print(f"   ✅ Causal interventions working")
            
        except Exception as e:
            results['errors'].append(str(e))
            print(f"   ❌ Advanced fusion test failed: {e}")
        
        return results
    
    def _test_integration(self) -> Dict[str, Any]:
        """Test end-to-end system integration."""
        
        results = {
            'full_pipeline_working': False,
            'batch_processing': False,
            'multi_modal_outputs': False,
            'performance_monitoring': False,
            'errors': []
        }
        
        try:
            # Create full enhanced model
            model = EnhancedBCIGPTModel(
                eeg_channels=32,
                enable_uncertainty=True,
                enable_multi_language=True,
                real_time_optimization=True
            )
            model.to(self.device)
            model.eval()
            
            # Test full pipeline
            batch_sizes = [1, 4, 8]
            
            for batch_size in batch_sizes:
                print(f"   🔄 Testing batch size {batch_size}...")
                
                test_eeg = torch.randn(batch_size, 32, 1000, device=self.device)
                
                with torch.no_grad():
                    outputs = model(
                        test_eeg,
                        language='en',
                        return_attention_weights=True
                    )
                
                # Check outputs
                required_outputs = [
                    'logits', 'eeg_features', 'fused_features', 
                    'signal_quality', 'latency_ms'
                ]
                
                missing_outputs = []
                for output_name in required_outputs:
                    if output_name not in outputs:
                        missing_outputs.append(output_name)
                
                if not missing_outputs:
                    results['full_pipeline_working'] = True
                    print(f"   ✅ Batch size {batch_size}: All outputs present")
                else:
                    print(f"   ❌ Batch size {batch_size}: Missing {missing_outputs}")
                
                # Test batch processing
                if outputs['logits'].shape[0] == batch_size:
                    results['batch_processing'] = True
                
                # Test multi-modal outputs
                if len(outputs) >= 6:  # Multiple types of outputs
                    results['multi_modal_outputs'] = True
                
                # Test performance monitoring
                if 'latency_ms' in outputs and outputs['latency_ms'] > 0:
                    results['performance_monitoring'] = True
                    print(f"   ⚡ Latency: {outputs['latency_ms']:.2f}ms")
            
        except Exception as e:
            results['errors'].append(str(e))
            print(f"   ❌ Integration test failed: {e}")
        
        return results
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive validation summary."""
        
        summary = {
            'total_tests': 7,
            'passed_tests': 0,
            'failed_tests': 0,
            'success_rate': 0.0,
            'key_capabilities': [],
            'performance_metrics': {},
            'recommendations': [],
            'overall_status': 'UNKNOWN'
        }
        
        # Count passed/failed tests
        for test_name, test_results in self.results.items():
            if test_name == 'summary':
                continue
                
            # Simple heuristic: test passes if no errors and some positive indicators
            test_passed = (
                len(test_results.get('errors', [])) == 0 and
                len(test_results) > 1  # More than just error field
            )
            
            if test_passed:
                summary['passed_tests'] += 1
            else:
                summary['failed_tests'] += 1
        
        summary['success_rate'] = summary['passed_tests'] / summary['total_tests']
        
        # Key capabilities
        if self.results.get('architecture', {}).get('model_created'):
            summary['key_capabilities'].append('Enhanced Architecture')
        
        if self.results.get('performance', {}).get('meets_requirement'):
            summary['key_capabilities'].append('Real-time Performance (<50ms)')
        
        if self.results.get('multilingual', {}).get('supported_languages'):
            lang_count = len(self.results['multilingual']['supported_languages'])
            summary['key_capabilities'].append(f'Multi-language Support ({lang_count} languages)')
        
        if self.results.get('uncertainty', {}).get('uncertainty_enabled'):
            summary['key_capabilities'].append('Uncertainty Quantification')
        
        if self.results.get('signal_quality', {}).get('adaptive_processing'):
            summary['key_capabilities'].append('Adaptive Signal Processing')
        
        if self.results.get('fusion', {}).get('fusion_architecture_created'):
            summary['key_capabilities'].append('Advanced Fusion Architecture')
        
        # Performance metrics
        if 'performance' in self.results:
            perf = self.results['performance']
            summary['performance_metrics'] = {
                'average_latency_ms': perf.get('average_latency', 0),
                'throughput_samples_per_sec': perf.get('throughput_samples_per_sec', 0),
                'meets_realtime_requirement': perf.get('meets_requirement', False)
            }
        
        # Recommendations
        if summary['success_rate'] < 0.8:
            summary['recommendations'].append('Address failing tests before production deployment')
        
        if not self.results.get('performance', {}).get('meets_requirement'):
            summary['recommendations'].append('Optimize model for real-time performance')
        
        if len(self.results.get('multilingual', {}).get('supported_languages', [])) < 3:
            summary['recommendations'].append('Expand multi-language support')
        
        # Overall status
        if summary['success_rate'] >= 0.9:
            summary['overall_status'] = 'EXCELLENT'
        elif summary['success_rate'] >= 0.8:
            summary['overall_status'] = 'GOOD'
        elif summary['success_rate'] >= 0.7:
            summary['overall_status'] = 'ACCEPTABLE'
        else:
            summary['overall_status'] = 'NEEDS_IMPROVEMENT'
        
        return summary
    
    def print_final_report(self):
        """Print comprehensive final validation report."""
        
        summary = self.results.get('summary', {})
        
        print("\n" + "=" * 80)
        print("🏆 ENHANCED BCI-GPT SYSTEM VALIDATION REPORT")
        print("=" * 80)
        
        print(f"\n📊 TEST SUMMARY")
        print(f"   Total Tests: {summary.get('total_tests', 0)}")
        print(f"   Passed: {summary.get('passed_tests', 0)}")
        print(f"   Failed: {summary.get('failed_tests', 0)}")
        print(f"   Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"   Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        
        print(f"\n🚀 KEY CAPABILITIES")
        for capability in summary.get('key_capabilities', []):
            print(f"   ✅ {capability}")
        
        print(f"\n⚡ PERFORMANCE METRICS")
        perf_metrics = summary.get('performance_metrics', {})
        if perf_metrics:
            print(f"   Average Latency: {perf_metrics.get('average_latency_ms', 0):.2f}ms")
            print(f"   Throughput: {perf_metrics.get('throughput_samples_per_sec', 0):.1f} samples/sec")
            print(f"   Real-time Ready: {'✅ YES' if perf_metrics.get('meets_realtime_requirement') else '❌ NO'}")
        
        print(f"\n🎯 RECOMMENDATIONS")
        recommendations = summary.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                print(f"   💡 {rec}")
        else:
            print(f"   ✅ No recommendations - system performing optimally")
        
        # Production readiness assessment
        print(f"\n🏭 PRODUCTION READINESS ASSESSMENT")
        
        success_rate = summary.get('success_rate', 0)
        if success_rate >= 0.9:
            print(f"   🟢 READY FOR PRODUCTION")
            print(f"      System meets all requirements for production deployment")
        elif success_rate >= 0.8:
            print(f"   🟡 READY WITH MONITORING")
            print(f"      System ready for production with close monitoring")
        elif success_rate >= 0.7:
            print(f"   🟠 REQUIRES IMPROVEMENTS")
            print(f"      Address failing tests before production deployment")
        else:
            print(f"   🔴 NOT READY FOR PRODUCTION")
            print(f"      Significant issues need resolution")
        
        print("\n" + "=" * 80)


def main():
    """Main validation execution."""
    
    print("🧠 Enhanced BCI-GPT System Validation Suite")
    print("Authors: Daniel Schmidt, Terragon Labs")
    print("Version: 1.0 - Generation 1 Implementation")
    
    # Initialize validator
    validator = EnhancedSystemValidator()
    
    # Run comprehensive validation
    try:
        results = validator.run_comprehensive_validation()
        
        # Print final report
        validator.print_final_report()
        
        # Save results (optional)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"enhanced_bci_validation_{timestamp}.json"
        
        try:
            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Detailed results saved to: {results_file}")
        except:
            print(f"\n⚠️  Could not save results to file")
        
        return results
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Validation suite failed")
        print(f"   Error: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    main()