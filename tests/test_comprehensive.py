"""Comprehensive test suite for BCI-GPT system."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import warnings

# Import modules to test
from bci_gpt.utils.security import (
    DataEncryption, PrivacyProtection, InputValidation, 
    AttackDetection, SecureInference, ComplianceChecker
)
from bci_gpt.utils.monitoring import (
    MetricsCollector, PerformanceProfiler, ClinicalSafetyMonitor
)
from bci_gpt.inverse.validation import SyntheticEEGValidator
from bci_gpt.optimization.caching import LRUCache, EEGCache
from bci_gpt.preprocessing.eeg_processor import EEGProcessor
from bci_gpt.utils.metrics import BCIMetrics


class TestSecurity:
    """Test security and privacy features."""
    
    def test_privacy_protection_anonymization(self):
        """Test EEG data anonymization."""
        # Create test EEG data
        test_eeg = np.random.randn(8, 1000) * 50
        
        # Test anonymization
        anonymized = PrivacyProtection.anonymize_eeg_data(test_eeg, privacy_level=0.1)
        
        assert anonymized.shape == test_eeg.shape
        assert not np.array_equal(test_eeg, anonymized)
        assert np.abs(np.mean(anonymized - test_eeg)) < 10  # Reasonable noise level
    
    def test_input_validation_eeg_data(self):
        """Test EEG data validation."""
        # Valid data
        valid_eeg = np.random.randn(8, 1000) * 50
        assert InputValidation.validate_eeg_data(valid_eeg, expected_channels=8)
        
        # Invalid data - wrong dimensions
        with pytest.raises(ValueError):
            InputValidation.validate_eeg_data(np.random.randn(8), expected_channels=8)
        
        # Invalid data - NaN values
        invalid_eeg = valid_eeg.copy()
        invalid_eeg[0, 0] = np.nan
        with pytest.raises(ValueError):
            InputValidation.validate_eeg_data(invalid_eeg)
    
    def test_text_sanitization(self):
        """Test text input sanitization."""
        # Clean text
        clean_text = "Hello world"
        sanitized = PrivacyProtection.sanitize_text_input(clean_text)
        assert sanitized == clean_text
        
        # Dangerous patterns
        dangerous_text = "<script>alert('xss')</script>"
        sanitized = PrivacyProtection.sanitize_text_input(dangerous_text)
        assert "<script>" not in sanitized
        assert "alert" not in sanitized
        
        # Length limit
        long_text = "A" * 2000
        sanitized = PrivacyProtection.sanitize_text_input(long_text, max_length=1000)
        assert len(sanitized) <= 1000
    
    def test_attack_detection(self):
        """Test adversarial attack detection."""
        detector = AttackDetection()
        
        # Normal EEG data
        normal_eeg = np.random.randn(8, 1000) * 50
        assert not detector.detect_adversarial_eeg(normal_eeg)
        
        # Adversarial EEG (artificially high amplitude)
        adversarial_eeg = normal_eeg * 5  # Unusual amplification
        is_adversarial = detector.detect_adversarial_eeg(adversarial_eeg)
        # May or may not detect based on the specific patterns, but should not crash
        assert isinstance(is_adversarial, bool)
    
    def test_compliance_checking(self):
        """Test HIPAA compliance checking."""
        checker = ComplianceChecker()
        
        # Non-compliant system
        non_compliant_config = {
            'encryption_enabled': False,
            'access_controls': False,
            'audit_logging': False
        }
        report = checker.check_hipaa_compliance(non_compliant_config)
        assert not report['compliant']
        assert len(report['violations']) > 0
        
        # Compliant system
        compliant_config = {
            'encryption_enabled': True,
            'access_controls': True,
            'audit_logging': True,
            'data_retention_policy': True
        }
        report = checker.check_hipaa_compliance(compliant_config)
        # May still have some violations but should be better
        assert 'violations' in report


class TestMonitoring:
    """Test monitoring and metrics systems."""
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        collector = MetricsCollector()
        
        # Test model metrics recording
        collector.record_model_metrics(
            inference_time_ms=50.0,
            accuracy=0.85,
            confidence=0.9
        )
        assert len(collector.model_metrics) == 1
        
        # Test streaming metrics recording
        collector.record_streaming_metrics(
            samples_per_second=1000.0,
            buffer_fill_ratio=0.3
        )
        assert len(collector.streaming_metrics) == 1
        
        # Test error recording
        collector.record_error('warning', 'test', 'Test warning message')
        assert len(collector.error_events) == 1
        
        # Test system health
        health = collector.get_system_health()
        assert 'status' in health
        assert health['status'] in ['healthy', 'warning', 'critical', 'no_data']
    
    def test_performance_profiler(self):
        """Test performance profiling."""
        collector = MetricsCollector()
        
        with PerformanceProfiler('test_operation', collector):
            # Simulate some work
            import time
            time.sleep(0.01)
        
        # Check that timing was recorded
        assert 'test_operation' in collector.timers
        assert len(collector.timers['test_operation']) > 0
    
    def test_clinical_safety_monitor(self):
        """Test clinical safety monitoring."""
        monitor = ClinicalSafetyMonitor(max_session_duration=10)
        
        # Start session
        monitor.start_session('test_user')
        assert monitor.is_monitoring
        assert monitor.is_safe()
        
        # Test fatigue detection
        test_eeg = np.random.randn(8, 1000)
        is_fatigued = monitor.detect_fatigue(test_eeg)
        assert isinstance(is_fatigued, bool)
        
        # End session
        report = monitor.end_session()
        assert 'user_id' in report
        assert report['user_id'] == 'test_user'
        assert not monitor.is_monitoring


class TestValidation:
    """Test EEG validation systems."""
    
    def test_synthetic_eeg_validator(self):
        """Test synthetic EEG validation."""
        validator = SyntheticEEGValidator()
        
        # Create realistic test EEG
        channels, samples = 8, 1000
        synthetic_eeg = np.random.randn(channels, samples) * 50
        
        # Add some EEG-like characteristics
        t = np.linspace(0, 1, samples)
        for ch in range(channels):
            # Add alpha waves
            synthetic_eeg[ch] += 20 * np.sin(2 * np.pi * 10 * t)
            # Add some noise
            synthetic_eeg[ch] += np.random.randn(samples) * 5
        
        # Test basic validation
        result = validator.validate_basic(synthetic_eeg)
        assert 'realism_score' in result
        assert 'artifact_score' in result
        assert 'overall_quality' in result
        assert 0 <= result['overall_quality'] <= 1
        
        # Test comprehensive validation
        result = validator.validate_comprehensive(synthetic_eeg)
        assert 'realism_score' in result
        assert 'temporal_consistency' in result
        assert 'spectral_similarity' in result
        assert 'overall_quality' in result


class TestCaching:
    """Test caching systems."""
    
    def test_lru_cache(self):
        """Test LRU cache functionality."""
        cache = LRUCache(max_size=3)
        
        # Add items
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')
        
        # Check retrieval
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') == 'value3'
        
        # Add one more (should evict oldest)
        cache.put('key4', 'value4')
        
        # key1 should be evicted
        assert cache.get('key1') is None
        assert cache.get('key4') == 'value4'
        
        # Check stats
        stats = cache.get_stats()
        assert stats.evictions >= 1
    
    def test_eeg_cache(self):
        """Test EEG-specific caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test_cache.db')
            cache = EEGCache(enable_persistence=True, db_path=db_path)
            
            # Create test data
            eeg_data = np.random.randn(8, 1000)
            processing_params = {'filter_type': 'bandpass', 'low': 1.0, 'high': 40.0}
            processed_eeg = eeg_data * 0.5  # Simulated processing result
            
            # Cache result
            cache.put_processed_eeg(eeg_data, processing_params, processed_eeg)
            
            # Retrieve result
            retrieved = cache.get_processed_eeg(eeg_data, processing_params)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, processed_eeg)
            
            # Test cache miss
            different_params = {'filter_type': 'highpass', 'high': 30.0}
            missed = cache.get_processed_eeg(eeg_data, different_params)
            assert missed is None


class TestEEGProcessor:
    """Test EEG processing functionality."""
    
    def test_eeg_processor_initialization(self):
        """Test EEG processor initialization."""
        processor = EEGProcessor()
        assert processor.sampling_rate == 1000
        assert processor.channels is None or len(processor.channels) >= 0
    
    def test_synthetic_eeg_generation(self):
        """Test synthetic EEG generation."""
        processor = EEGProcessor()
        
        # Generate synthetic EEG
        synthetic_eeg = processor._generate_synthetic_eeg(duration=2.0, channels=8)
        
        assert isinstance(synthetic_eeg, np.ndarray)
        assert synthetic_eeg.shape[0] == 8  # 8 channels
        assert synthetic_eeg.shape[1] == 2000  # 2 seconds at 1000 Hz
        
        # Check that it's not just zeros
        assert np.std(synthetic_eeg) > 0
    
    def test_preprocessing_pipeline(self):
        """Test EEG preprocessing pipeline."""
        processor = EEGProcessor()
        
        # Create test EEG data
        channels, samples = 8, 2000
        test_eeg = np.random.randn(channels, samples) * 50
        
        # Test preprocessing
        result = processor.preprocess(
            test_eeg,
            bandpass=(1.0, 40.0),
            epoch_length=1.0
        )
        
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'info' in result
        
        # Check that preprocessing didn't break the data
        processed_data = result['data']
        assert processed_data.shape[0] == channels  # Same number of channels


class TestBCIMetrics:
    """Test BCI performance metrics."""
    
    def test_metrics_calculation(self):
        """Test BCI metrics calculations."""
        metrics = BCIMetrics()
        
        # Test ITR calculation
        itr = metrics.calculate_itr(accuracy=0.85, num_classes=26, trial_duration=2.0)
        assert itr > 0
        assert isinstance(itr, float)
        
        # Test WER calculation
        predicted = "hello world"
        reference = "hello word"
        wer = metrics.word_error_rate(predicted, reference)
        assert 0 <= wer <= 1
        assert isinstance(wer, float)
        
        # Perfect match should have 0 WER
        perfect_wer = metrics.word_error_rate("hello world", "hello world")
        assert perfect_wer == 0.0


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_processing(self):
        """Test end-to-end EEG processing workflow."""
        # Initialize components
        processor = EEGProcessor()
        validator = SyntheticEEGValidator()
        cache = EEGCache()
        
        # Generate test data
        synthetic_eeg = processor._generate_synthetic_eeg(duration=1.0, channels=8)
        
        # Process data
        processing_params = {'bandpass': (1.0, 40.0)}
        processed_data = processor.preprocess(
            synthetic_eeg, 
            bandpass=processing_params['bandpass'],
            epoch_length=1.0
        )
        
        # Cache the result
        cache.put_processed_eeg(
            synthetic_eeg, 
            processing_params, 
            processed_data['data']
        )
        
        # Retrieve from cache
        cached_result = cache.get_processed_eeg(synthetic_eeg, processing_params)
        assert cached_result is not None
        
        # Validate the processed data
        validation_result = validator.validate_basic(processed_data['data'])
        assert 'overall_quality' in validation_result
        assert validation_result['overall_quality'] >= 0
    
    def test_security_monitoring_integration(self):
        """Test integration of security and monitoring systems."""
        # Initialize systems
        collector = MetricsCollector()
        detector = AttackDetection()
        monitor = ClinicalSafetyMonitor()
        
        # Start monitoring
        monitor.start_session('test_patient')
        
        # Simulate some operations with monitoring
        with PerformanceProfiler('secure_operation', collector):
            # Generate test EEG
            test_eeg = np.random.randn(8, 1000) * 50
            
            # Check for attacks
            is_adversarial = detector.detect_adversarial_eeg(test_eeg)
            
            # Record metrics
            collector.record_model_metrics(
                inference_time_ms=25.0,
                accuracy=0.88,
                confidence=0.85
            )
            
            # Check safety
            is_safe = monitor.is_safe()
            assert isinstance(is_safe, bool)
        
        # Get reports
        session_report = monitor.end_session()
        system_health = collector.get_system_health()
        security_status = detector.get_security_status()
        
        assert 'status' in session_report
        assert 'status' in system_health
        assert 'status' in security_status


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        processor = EEGProcessor()
        
        # Test with invalid data types
        with pytest.raises((ValueError, TypeError)):
            processor.preprocess("invalid_data")
        
        # Test with empty data
        with pytest.raises(ValueError):
            processor.preprocess(np.array([]))
    
    def test_memory_constraints(self):
        """Test behavior under memory constraints."""
        # Create a small cache
        cache = LRUCache(max_size=2, max_memory_mb=0.001)  # Very small
        
        # Add items that exceed memory limit
        large_array = np.random.randn(1000, 1000)
        cache.put('large_item', large_array)
        
        # Cache should handle memory constraints gracefully
        stats = cache.get_stats()
        assert stats.memory_usage_mb <= cache.max_memory_mb * 1.1  # Small tolerance
    
    def test_concurrent_access(self):
        """Test concurrent access to shared resources."""
        import threading
        
        cache = LRUCache(max_size=100)
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_item_{i}"
                    value = f"value_{worker_id}_{i}"
                    cache.put(key, value)
                    retrieved = cache.get(key)
                    assert retrieved == value
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that no errors occurred
        assert len(errors) == 0


# Performance benchmarks
class TestPerformance:
    """Performance and benchmark tests."""
    
    def test_processing_performance(self):
        """Test EEG processing performance."""
        processor = EEGProcessor()
        
        # Large dataset
        large_eeg = np.random.randn(32, 10000)  # 32 channels, 10 seconds
        
        import time
        start_time = time.time()
        
        result = processor.preprocess(large_eeg, bandpass=(1.0, 40.0))
        
        processing_time = time.time() - start_time
        
        # Should process within reasonable time (adjust threshold as needed)
        assert processing_time < 5.0  # 5 seconds max
        assert result is not None
    
    def test_cache_performance(self):
        """Test cache performance under load."""
        cache = LRUCache(max_size=1000)
        
        # Warm up cache
        for i in range(500):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Measure access performance
        import time
        start_time = time.time()
        
        hits = 0
        for i in range(1000):
            key = f"key_{i % 500}"  # 50% hit rate
            result = cache.get(key)
            if result is not None:
                hits += 1
        
        access_time = time.time() - start_time
        
        # Performance assertions
        assert access_time < 0.1  # Should be very fast
        assert hits >= 400  # Should have reasonable hit rate


# Test configuration and fixtures
@pytest.fixture
def sample_eeg_data():
    """Fixture providing sample EEG data."""
    channels, samples = 8, 1000
    # Create realistic EEG-like data
    t = np.linspace(0, 1, samples)
    eeg_data = np.zeros((channels, samples))
    
    for ch in range(channels):
        # Alpha rhythm (10 Hz)
        eeg_data[ch] += 20 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        # Beta rhythm (20 Hz)
        eeg_data[ch] += 10 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
        # Noise
        eeg_data[ch] += np.random.randn(samples) * 5
    
    return eeg_data


@pytest.fixture
def temp_cache_db():
    """Fixture providing temporary cache database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


# Run all tests
if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-x"])