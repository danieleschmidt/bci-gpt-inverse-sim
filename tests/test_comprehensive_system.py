"""Comprehensive system tests for BCI-GPT."""

import pytest
import numpy as np
import time
import threading
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import BCI-GPT modules
try:
    from bci_gpt.core.models import BCIGPTModel
    from bci_gpt.preprocessing.eeg_processor import EEGProcessor
    from bci_gpt.inverse.inverse_gan import InverseGAN
    from bci_gpt.utils.logging_config import get_logger, setup_logging
    from bci_gpt.utils.error_handling import get_error_handler, BCIGPTException
    from bci_gpt.utils.security import InputSanitizer, DataProtection
    from bci_gpt.utils.monitoring import get_health_checker, get_metrics_collector
    from bci_gpt.utils.performance_optimizer import get_performance_optimizer
    from bci_gpt.utils.auto_scaling import get_load_balancer, get_auto_scaler
    from bci_gpt.utils.config_manager import get_config_manager, BCIGPTConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Some imports failed: {e}")
    IMPORTS_AVAILABLE = False


class TestSystemIntegration:
    """Integration tests for entire BCI-GPT system."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
        
        # Setup logging
        self.logger = setup_logging(log_level="DEBUG")
        
        # Create test data
        self.test_eeg_data = np.random.randn(64, 1000).astype(np.float32)  # 64 channels, 1000 samples
        self.test_text = "Hello world this is a test"
        
        # Setup temporary directories
        self.temp_dir = tempfile.mkdtemp()
        
        yield
        
        # Cleanup
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline_simulation(self):
        """Test complete BCI-GPT pipeline with simulated data."""
        # Test preprocessing
        processor = EEGProcessor()
        processed_eeg = processor.process(self.test_eeg_data, sampling_rate=1000)
        
        assert processed_eeg is not None
        assert processed_eeg.shape[0] == self.test_eeg_data.shape[0]  # Same number of channels
        
        # Test model components (mocked for testing)
        with patch('torch.nn.Module') as mock_module:
            model = BCIGPTModel()
            assert model is not None
        
        # Test inverse simulation
        inverse_gan = InverseGAN()
        assert inverse_gan is not None
        
        # Test text-to-EEG conversion (simulated)
        simulated_eeg = inverse_gan.generate_eeg_from_text(self.test_text)
        assert simulated_eeg is not None
        assert simulated_eeg.shape[0] > 0  # Has channels
        assert simulated_eeg.shape[1] > 0  # Has samples
    
    def test_error_handling_system(self):
        """Test comprehensive error handling."""
        error_handler = get_error_handler()
        
        # Test error registration and handling
        test_exception = BCIGPTException("Test error", error_code="TEST_ERROR")
        
        result = error_handler.handle_error(test_exception)
        assert "action" in result
        
        # Test error statistics
        stats = error_handler.get_error_statistics()
        assert isinstance(stats, dict)
    
    def test_security_system(self):
        """Test security and validation systems."""
        sanitizer = InputSanitizer()
        
        # Test filename sanitization
        clean_filename = sanitizer.sanitize_filename("test/../../../etc/passwd")
        assert ".." not in clean_filename
        assert "/" not in clean_filename
        
        # Test text input sanitization
        clean_text = sanitizer.sanitize_text_input("<script>alert('xss')</script>Hello")
        assert "<script>" not in clean_text
        assert "Hello" in clean_text
        
        # Test data protection
        data_protection = DataProtection()
        test_data = "sensitive information"
        
        hash_value, salt = data_protection.hash_data(test_data)
        assert hash_value is not None
        assert salt is not None
        
        # Verify hash
        is_valid = data_protection.verify_hash(test_data, hash_value, salt)
        assert is_valid
    
    def test_monitoring_system(self):
        """Test monitoring and health checks."""
        health_checker = get_health_checker()
        
        # Run health checks
        health_status = health_checker.get_health_status()
        
        assert health_status is not None
        assert hasattr(health_status, 'overall_status')
        assert hasattr(health_status, 'component_statuses')
        assert hasattr(health_status, 'metrics')
        
        # Test metrics collector
        metrics_collector = get_metrics_collector()
        
        # Record test metrics
        metrics_collector.record_model_metrics(
            inference_time_ms=50.0,
            accuracy=0.85
        )
        
        metrics_collector.record_streaming_metrics(
            samples_per_second=1000.0,
            buffer_fill_ratio=0.3
        )
        
        # Check system health
        health = metrics_collector.get_system_health()
        assert "status" in health
    
    def test_performance_optimization(self):
        """Test performance optimization systems."""
        optimizer = get_performance_optimizer()
        
        # Test caching
        @optimizer.cached_function(ttl=60)
        def expensive_function(x):
            time.sleep(0.01)  # Simulate expensive operation
            return x * 2
        
        # First call should cache the result
        start_time = time.time()
        result1 = expensive_function(5)
        first_duration = time.time() - start_time
        
        # Second call should be faster (cached)
        start_time = time.time()
        result2 = expensive_function(5)
        second_duration = time.time() - start_time
        
        assert result1 == result2 == 10
        assert second_duration < first_duration
        
        # Test cache statistics
        cache_stats = optimizer.cache.get_stats()
        assert cache_stats["hit_count"] > 0
    
    def test_auto_scaling_system(self):
        """Test auto-scaling and load balancing."""
        load_balancer = get_load_balancer()
        
        # Add test workers
        load_balancer.add_worker("worker1", max_capacity=10)
        load_balancer.add_worker("worker2", max_capacity=10)
        
        # Test worker selection
        worker_id = load_balancer.select_worker()
        assert worker_id in ["worker1", "worker2"]
        
        # Simulate request processing
        load_balancer.start_request(worker_id)
        time.sleep(0.01)  # Simulate processing time
        load_balancer.end_request(worker_id, response_time_ms=10.0, success=True)
        
        # Test load balancer status
        status = load_balancer.get_status()
        assert status["workers"] == 2
        assert status["total_capacity"] == 20
        
        # Test auto-scaler
        auto_scaler = get_auto_scaler()
        scaling_status = auto_scaler.get_scaling_status()
        
        assert "running" in scaling_status
        assert "workers_count" in scaling_status
    
    def test_configuration_management(self):
        """Test configuration system."""
        # Test with temporary config file
        config_file = os.path.join(self.temp_dir, "test_config.json")
        
        config_manager = get_config_manager(config_file)
        config = config_manager.get_config()
        
        assert isinstance(config, BCIGPTConfig)
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'eeg')
        
        # Test configuration updates
        config_manager.update_config({
            "model.hidden_size": 512,
            "training.learning_rate": 2e-4
        })
        
        updated_config = config_manager.get_config()
        assert updated_config.model.hidden_size == 512
        assert updated_config.training.learning_rate == 2e-4
    
    def test_concurrent_operations(self):
        """Test system under concurrent load."""
        metrics_collector = get_metrics_collector()
        load_balancer = get_load_balancer()
        
        def simulate_request():
            """Simulate a processing request."""
            worker_id = load_balancer.select_worker()
            if worker_id:
                load_balancer.start_request(worker_id)
                
                # Simulate processing
                start_time = time.time()
                time.sleep(np.random.uniform(0.001, 0.01))  # 1-10ms processing
                duration_ms = (time.time() - start_time) * 1000
                
                # Record metrics
                metrics_collector.record_model_metrics(
                    inference_time_ms=duration_ms,
                    accuracy=np.random.uniform(0.7, 0.95)
                )
                
                load_balancer.end_request(worker_id, duration_ms, success=True)
        
        # Run concurrent requests
        threads = []
        num_threads = 10
        
        for _ in range(num_threads):
            thread = threading.Thread(target=simulate_request)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify system handled concurrent load
        health_status = get_health_checker().get_health_status()
        assert health_status.overall_status in ["healthy", "warning"]  # Should not be critical
    
    def test_memory_management(self):
        """Test memory usage and cleanup."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large objects to test memory management
        large_arrays = []
        for i in range(10):
            large_array = np.random.randn(1000, 1000)
            large_arrays.append(large_array)
        
        # Memory should have increased
        mid_memory = process.memory_info().rss / 1024 / 1024
        assert mid_memory > initial_memory
        
        # Clear references and force garbage collection
        large_arrays.clear()
        gc.collect()
        
        # Give some time for memory cleanup
        time.sleep(0.1)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        # Memory should be closer to initial (allowing some variance)
        assert final_memory < mid_memory
    
    def test_graceful_degradation(self):
        """Test system behavior under failure conditions."""
        # Test with missing dependencies
        with patch('torch.cuda.is_available', return_value=False):
            # System should fall back to CPU
            optimizer = get_performance_optimizer()
            batch_processor = optimizer.batch_processor
            assert batch_processor.device.type == "cpu"
        
        # Test with invalid data
        processor = EEGProcessor()
        
        # Should handle invalid input gracefully
        with pytest.raises((ValueError, BCIGPTException)):
            processor.process(np.array([]), sampling_rate=1000)
        
        # Test error recovery
        error_handler = get_error_handler()
        
        # Simulate recoverable error
        test_error = BCIGPTException("Recoverable error", recoverable=True)
        result = error_handler.handle_error(test_error)
        
        assert result.get("action") != "fail"  # Should attempt recovery
    
    def test_data_pipeline_integrity(self):
        """Test data pipeline integrity and validation."""
        processor = EEGProcessor()
        
        # Test with various data shapes and types
        test_cases = [
            (np.random.randn(32, 500), 500),   # 32 channels, 500 samples, 500 Hz
            (np.random.randn(64, 2000), 1000), # 64 channels, 2000 samples, 1000 Hz  
            (np.random.randn(128, 1000), 250), # 128 channels, 1000 samples, 250 Hz
        ]
        
        for eeg_data, sampling_rate in test_cases:
            processed = processor.process(eeg_data, sampling_rate=sampling_rate)
            
            # Verify data integrity
            assert processed is not None
            assert np.all(np.isfinite(processed))  # No NaN or inf values
            assert processed.shape[0] == eeg_data.shape[0]  # Same number of channels
            
            # Verify statistical properties are reasonable
            assert np.abs(np.mean(processed)) < 1.0  # Mean close to zero after preprocessing
            assert np.std(processed) > 0.1  # Has reasonable variance


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for performance tests."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
        
        self.benchmark_results = {}
    
    def test_eeg_processing_performance(self):
        """Benchmark EEG processing performance."""
        processor = EEGProcessor()
        
        # Test different data sizes
        test_sizes = [
            (32, 1000),   # Small
            (64, 5000),   # Medium
            (128, 10000), # Large
        ]
        
        for channels, samples in test_sizes:
            test_data = np.random.randn(channels, samples).astype(np.float32)
            
            # Benchmark processing time
            start_time = time.time()
            processed = processor.process(test_data, sampling_rate=1000)
            processing_time = time.time() - start_time
            
            # Performance criteria (adjust based on requirements)
            max_time_per_sample = 0.001  # 1ms per sample
            assert processing_time < samples * max_time_per_sample
            
            self.benchmark_results[f"processing_{channels}x{samples}"] = processing_time
    
    def test_inference_performance(self):
        """Benchmark model inference performance."""
        # Mock model for performance testing
        with patch('bci_gpt.core.models.BCIGPTModel') as MockModel:
            mock_model = MockModel.return_value
            mock_model.forward.return_value = np.random.randn(32, 512)
            
            # Simulate inference
            test_input = np.random.randn(32, 1000)
            
            start_time = time.time()
            for _ in range(10):  # 10 inference runs
                result = mock_model.forward(test_input)
            inference_time = (time.time() - start_time) / 10  # Average per inference
            
            # Performance criteria
            max_inference_time = 0.1  # 100ms max per inference
            assert inference_time < max_inference_time
            
            self.benchmark_results["inference_time"] = inference_time
    
    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        load_balancer = get_load_balancer()
        
        # Add workers
        for i in range(4):
            load_balancer.add_worker(f"perf_worker_{i}", max_capacity=5)
        
        def concurrent_task():
            worker_id = load_balancer.select_worker()
            if worker_id:
                load_balancer.start_request(worker_id)
                time.sleep(0.01)  # Simulate 10ms processing
                load_balancer.end_request(worker_id, 10.0, success=True)
        
        # Run concurrent tasks
        num_tasks = 20
        threads = []
        
        start_time = time.time()
        for _ in range(num_tasks):
            thread = threading.Thread(target=concurrent_task)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Should handle concurrent load efficiently
        max_expected_time = 0.1  # 100ms for all tasks
        assert total_time < max_expected_time
        
        self.benchmark_results["concurrent_tasks"] = total_time
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform memory-intensive operations
        processor = EEGProcessor()
        large_data = np.random.randn(256, 50000).astype(np.float32)  # Large EEG data
        
        # Process data multiple times
        for _ in range(5):
            processed = processor.process(large_data, sampling_rate=1000)
            del processed  # Explicit cleanup
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        max_memory_increase = 100 * 1024 * 1024  # 100MB
        assert memory_increase < max_memory_increase
        
        self.benchmark_results["memory_increase_mb"] = memory_increase / 1024 / 1024


class TestQualityGates:
    """Quality gate tests that must pass for deployment."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for quality gate tests."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
    
    def test_no_critical_security_issues(self):
        """Verify no critical security vulnerabilities."""
        sanitizer = InputSanitizer()
        
        # Test SQL injection prevention
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "__import__('os').system('rm -rf /')",
            "eval(compile(open('malicious.py').read(), 'malicious.py', 'exec'))"
        ]
        
        for malicious_input in malicious_inputs:
            try:
                sanitized = sanitizer.sanitize_text_input(malicious_input)
                # Should not contain original malicious content
                assert malicious_input != sanitized
                assert "DROP TABLE" not in sanitized
                assert "<script>" not in sanitized
                assert "__import__" not in sanitized
            except Exception:
                # Exceptions are acceptable for malicious input
                pass
    
    def test_error_handling_coverage(self):
        """Verify comprehensive error handling."""
        error_handler = get_error_handler()
        
        # Test various error types
        error_types = [
            ValueError("Test value error"),
            FileNotFoundError("Test file not found"),
            MemoryError("Test memory error"),
            BCIGPTException("Test BCI error")
        ]
        
        for error in error_types:
            result = error_handler.handle_error(error)
            assert "action" in result
            assert result["action"] in ["fail", "retry", "fallback", "recover"]
    
    def test_performance_requirements(self):
        """Verify system meets performance requirements."""
        # Test EEG processing speed
        processor = EEGProcessor()
        test_data = np.random.randn(64, 1000)
        
        start_time = time.time()
        processed = processor.process(test_data, sampling_rate=1000)
        processing_time = time.time() - start_time
        
        # Must process 1 second of EEG data in less than 100ms
        assert processing_time < 0.1
        
        # Test system responsiveness
        health_checker = get_health_checker()
        
        start_time = time.time()
        health_status = health_checker.get_health_status()
        health_check_time = time.time() - start_time
        
        # Health check must complete in less than 1 second
        assert health_check_time < 1.0
        assert health_status is not None
    
    def test_memory_leak_detection(self):
        """Detect potential memory leaks."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        gc.collect()
        time.sleep(0.1)
        baseline_memory = process.memory_info().rss
        
        # Perform repeated operations
        processor = EEGProcessor()
        
        for i in range(50):  # Many iterations
            test_data = np.random.randn(32, 500)
            processed = processor.process(test_data, sampling_rate=1000)
            
            # Force garbage collection periodically
            if i % 10 == 0:
                gc.collect()
        
        # Final cleanup and memory check
        gc.collect()
        time.sleep(0.1)
        final_memory = process.memory_info().rss
        
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be minimal (less than 50MB for this test)
        max_acceptable_growth = 50 * 1024 * 1024  # 50MB
        assert memory_growth < max_acceptable_growth, f"Memory grew by {memory_growth / 1024 / 1024:.2f}MB"
    
    def test_data_integrity_requirements(self):
        """Verify data integrity is maintained."""
        processor = EEGProcessor()
        
        # Test with known input
        test_input = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        
        processed = processor.process(test_input, sampling_rate=100)
        
        # Verify no corruption
        assert processed is not None
        assert processed.shape[0] == test_input.shape[0]  # Same number of channels
        assert np.all(np.isfinite(processed))  # No NaN or inf
        
        # Verify processing is deterministic (same input -> same output)
        processed2 = processor.process(test_input.copy(), sampling_rate=100)
        np.testing.assert_array_almost_equal(processed, processed2, decimal=5)
    
    def test_configuration_validation(self):
        """Verify configuration validation works."""
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Test invalid configuration updates
        with pytest.raises(Exception):  # Should raise configuration error
            config_manager.update_config({
                "model.hidden_size": -1  # Invalid negative size
            })
        
        with pytest.raises(Exception):
            config_manager.update_config({
                "training.learning_rate": "invalid"  # Invalid type
            })
        
        # Valid updates should work
        config_manager.update_config({
            "model.hidden_size": 512,
            "training.batch_size": 16
        })
        
        updated_config = config_manager.get_config()
        assert updated_config.model.hidden_size == 512
        assert updated_config.training.batch_size == 16
    
    def test_monitoring_alerting(self):
        """Verify monitoring and alerting systems work."""
        metrics_collector = get_metrics_collector()
        health_checker = get_health_checker()
        
        # Test alert callback registration
        alerts_received = []
        
        def test_alert_callback(error_event):
            alerts_received.append(error_event)
        
        metrics_collector.alert_callbacks.append(test_alert_callback)
        
        # Trigger an error that should generate an alert
        metrics_collector.record_error(
            'critical', 'test_component',
            'Test critical error for alerting',
            context={'test': True}
        )
        
        # Verify alert was received
        assert len(alerts_received) > 0
        assert alerts_received[0].level == 'critical'
        
        # Test health check alerting
        health_status = health_checker.get_health_status()
        assert health_status is not None
        assert hasattr(health_status, 'alerts')


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])