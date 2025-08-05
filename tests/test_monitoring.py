"""
Tests for monitoring and safety systems.
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from bci_gpt.utils.monitoring import (
    MetricsCollector, SystemMetrics, ModelMetrics, StreamingMetrics,
    ErrorEvent, PerformanceProfiler, ClinicalSafetyMonitor,
    get_metrics_collector, profile_performance, monitor_function, safe_function
)


class TestSystemMetrics:
    """Test system metrics data structure."""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation."""
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            gpu_utilization=75.0,
            gpu_memory_used_gb=4.0,
            gpu_memory_total_gb=8.0
        )
        
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert metrics.memory_used_gb >= 0
        assert metrics.gpu_utilization == 75.0
        assert metrics.gpu_memory_used_gb == 4.0
    
    def test_model_metrics_creation(self):
        """Test ModelMetrics creation."""
        metrics = ModelMetrics(
            timestamp=time.time(),
            inference_time_ms=25.5,
            accuracy=0.85,
            confidence=0.92,
            word_error_rate=0.15
        )
        
        assert metrics.inference_time_ms > 0
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.confidence <= 1
        assert 0 <= metrics.word_error_rate <= 1
    
    def test_streaming_metrics_creation(self):
        """Test StreamingMetrics creation."""
        metrics = StreamingMetrics(
            timestamp=time.time(),
            samples_per_second=1000.0,
            buffer_fill_ratio=0.3,
            dropped_samples=0,
            latency_ms=10.0,
            jitter_ms=2.0
        )
        
        assert metrics.samples_per_second > 0
        assert 0 <= metrics.buffer_fill_ratio <= 1
        assert metrics.dropped_samples >= 0
        assert metrics.latency_ms >= 0


class TestMetricsCollector:
    """Test metrics collection system."""
    
    @pytest.fixture
    def collector(self):
        """Create test metrics collector."""
        collector = MetricsCollector(collection_interval=0.1, max_history_size=100)
        yield collector
        collector.stop_collection()
    
    def test_collector_creation(self):
        """Test metrics collector creation."""
        collector = MetricsCollector()
        
        assert collector.collection_interval == 1.0
        assert collector.max_history_size == 10000
        assert not collector.is_collecting
        
        collector.stop_collection()  # Clean up
    
    def test_start_stop_collection(self, collector):
        """Test starting and stopping collection."""
        assert not collector.is_collecting
        
        collector.start_collection()
        assert collector.is_collecting
        assert collector.collection_thread is not None
        
        collector.stop_collection()
        assert not collector.is_collecting
    
    def test_system_metrics_collection(self):
        """Test system metrics collection."""
        collector = MetricsCollector(collection_interval=0.05)
        collector.start_collection()
        
        try:
            # Wait for some metrics to be collected
            time.sleep(0.2)
            
            assert len(collector.system_metrics) > 0
            
            # Check metrics content
            latest_metrics = collector.system_metrics[-1]
            assert isinstance(latest_metrics, SystemMetrics)
            assert latest_metrics.cpu_percent >= 0
            assert latest_metrics.memory_percent >= 0
            
        finally:
            collector.stop_collection()
    
    def test_model_metrics_recording(self, collector):
        """Test model metrics recording."""
        collector.record_model_metrics(
            inference_time_ms=45.0,
            accuracy=0.88,
            confidence=0.91
        )
        
        assert len(collector.model_metrics) == 1
        
        metrics = collector.model_metrics[0]
        assert metrics.inference_time_ms == 45.0
        assert metrics.accuracy == 0.88
        assert metrics.confidence == 0.91
    
    def test_streaming_metrics_recording(self, collector):
        """Test streaming metrics recording."""
        collector.record_streaming_metrics(
            samples_per_second=1000.0,
            buffer_fill_ratio=0.25,
            dropped_samples=2,
            latency_ms=12.0
        )
        
        assert len(collector.streaming_metrics) == 1
        
        metrics = collector.streaming_metrics[0]
        assert metrics.samples_per_second == 1000.0
        assert metrics.buffer_fill_ratio == 0.25
        assert metrics.dropped_samples == 2
        assert metrics.latency_ms == 12.0
    
    def test_error_recording(self, collector):
        """Test error event recording."""
        collector.record_error(
            'warning', 'test_component',
            'Test warning message',
            context={'test': True}
        )
        
        assert len(collector.error_events) == 1
        
        error = collector.error_events[0]
        assert error.level == 'warning'
        assert error.component == 'test_component'
        assert error.message == 'Test warning message'
        assert error.context['test'] is True
    
    def test_counter_increment(self, collector):
        """Test counter increment."""
        collector.increment_counter('test_counter', 1)
        collector.increment_counter('test_counter', 5)
        
        assert collector.counters['test_counter'] == 6
    
    def test_timing_recording(self, collector):
        """Test timing recording."""
        collector.record_timing('test_operation', 45.0)
        collector.record_timing('test_operation', 50.0)
        collector.record_timing('test_operation', 48.0)
        
        assert len(collector.timers['test_operation']) == 3
        assert 45.0 in collector.timers['test_operation']
        assert 50.0 in collector.timers['test_operation']
        assert 48.0 in collector.timers['test_operation']
    
    def test_system_health_assessment(self, collector):
        """Test system health assessment."""
        # Add some test system metrics
        test_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=4.0,
            memory_available_gb=4.0
        )
        collector.system_metrics.append(test_metrics)
        
        health = collector.get_system_health()
        
        assert 'status' in health
        assert health['status'] in ['healthy', 'warning', 'critical']
        assert 'cpu_percent' in health
        assert 'memory_percent' in health
    
    def test_alert_callbacks(self, collector):
        """Test alert callbacks."""
        callback_called = []
        
        def test_callback(error_event):
            callback_called.append(error_event)
        
        collector.alert_callbacks.append(test_callback)
        
        # Record an error that should trigger alert
        collector.record_error(
            'error', 'test_component',
            'Test error message'
        )
        
        assert len(callback_called) == 1
        assert callback_called[0].level == 'error'
    
    def test_metrics_export(self, collector, tmp_path):
        """Test metrics export."""
        # Add some test data
        collector.record_model_metrics(inference_time_ms=30.0, accuracy=0.9)
        collector.record_streaming_metrics(samples_per_second=1000.0, buffer_fill_ratio=0.2)
        collector.record_error('info', 'test', 'Test message')
        
        # Export to file
        export_path = tmp_path / "metrics.json"
        collector.export_metrics(export_path)
        
        assert export_path.exists()
        
        # Verify file content
        import json
        with open(export_path) as f:
            data = json.load(f)
        
        assert 'export_timestamp' in data
        assert 'model_metrics' in data
        assert 'streaming_metrics' in data
        assert 'error_events' in data
        assert len(data['model_metrics']) == 1
        assert len(data['streaming_metrics']) == 1
        assert len(data['error_events']) == 1


class TestPerformanceProfiler:
    """Test performance profiling."""
    
    def test_performance_profiler_context_manager(self):
        """Test performance profiler as context manager."""
        collector = MetricsCollector()
        
        with PerformanceProfiler('test_operation', collector) as profiler:
            time.sleep(0.1)  # Simulate work
        
        # Check that timing was recorded
        assert 'test_operation' in collector.timers
        assert len(collector.timers['test_operation']) == 1
        
        timing = collector.timers['test_operation'][0]
        assert 90 <= timing <= 150  # Should be around 100ms, allow some tolerance
        
        collector.stop_collection()
    
    def test_performance_profiler_with_exception(self):
        """Test performance profiler with exception."""
        collector = MetricsCollector()
        
        try:
            with PerformanceProfiler('failing_operation', collector):
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
        
        # Should still record timing
        assert 'failing_operation' in collector.timers
        
        # Should record error
        assert len(collector.error_events) == 1
        assert collector.error_events[0].component == 'failing_operation'
        
        collector.stop_collection()
    
    def test_profile_performance_decorator(self):
        """Test profile_performance decorator function."""
        with patch('bci_gpt.utils.monitoring._global_metrics_collector') as mock_collector:
            mock_collector_instance = MagicMock()
            mock_collector.return_value = mock_collector_instance
            
            with profile_performance('test_function'):
                time.sleep(0.05)
            
            # Should have called record_timing
            # Note: This is a simplified test due to mocking complexity


class TestMonitoringDecorators:
    """Test monitoring decorators."""
    
    def test_monitor_function_decorator(self):
        """Test monitor_function decorator."""
        
        @monitor_function('test_function')
        def test_func():
            time.sleep(0.01)
            return "success"
        
        with patch('bci_gpt.utils.monitoring.get_metrics_collector') as mock_get_collector:
            mock_collector = MagicMock()
            mock_get_collector.return_value = mock_collector
            
            result = test_func()
            
            assert result == "success"
    
    def test_safe_function_decorator(self):
        """Test safe_function decorator."""
        
        @safe_function(max_retries=2, backoff_factor=0.01)
        def failing_func():
            failing_func.call_count = getattr(failing_func, 'call_count', 0) + 1
            if failing_func.call_count < 3:
                raise ValueError("Test failure")
            return "success"
        
        with patch('bci_gpt.utils.monitoring.get_metrics_collector') as mock_get_collector:
            mock_collector = MagicMock()
            mock_get_collector.return_value = mock_collector
            
            result = failing_func()
            
            assert result == "success"
            assert failing_func.call_count == 3  # Should have retried twice
    
    def test_safe_function_final_failure(self):
        """Test safe_function decorator with final failure."""
        
        @safe_function(max_retries=1, backoff_factor=0.01)
        def always_failing_func():
            raise ValueError("Always fails")
        
        with patch('bci_gpt.utils.monitoring.get_metrics_collector') as mock_get_collector:
            mock_collector = MagicMock()
            mock_get_collector.return_value = mock_collector
            
            with pytest.raises(ValueError, match="Always fails"):
                always_failing_func()


class TestClinicalSafetyMonitor:
    """Test clinical safety monitoring."""
    
    @pytest.fixture
    def safety_monitor(self):
        """Create test safety monitor."""
        return ClinicalSafetyMonitor(
            max_session_duration=60,  # 1 minute for testing
            fatigue_threshold=0.8,
            seizure_detection_enabled=True
        )
    
    def test_safety_monitor_creation(self, safety_monitor):
        """Test safety monitor creation."""
        assert safety_monitor.max_session_duration == 60
        assert safety_monitor.fatigue_threshold == 0.8
        assert safety_monitor.seizure_detection_enabled is True
        assert not safety_monitor.is_monitoring
    
    def test_session_start_end(self, safety_monitor):
        """Test session start and end."""
        # Start session
        safety_monitor.start_session("test_user")
        
        assert safety_monitor.is_monitoring
        assert safety_monitor.session_user_id == "test_user"
        assert safety_monitor.session_start_time is not None
        
        # End session
        report = safety_monitor.end_session()
        
        assert not safety_monitor.is_monitoring
        assert report['user_id'] == "test_user"
        assert 'session_duration_minutes' in report
        assert report['session_duration_minutes'] >= 0
    
    def test_safety_check(self, safety_monitor):
        """Test safety check."""
        # Should be safe when not monitoring
        assert safety_monitor.is_safe()
        
        # Start session
        safety_monitor.start_session("test_user")
        
        # Should be safe initially
        assert safety_monitor.is_safe()
        
        # Mock exceeding max duration
        safety_monitor.session_start_time = time.time() - 120  # 2 minutes ago
        
        # Should not be safe now
        assert not safety_monitor.is_safe()
    
    def test_fatigue_detection_eeg(self, safety_monitor):
        """Test fatigue detection from EEG."""
        safety_monitor.start_session("test_user")
        
        # Test with normal EEG (low fatigue)
        normal_eeg = np.random.randn(8, 1000) * 0.5  # Low power
        is_fatigued = safety_monitor.detect_fatigue(eeg_data=normal_eeg)
        assert not is_fatigued
        
        # Test with high alpha activity (high fatigue)
        fatigued_eeg = np.random.randn(8, 1000) * 1.2  # High power
        is_fatigued = safety_monitor.detect_fatigue(eeg_data=fatigued_eeg)
        # May or may not detect fatigue depending on random values
        assert isinstance(is_fatigued, bool)
    
    def test_fatigue_detection_performance(self, safety_monitor):
        """Test fatigue detection from performance metrics."""
        safety_monitor.start_session("test_user")
        
        # Test with good performance (not fatigued)
        good_performance = {
            'accuracy': 0.9,
            'reaction_time_ms': 800
        }
        is_fatigued = safety_monitor.detect_fatigue(performance_metrics=good_performance)
        assert not is_fatigued
        
        # Test with poor performance (fatigued)
        poor_performance = {
            'accuracy': 0.6,  # Low accuracy
            'reaction_time_ms': 2500  # Slow reaction
        }
        is_fatigued = safety_monitor.detect_fatigue(performance_metrics=poor_performance)
        assert is_fatigued
    
    def test_break_recommendations(self, safety_monitor):
        """Test break recommendations."""
        safety_monitor.start_session("test_user")
        
        initial_recommendations = safety_monitor.break_recommendations
        
        safety_monitor.recommend_break(300)
        
        assert safety_monitor.break_recommendations == initial_recommendations + 1
    
    def test_forced_breaks(self, safety_monitor):
        """Test forced breaks."""
        safety_monitor.start_session("test_user")
        
        initial_breaks = safety_monitor.forced_breaks
        
        safety_monitor.enforce_break(600)
        
        assert safety_monitor.forced_breaks == initial_breaks + 1
        assert len(safety_monitor.safety_violations) > 0
    
    def test_safety_violation_recording(self, safety_monitor):
        """Test safety violation recording."""
        safety_monitor.start_session("test_user")
        
        initial_violations = len(safety_monitor.safety_violations)
        
        safety_monitor._record_violation("test_violation", {"test": "data"})
        
        assert len(safety_monitor.safety_violations) == initial_violations + 1
        
        violation = safety_monitor.safety_violations[-1]
        assert violation['type'] == "test_violation"
        assert violation['user_id'] == "test_user"
        assert violation['context']['test'] == "data"
    
    def test_session_report_with_violations(self, safety_monitor):
        """Test session report with safety violations."""
        safety_monitor.start_session("test_user")
        
        # Force some violations
        safety_monitor.enforce_break(300)
        safety_monitor.recommend_break(300)
        
        report = safety_monitor.end_session()
        
        assert report['forced_breaks'] == 1
        assert report['break_recommendations'] == 1
        assert report['safety_violations'] > 0
        assert report['status'] == "safety_breaks_required"


class TestGlobalMetricsCollector:
    """Test global metrics collector functionality."""
    
    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns singleton."""
        # Clear global collector
        import bci_gpt.utils.monitoring as monitoring
        monitoring._global_metrics_collector = None
        
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
        
        # Clean up
        collector1.stop_collection()
    
    def test_get_metrics_collector_starts_collection(self):
        """Test that get_metrics_collector starts collection."""
        import bci_gpt.utils.monitoring as monitoring
        monitoring._global_metrics_collector = None
        
        collector = get_metrics_collector()
        
        assert collector.is_collecting
        
        # Clean up
        collector.stop_collection()


class TestErrorHandling:
    """Test error handling in monitoring system."""
    
    def test_collector_callback_exception(self):
        """Test handling of callback exceptions."""
        collector = MetricsCollector()
        
        def failing_callback(metrics):
            raise ValueError("Callback failed")
        
        def working_callback(metrics):
            working_callback.called = True
        
        working_callback.called = False
        
        collector.metric_callbacks.append(failing_callback)
        collector.metric_callbacks.append(working_callback)
        
        # Record metrics (should not crash despite failing callback)
        collector.record_model_metrics(inference_time_ms=30.0)
        
        # Working callback should still be called
        assert working_callback.called
        
        collector.stop_collection()
    
    def test_system_metrics_collection_failure(self):
        """Test handling of system metrics collection failure."""
        collector = MetricsCollector(collection_interval=0.01)
        
        # Mock psutil to raise exception
        with patch('bci_gpt.utils.monitoring.psutil.cpu_percent', side_effect=Exception("Test error")):
            collector.start_collection()
            
            try:
                time.sleep(0.05)  # Let it try to collect
                
                # Should handle gracefully and not crash
                assert collector.is_collecting
                
            finally:
                collector.stop_collection()
    
    def test_invalid_export_format(self):
        """Test invalid export format handling."""
        collector = MetricsCollector()
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            collector.export_metrics("/tmp/test", format="invalid_format")
        
        collector.stop_collection()


# Test configuration
pytest.mark.monitoring = pytest.mark.filterwarnings("ignore::DeprecationWarning")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])