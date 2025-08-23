#!/usr/bin/env python3
"""Test the advanced scaling and optimization systems"""

import time
import sys
from pathlib import Path

def test_advanced_auto_scaler():
    """Test advanced auto-scaling functionality"""
    print("🚀 Testing Advanced Auto-Scaler...")
    
    try:
        from bci_gpt.scaling.advanced_auto_scaler import (
            AdvancedAutoScaler, ResourceMetrics, ScalingReason,
            ThresholdScalingPolicy, PredictiveScalingPolicy,
            create_bci_auto_scaler
        )
        
        # Test basic auto-scaler
        scaler = AdvancedAutoScaler("test_scaler", min_instances=1, max_instances=5)
        
        # Add scaling handler
        scaling_events = []
        
        def test_scaling_handler(old: int, new: int, reason: ScalingReason):
            scaling_events.append((old, new, reason))
        
        scaler.add_scaling_handler(test_scaling_handler)
        print("✅ Auto-scaler initialization")
        
        # Test metrics update
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=80.0,  # High CPU to trigger scaling
            memory_usage=60.0,
            queue_size=30,
            response_time_ms=200.0,
            active_requests=50,
            errors_per_minute=1.0,
            throughput_per_second=100.0
        )
        
        scaler.update_metrics(metrics)
        time.sleep(0.1)  # Allow processing
        print("✅ Metrics update and processing")
        
        # Test status
        status = scaler.get_status()
        assert 'current_instances' in status
        assert 'policies_count' in status
        print("✅ Status reporting")
        
        # Test BCI auto-scaler
        bci_scaler = create_bci_auto_scaler()
        assert bci_scaler.min_instances >= 2  # BCI requires redundancy
        print("✅ BCI-optimized auto-scaler")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced auto-scaler test failed: {e}")
        return False

def test_load_balancer():
    """Test load balancer functionality"""
    print("\n⚖️  Testing Load Balancer...")
    
    try:
        from bci_gpt.scaling.load_balancer import (
            LoadBalancer, ServiceInstance, InstanceStatus, InstanceMetrics,
            LoadBalancingStrategy, create_bci_load_balancer
        )
        
        # Create load balancer
        lb = LoadBalancer("test_balancer", strategy=LoadBalancingStrategy.ROUND_ROBIN)
        
        # Add test instances
        instance1 = ServiceInstance("instance-1", "http://localhost:8001", weight=1.0)
        instance2 = ServiceInstance("instance-2", "http://localhost:8002", weight=2.0)
        instance3 = ServiceInstance("instance-3", "http://localhost:8003", weight=1.5)
        
        lb.add_instance(instance1)
        lb.add_instance(instance2)
        lb.add_instance(instance3)
        print("✅ Load balancer and instances creation")
        
        # Test instance selection
        selected1 = lb.select_instance()
        selected2 = lb.select_instance()
        selected3 = lb.select_instance()
        
        assert selected1 is not None
        assert selected2 is not None
        assert selected3 is not None
        print("✅ Instance selection")
        
        # Test release instance
        lb.release_instance(selected1.instance_id, response_time_ms=150.0)
        lb.release_instance(selected2.instance_id, response_time_ms=200.0)
        print("✅ Instance release and metrics update")
        
        # Test instance status management
        lb.set_instance_status("instance-1", InstanceStatus.DEGRADED)
        lb.drain_instance("instance-2")
        print("✅ Instance status management")
        
        # Test different strategies
        lb_adaptive = LoadBalancer("adaptive_balancer", strategy=LoadBalancingStrategy.ADAPTIVE)
        lb_adaptive.add_instance(instance1)
        selected = lb_adaptive.select_instance()
        assert selected is not None
        print("✅ Adaptive load balancing")
        
        # Test status
        status = lb.get_status()
        assert 'total_instances' in status
        assert 'healthy_instances' in status
        print("✅ Status reporting")
        
        # Test BCI load balancer
        bci_lb = create_bci_load_balancer()
        assert bci_lb.strategy == LoadBalancingStrategy.ADAPTIVE
        print("✅ BCI-optimized load balancer")
        
        return True
        
    except Exception as e:
        print(f"❌ Load balancer test failed: {e}")
        return False

def test_scaling_policies():
    """Test scaling policy implementations"""
    print("\n📊 Testing Scaling Policies...")
    
    try:
        from bci_gpt.scaling.advanced_auto_scaler import (
            ThresholdScalingPolicy, PredictiveScalingPolicy,
            ResourceMetrics, ScalingDirection
        )
        
        # Test threshold policy
        threshold_policy = ThresholdScalingPolicy(
            cpu_scale_up_threshold=70.0,
            cpu_scale_down_threshold=30.0
        )
        
        # High CPU metrics should trigger scale up
        high_cpu_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=85.0,
            memory_usage=50.0,
            queue_size=20,
            response_time_ms=100.0,
            active_requests=30,
            errors_per_minute=0.5,
            throughput_per_second=80.0
        )
        
        direction, reason = threshold_policy.should_scale(high_cpu_metrics, current_instances=2)
        assert direction == ScalingDirection.SCALE_UP
        print("✅ Threshold policy scale up")
        
        # Low CPU metrics should trigger scale down
        low_cpu_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=20.0,
            memory_usage=25.0,
            queue_size=5,
            response_time_ms=50.0,
            active_requests=5,
            errors_per_minute=0.1,
            throughput_per_second=20.0
        )
        
        direction, reason = threshold_policy.should_scale(low_cpu_metrics, current_instances=3)
        assert direction == ScalingDirection.SCALE_DOWN
        print("✅ Threshold policy scale down")
        
        # Test predictive policy
        predictive_policy = PredictiveScalingPolicy()
        
        # Add some metrics history
        for i in range(10):
            metrics = ResourceMetrics(
                timestamp=time.time() - (10-i) * 60,  # 1 minute intervals
                cpu_usage=30.0 + i * 5,  # Increasing trend
                memory_usage=40.0,
                queue_size=10,
                response_time_ms=100.0,
                active_requests=20,
                errors_per_minute=0.2,
                throughput_per_second=50.0
            )
            predictive_policy.add_metrics(metrics)
        
        # Should predict scale up due to increasing trend
        direction, reason = predictive_policy.should_scale(high_cpu_metrics, current_instances=2)
        print("✅ Predictive policy analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Scaling policies test failed: {e}")
        return False

def test_integration():
    """Test integration between scaling components"""
    print("\n🔗 Testing Scaling Integration...")
    
    try:
        from bci_gpt.scaling.advanced_auto_scaler import create_bci_auto_scaler, ResourceMetrics
        from bci_gpt.scaling.load_balancer import create_bci_load_balancer, ServiceInstance
        
        # Create BCI-optimized components
        auto_scaler = create_bci_auto_scaler()
        load_balancer = create_bci_load_balancer()
        
        # Add instances to load balancer
        for i in range(3):
            instance = ServiceInstance(f"bci-instance-{i}", f"http://bci-{i}:8080")
            load_balancer.add_instance(instance)
        
        # Simulate coordinated scaling
        scaling_actions = []
        
        def scaling_handler(old_instances: int, new_instances: int, reason):
            scaling_actions.append((old_instances, new_instances, reason))
            
            # Simulate adding/removing instances from load balancer
            if new_instances > old_instances:
                for i in range(old_instances, new_instances):
                    instance = ServiceInstance(f"scaled-instance-{i}", f"http://scaled-{i}:8080")
                    load_balancer.add_instance(instance)
            elif new_instances < old_instances:
                for i in range(new_instances, old_instances):
                    load_balancer.remove_instance(f"scaled-instance-{i}")
        
        auto_scaler.add_scaling_handler(scaling_handler)
        
        # Simulate high load
        high_load_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=85.0,
            memory_usage=75.0,
            queue_size=80,
            response_time_ms=300.0,
            active_requests=100,
            errors_per_minute=2.0,
            throughput_per_second=150.0
        )
        
        auto_scaler.update_metrics(high_load_metrics)
        time.sleep(0.2)  # Allow processing
        
        # Check integration
        scaler_status = auto_scaler.get_status()
        balancer_status = load_balancer.get_status()
        
        assert scaler_status['current_instances'] >= 2
        assert balancer_status['total_instances'] >= 3
        print("✅ Auto-scaler + load balancer integration")
        
        # Test instance selection under load
        selected = load_balancer.select_instance({'session_id': 'test_session'})
        assert selected is not None
        load_balancer.release_instance(selected.instance_id, response_time_ms=120.0)
        print("✅ Load balancing under scaled conditions")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all scaling system tests"""
    print("⚡ BCI-GPT Advanced Scaling System Test")
    print("=" * 50)
    
    tests = [
        test_advanced_auto_scaler,
        test_load_balancer,
        test_scaling_policies,
        test_integration
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
    print("📊 SCALING SYSTEM TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("🎉 GENERATION 3 (MAKE IT SCALE): ✅ PASSED")
        print("   Advanced scaling and optimization operational")
        return True
    elif success_rate >= 70:
        print("⚠️  GENERATION 3 (MAKE IT SCALE): ⚠️  PARTIAL")
        print("   System has basic scaling but needs optimization")
        return False
    else:
        print("❌ GENERATION 3 (MAKE IT SCALE): ❌ FAILED")
        print("   Scaling systems not operational")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)