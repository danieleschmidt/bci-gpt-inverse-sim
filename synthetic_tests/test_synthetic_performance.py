#!/usr/bin/env python3
"""Synthetic performance tests."""

import unittest
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestSyntheticPerformance(unittest.TestCase):
    """Synthetic performance tests."""
    
    def test_list_performance(self):
        """Test list operation performance."""
        start_time = time.time()
        test_list = [i for i in range(10000)]
        execution_time = time.time() - start_time
        
        self.assertEqual(len(test_list), 10000)
        self.assertLess(execution_time, 1.0)  # Should complete in under 1 second
    
    def test_dict_performance(self):
        """Test dictionary operation performance."""
        start_time = time.time()
        test_dict = {f"key_{i}": i for i in range(10000)}
        execution_time = time.time() - start_time
        
        self.assertEqual(len(test_dict), 10000)
        self.assertLess(execution_time, 1.0)
    
    def test_file_io_performance(self):
        """Test file I/O performance."""
        test_file = Path("performance_test.txt")
        test_data = "test data\n" * 1000
        
        start_time = time.time()
        test_file.write_text(test_data)
        content = test_file.read_text()
        test_file.unlink(missing_ok=True)
        execution_time = time.time() - start_time
        
        self.assertEqual(len(content), len(test_data))
        self.assertLess(execution_time, 0.5)
    
    def test_computation_performance(self):
        """Test computation performance."""
        start_time = time.time()
        result = sum(i * i for i in range(1000))
        execution_time = time.time() - start_time
        
        expected = sum(i * i for i in range(1000))
        self.assertEqual(result, expected)
        self.assertLess(execution_time, 0.1)

if __name__ == "__main__":
    unittest.main()
