#!/usr/bin/env python3
"""Lightweight BCI-GPT demonstration without heavy dependencies."""

import json
import random
import time
from typing import Dict, Any, List
import numpy as np  # Mock if not available

class MockBCIGPTDemo:
    """Lightweight demonstration of BCI-GPT functionality."""
    
    def __init__(self):
        self.mock_vocabulary = [
            "hello", "world", "yes", "no", "help", "stop", "more", "please",
            "thank", "you", "good", "morning", "afternoon", "evening"
        ]
    
    def simulate_eeg_signal(self, duration: float = 1.0) -> Dict[str, Any]:
        """Simulate EEG signal data."""
        sampling_rate = 1000
        n_samples = int(duration * sampling_rate)
        n_channels = 9
        
        # Generate realistic-looking EEG data
        signal = []
        for ch in range(n_channels):
            # Simulate brain rhythms (alpha, beta, gamma)
            alpha = np.sin(2 * np.pi * 10 * np.linspace(0, duration, n_samples))
            beta = 0.5 * np.sin(2 * np.pi * 20 * np.linspace(0, duration, n_samples))
            noise = 0.1 * np.random.normal(0, 1, n_samples)
            signal.append(alpha + beta + noise)
        
        return {
            "data": signal,
            "sampling_rate": sampling_rate,
            "n_channels": n_channels,
            "duration": duration,
            "quality_score": random.uniform(0.7, 0.95)
        }
    
    def decode_thought(self, eeg_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Mock thought decoding from EEG signal."""
        
        # Simulate processing time
        time.sleep(0.05)  # 50ms latency
        
        # Mock decoding result
        predicted_word = random.choice(self.mock_vocabulary)
        confidence = random.uniform(0.6, 0.95)
        
        # Simulate token probabilities
        token_probs = {word: random.uniform(0.01, 0.3) for word in self.mock_vocabulary}
        token_probs[predicted_word] = confidence
        
        return {
            "predicted_text": predicted_word,
            "confidence": confidence,
            "token_probabilities": token_probs,
            "latency_ms": 50,
            "signal_quality": eeg_signal["quality_score"]
        }
    
    def run_demo(self, n_trials: int = 5) -> List[Dict[str, Any]]:
        """Run complete BCI-GPT demonstration."""
        print("🧠 Starting BCI-GPT Demo...")
        
        results = []
        for i in range(n_trials):
            print(f"Trial {i+1}/{n_trials}")
            
            # Simulate EEG recording
            eeg_data = self.simulate_eeg_signal(duration=2.0)
            print(f"  EEG Quality: {eeg_data['quality_score']:.2%}")
            
            # Decode thought
            decoded = self.decode_thought(eeg_data)
            print(f"  Decoded: '{decoded['predicted_text']}' (confidence: {decoded['confidence']:.2%})")
            
            results.append({
                "trial": i + 1,
                "eeg_quality": eeg_data["quality_score"],
                "predicted_text": decoded["predicted_text"],
                "confidence": decoded["confidence"],
                "latency_ms": decoded["latency_ms"]
            })
            
            time.sleep(0.5)
        
        return results

if __name__ == "__main__":
    demo = MockBCIGPTDemo()
    results = demo.run_demo(5)
    print("\n📊 Demo Results:")
    print(json.dumps(results, indent=2))
