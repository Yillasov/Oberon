"""
Tools to validate that hardware behavior matches simulation predictions.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
from ..hardware.processor_interface import NeuromorphicProcessor

def validate_spike_timing(sim_spikes: np.ndarray, hw_spikes: np.ndarray, 
                         tolerance_ms: float = 1.0) -> Dict[str, Any]:
    """
    Validate spike timing between simulation and hardware.
    
    Args:
        sim_spikes: Simulation spike train (neurons x timesteps)
        hw_spikes: Hardware spike train (neurons x timesteps)
        tolerance_ms: Timing tolerance in milliseconds
        
    Returns:
        Dictionary with validation metrics
    """
    if sim_spikes.shape != hw_spikes.shape:
        raise ValueError(f"Shape mismatch: sim {sim_spikes.shape} vs hw {hw_spikes.shape}")
    
    # Count matching spikes within tolerance
    match_count = 0
    total_sim_spikes = 0
    total_hw_spikes = 0
    
    for n in range(sim_spikes.shape[0]):
        sim_spike_times = np.where(sim_spikes[n] > 0)[0]
        hw_spike_times = np.where(hw_spikes[n] > 0)[0]
        
        total_sim_spikes += len(sim_spike_times)
        total_hw_spikes += len(hw_spike_times)
        
        # Count matches within tolerance
        for sim_t in sim_spike_times:
            if np.any(np.abs(hw_spike_times - sim_t) <= tolerance_ms):
                match_count += 1
    
    # Calculate metrics
    precision = match_count / total_hw_spikes if total_hw_spikes > 0 else 0
    recall = match_count / total_sim_spikes if total_sim_spikes > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "match_count": match_count,
        "sim_spike_count": total_sim_spikes,
        "hw_spike_count": total_hw_spikes,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def run_validation_test(processor: NeuromorphicProcessor, 
                       test_inputs: List[np.ndarray],
                       sim_outputs: List[np.ndarray],
                       duration_ms: float = 100.0) -> Dict[str, Any]:
    """
    Run validation tests comparing hardware outputs to simulation predictions.
    
    Args:
        processor: Neuromorphic processor to validate
        test_inputs: List of test input patterns
        sim_outputs: List of expected simulation outputs
        duration_ms: Duration to run each test in milliseconds
        
    Returns:
        Dictionary with validation results
    """
    if not processor.is_connected():
        raise RuntimeError("Processor not connected")
    
    results = []
    
    for i, (test_input, sim_output) in enumerate(zip(test_inputs, sim_outputs)):
        # Run on hardware
        hw_output = processor.run(test_input, duration_ms)
        
        # Validate results
        validation = validate_spike_timing(sim_output, hw_output)
        validation["test_id"] = i
        results.append(validation)
        
        print(f"Test {i}: F1 Score = {validation['f1_score']:.4f}, "
              f"Precision = {validation['precision']:.4f}, "
              f"Recall = {validation['recall']:.4f}")
    
    # Calculate overall metrics
    overall_f1 = np.mean([r["f1_score"] for r in results])
    overall_precision = np.mean([r["precision"] for r in results])
    overall_recall = np.mean([r["recall"] for r in results])
    
    return {
        "test_results": results,
        "overall_f1": overall_f1,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "passed": overall_f1 > 0.8  # Consider test passed if F1 > 0.8
    }

def plot_validation_results(sim_spikes: np.ndarray, hw_spikes: np.ndarray, 
                           neuron_ids: Optional[List[int]] = None,
                           title: str = "Spike Timing Validation"):
    """
    Plot simulation vs hardware spike trains for visual comparison.
    
    Args:
        sim_spikes: Simulation spike train (neurons x timesteps)
        hw_spikes: Hardware spike train (neurons x timesteps)
        neuron_ids: List of neuron IDs to plot (default: first 10)
        title: Plot title
    """
    if neuron_ids is None:
        neuron_ids = list(range(min(10, sim_spikes.shape[0])))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, nid in enumerate(neuron_ids):
        sim_times = np.where(sim_spikes[nid] > 0)[0]
        hw_times = np.where(hw_spikes[nid] > 0)[0]
        
        if len(sim_times) > 0:
            ax.scatter(sim_times, [i+0.2]*len(sim_times), color='blue', marker='|', s=100, label='Simulation' if i==0 else "")
        if len(hw_times) > 0:
            ax.scatter(hw_times, [i-0.2]*len(hw_times), color='red', marker='|', s=100, label='Hardware' if i==0 else "")
    
    ax.set_yticks(range(len(neuron_ids)))
    ax.set_yticklabels([f"Neuron {nid}" for nid in neuron_ids])
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig