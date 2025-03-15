"""
Specialized debugging utilities for neuromorphic systems.
"""
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import base64

class SpikeDebugger:
    """Simple debugger for spike-based neuromorphic systems."""
    
    def __init__(self, name: str = "SNN Debugger"):
        """
        Initialize spike debugger.
        
        Args:
            name: Name of the debugger instance
        """
        self.name = name
        self.logger = logging.getLogger(f"SpikeDebugger-{name}")
        self.spike_history = []
        self.neuron_states = {}
        self.breakpoints = {}
        self.max_history_length = 1000
        self.recording = False
    
    def record_spikes(self, spikes: np.ndarray, layer_name: str, timestamp: Optional[float] = None):
        """
        Record spike activity.
        
        Args:
            spikes: Binary spike array
            layer_name: Name of the layer/population
            timestamp: Timestamp (defaults to current time)
        """
        if not self.recording:
            return
            
        if timestamp is None:
            timestamp = time.time()
        
        # Store spike data
        self.spike_history.append({
            'spikes': spikes.copy(),
            'layer': layer_name,
            'timestamp': timestamp,
            'active_count': np.sum(spikes)
        })
        
        # Limit history length
        if len(self.spike_history) > self.max_history_length:
            self.spike_history.pop(0)
    
    def record_neuron_state(self, neuron_id: int, state: Dict[str, Any], 
                           layer_name: str, timestamp: Optional[float] = None):
        """
        Record neuron state.
        
        Args:
            neuron_id: ID of the neuron
            state: Neuron state variables
            layer_name: Name of the layer/population
            timestamp: Timestamp (defaults to current time)
        """
        if not self.recording:
            return
            
        if timestamp is None:
            timestamp = time.time()
        
        key = f"{layer_name}_{neuron_id}"
        if key not in self.neuron_states:
            self.neuron_states[key] = []
        
        # Store neuron state
        self.neuron_states[key].append({
            'state': state.copy(),
            'timestamp': timestamp
        })
        
        # Limit history length
        if len(self.neuron_states[key]) > self.max_history_length:
            self.neuron_states[key].pop(0)
    
    def set_breakpoint(self, condition: str, layer_name: str, callback=None):
        """
        Set a breakpoint for debugging.
        
        Args:
            condition: Condition expression (e.g., "active_count > 10")
            layer_name: Name of the layer/population
            callback: Function to call when breakpoint is hit
        """
        self.breakpoints[f"{layer_name}_{condition}"] = {
            'condition': condition,
            'layer': layer_name,
            'callback': callback,
            'hit_count': 0
        }
        self.logger.info(f"Breakpoint set: {layer_name} when {condition}")
    
    def check_breakpoints(self, spikes: np.ndarray, layer_name: str):
        """
        Check if any breakpoints are triggered.
        
        Args:
            spikes: Binary spike array
            layer_name: Name of the layer/population
            
        Returns:
            True if a breakpoint was hit
        """
        active_count = np.sum(spikes)
        
        for bp_id, bp in self.breakpoints.items():
            if bp['layer'] != layer_name:
                continue
                
            # Simple condition evaluation
            condition = bp['condition']
            triggered = False
            
            if "active_count" in condition:
                # Replace with actual value
                eval_condition = condition.replace("active_count", str(active_count))
                try:
                    triggered = eval(eval_condition)
                except Exception as e:
                    self.logger.error(f"Error evaluating breakpoint condition: {e}")
            
            if triggered:
                bp['hit_count'] += 1
                self.logger.info(f"Breakpoint hit: {layer_name} {condition} (count: {bp['hit_count']})")
                
                if bp['callback']:
                    try:
                        bp['callback'](spikes, layer_name)
                    except Exception as e:
                        self.logger.error(f"Error in breakpoint callback: {e}")
                
                return True
        
        return False
    
    def start_recording(self):
        """Start recording spike and neuron data."""
        self.recording = True
        self.logger.info("Started recording")
    
    def stop_recording(self):
        """Stop recording spike and neuron data."""
        self.recording = False
        self.logger.info("Stopped recording")
    
    def clear_history(self):
        """Clear recorded history."""
        self.spike_history = []
        self.neuron_states = {}
        self.logger.info("History cleared")
    
    def get_spike_statistics(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about recorded spikes.
        
        Args:
            layer_name: Optional filter by layer name
            
        Returns:
            Dictionary of spike statistics
        """
        if not self.spike_history:
            return {"count": 0}
        
        # Filter by layer if specified
        history = self.spike_history
        if layer_name:
            history = [entry for entry in history if entry['layer'] == layer_name]
        
        if not history:
            return {"count": 0}
        
        # Calculate statistics
        active_counts = [entry['active_count'] for entry in history]
        return {
            "count": len(history),
            "total_spikes": sum(active_counts),
            "avg_active": np.mean(active_counts),
            "max_active": np.max(active_counts),
            "min_active": np.min(active_counts),
            "std_active": np.std(active_counts)
        }
    
    def plot_spike_raster(self, layer_name: Optional[str] = None, 
                         max_neurons: int = 100, max_timesteps: int = 100) -> str:
        """
        Generate a spike raster plot.
        
        Args:
            layer_name: Optional filter by layer name
            max_neurons: Maximum number of neurons to plot
            max_timesteps: Maximum number of timesteps to plot
            
        Returns:
            Base64 encoded PNG image
        """
        if not self.spike_history:
            return ""
        
        # Filter by layer if specified
        history = self.spike_history
        if layer_name:
            history = [entry for entry in history if entry['layer'] == layer_name]
        
        if not history:
            return ""
        
        # Limit to recent history
        history = history[-max_timesteps:]
        
        # Get spike data
        first_spikes = history[0]['spikes']
        neuron_count = min(len(first_spikes), max_neurons)
        
        # Create raster plot
        plt.figure(figsize=(10, 6))
        
        for t, entry in enumerate(history):
            spikes = entry['spikes'][:neuron_count]
            spike_indices = np.where(spikes > 0)[0]
            if len(spike_indices) > 0:
                plt.scatter([t] * len(spike_indices), spike_indices, s=2, c='black')
        
        plt.xlabel('Timestep')
        plt.ylabel('Neuron Index')
        plt.title(f'Spike Raster Plot - {layer_name if layer_name else "All Layers"}')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str


class WeightDebugger:
    """Simple debugger for neural network weights."""
    
    def __init__(self, name: str = "Weight Debugger"):
        """
        Initialize weight debugger.
        
        Args:
            name: Name of the debugger instance
        """
        self.name = name
        self.logger = logging.getLogger(f"WeightDebugger-{name}")
        self.weight_snapshots = {}
    
    def snapshot_weights(self, weights: np.ndarray, connection_name: str):
        """
        Take a snapshot of weights.
        
        Args:
            weights: Weight matrix
            connection_name: Name of the connection
        """
        timestamp = time.time()
        
        if connection_name not in self.weight_snapshots:
            self.weight_snapshots[connection_name] = []
        
        # Store weight snapshot
        self.weight_snapshots[connection_name].append({
            'weights': weights.copy(),
            'timestamp': timestamp,
            'stats': {
                'min': np.min(weights),
                'max': np.max(weights),
                'mean': np.mean(weights),
                'std': np.std(weights)
            }
        })
        
        # Limit history (keep only last 10 snapshots)
        if len(self.weight_snapshots[connection_name]) > 10:
            self.weight_snapshots[connection_name].pop(0)
    
    def compare_snapshots(self, connection_name: str) -> Dict[str, Any]:
        """
        Compare weight snapshots to detect changes.
        
        Args:
            connection_name: Name of the connection
            
        Returns:
            Dictionary with comparison results
        """
        if connection_name not in self.weight_snapshots:
            return {"error": "No snapshots available"}
        
        snapshots = self.weight_snapshots[connection_name]
        if len(snapshots) < 2:
            return {"error": "Need at least 2 snapshots to compare"}
        
        # Compare latest two snapshots
        latest = snapshots[-1]
        previous = snapshots[-2]
        
        # Calculate differences
        weight_diff = latest['weights'] - previous['weights']
        
        return {
            'connection': connection_name,
            'time_diff': latest['timestamp'] - previous['timestamp'],
            'diff_stats': {
                'min': np.min(weight_diff),
                'max': np.max(weight_diff),
                'mean': np.mean(weight_diff),
                'std': np.std(weight_diff),
                'abs_mean': np.mean(np.abs(weight_diff)),
                'changed_percent': np.mean(weight_diff != 0) * 100
            }
        }


# Simple usage example
def debug_example():
    """Example of using the debugging utilities."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create spike debugger
    spike_debugger = SpikeDebugger("TestNetwork")
    spike_debugger.start_recording()
    
    # Set a breakpoint
    def breakpoint_callback(spikes, layer):
        print(f"Breakpoint triggered in {layer} with {np.sum(spikes)} active neurons")
    
    spike_debugger.set_breakpoint("active_count > 5", "hidden_layer", breakpoint_callback)
    
    # Simulate some spikes
    for t in range(10):
        # Generate random spikes
        input_spikes = np.random.binomial(1, 0.2, 20)
        hidden_spikes = np.random.binomial(1, 0.1, 30)
        
        # Record spikes
        spike_debugger.record_spikes(input_spikes, "input_layer")
        spike_debugger.record_spikes(hidden_spikes, "hidden_layer")
        
        # Check breakpoints
        spike_debugger.check_breakpoints(hidden_spikes, "hidden_layer")
        
        # Record neuron state
        for i in range(3):  # Just record a few neurons
            state = {
                'potential': np.random.random(),
                'threshold': 0.5,
                'refractory': 0
            }
            spike_debugger.record_neuron_state(i, state, "hidden_layer")
    
    # Get statistics
    stats = spike_debugger.get_spike_statistics("hidden_layer")
    print(f"Spike statistics: {stats}")
    
    # Create weight debugger
    weight_debugger = WeightDebugger("TestWeights")
    
    # Take weight snapshots
    weights1 = np.random.normal(0, 0.1, (30, 20))
    weight_debugger.snapshot_weights(weights1, "input_to_hidden")
    
    # Simulate weight changes
    weights2 = weights1 + np.random.normal(0, 0.01, (30, 20))
    weight_debugger.snapshot_weights(weights2, "input_to_hidden")
    
    # Compare snapshots
    comparison = weight_debugger.compare_snapshots("input_to_hidden")
    print(f"Weight comparison: {comparison}")


if __name__ == "__main__":
    debug_example()