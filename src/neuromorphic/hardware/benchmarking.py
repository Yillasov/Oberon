"""
Benchmarking tools for neuromorphic hardware.

This module provides utilities to measure performance, energy efficiency,
and accuracy of neuromorphic systems.
"""
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Callable, Optional
from enum import Enum


class BenchmarkMetric(Enum):
    """Metrics that can be measured in benchmarks."""
    LATENCY = 0
    THROUGHPUT = 1
    ENERGY = 2
    ACCURACY = 3
    SPIKE_RATE = 4
    MEMORY_USAGE = 5


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        """
        Initialize benchmark results.
        
        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.metrics = {}
        self.start_time = time.time()
        self.end_time = None
    
    def add_metric(self, metric: BenchmarkMetric, value: float, unit: str = "") -> None:
        """
        Add a metric to the results.
        
        Args:
            metric: Type of metric
            value: Measured value
            unit: Unit of measurement
        """
        self.metrics[metric] = {"value": value, "unit": unit}
    
    def end_timing(self) -> float:
        """
        End timing and calculate elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of benchmark results.
        
        Returns:
            Dictionary with benchmark results
        """
        elapsed = self.end_time - self.start_time if self.end_time else time.time() - self.start_time
        
        summary = {
            "name": self.name,
            "elapsed_time": elapsed,
            "metrics": {}
        }
        
        for metric, data in self.metrics.items():
            summary["metrics"][metric.name] = {
                "value": data["value"],
                "unit": data["unit"]
            }
        
        return summary


class PerformanceBenchmark:
    """Measures performance metrics of neuromorphic systems."""
    
    def __init__(self, name: str):
        """
        Initialize the performance benchmark.
        
        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.result = BenchmarkResult(name)
    
    def measure_latency(self, func: Callable, *args, **kwargs) -> float:
        """
        Measure execution latency of a function.
        
        Args:
            func: Function to benchmark
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Execution time in milliseconds
        """
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        latency_ms = (end - start) * 1000
        self.result.add_metric(BenchmarkMetric.LATENCY, latency_ms, "ms")
        
        return latency_ms
    
    def measure_throughput(self, func: Callable, input_size: int, 
                          iterations: int = 100, *args, **kwargs) -> float:
        """
        Measure throughput of a function.
        
        Args:
            func: Function to benchmark
            input_size: Size of input data in elements
            iterations: Number of iterations to run
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Throughput in elements per second
        """
        start = time.time()
        
        for _ in range(iterations):
            func(*args, **kwargs)
        
        end = time.time()
        
        elapsed = end - start
        throughput = (input_size * iterations) / elapsed
        
        self.result.add_metric(BenchmarkMetric.THROUGHPUT, throughput, "elements/s")
        
        return throughput
    
    def measure_spike_rate(self, spike_data: np.ndarray, duration: float) -> float:
        """
        Measure average spike rate.
        
        Args:
            spike_data: Binary spike data
            duration: Duration in seconds
            
        Returns:
            Average spike rate in Hz
        """
        if len(spike_data.shape) == 1:
            # Single neuron over time
            num_spikes = np.sum(spike_data)
            num_neurons = 1
        else:
            # Multiple neurons over time
            num_spikes = np.sum(spike_data)
            num_neurons = spike_data.shape[1] if len(spike_data.shape) > 1 else 1
        
        avg_rate = num_spikes / (num_neurons * duration)
        
        self.result.add_metric(BenchmarkMetric.SPIKE_RATE, avg_rate, "Hz")
        
        return avg_rate
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the benchmark and get results.
        
        Returns:
            Benchmark results summary
        """
        self.result.end_timing()
        return self.result.get_summary()


class EnergyBenchmark:
    """Estimates energy consumption of neuromorphic systems."""
    
    def __init__(self, name: str, power_model: Optional[Callable] = None):
        """
        Initialize the energy benchmark.
        
        Args:
            name: Name of the benchmark
            power_model: Function that estimates power based on activity
        """
        self.name = name
        self.result = BenchmarkResult(name)
        self.power_model = power_model or self._default_power_model
    
    def _default_power_model(self, spike_rate: float, num_neurons: int) -> float:
        """
        Default power model based on spike rate.
        
        Args:
            spike_rate: Average spike rate in Hz
            num_neurons: Number of neurons
            
        Returns:
            Estimated power in mW
        """
        # Simple model: 1 pJ per spike + 0.1 mW static power
        power_per_spike_pJ = 1.0
        static_power_mW = 0.1
        
        dynamic_power = power_per_spike_pJ * spike_rate * num_neurons / 1000  # Convert to mW
        total_power = dynamic_power + static_power_mW
        
        return total_power
    
    def estimate_energy(self, spike_data: np.ndarray, duration: float) -> float:
        """
        Estimate energy consumption.
        
        Args:
            spike_data: Binary spike data
            duration: Duration in seconds
            
        Returns:
            Estimated energy in mJ
        """
        if len(spike_data.shape) == 1:
            num_neurons = 1
            spike_rate = np.sum(spike_data) / duration
        else:
            num_neurons = spike_data.shape[1] if len(spike_data.shape) > 1 else 1
            spike_rate = np.sum(spike_data) / (num_neurons * duration)
        
        power_mW = self.power_model(spike_rate, num_neurons)
        energy_mJ = power_mW * duration
        
        self.result.add_metric(BenchmarkMetric.ENERGY, energy_mJ, "mJ")
        
        return energy_mJ
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the benchmark and get results.
        
        Returns:
            Benchmark results summary
        """
        self.result.end_timing()
        return self.result.get_summary()


class AccuracyBenchmark:
    """Measures accuracy metrics of neuromorphic systems."""
    
    def __init__(self, name: str):
        """
        Initialize the accuracy benchmark.
        
        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.result = BenchmarkResult(name)
    
    def measure_classification_accuracy(self, predictions: np.ndarray, 
                                       targets: np.ndarray) -> float:
        """
        Measure classification accuracy.
        
        Args:
            predictions: Predicted classes
            targets: Target classes
            
        Returns:
            Classification accuracy (0-1)
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
        
        correct = np.sum(predictions == targets)
        accuracy = correct / len(targets)
        
        self.result.add_metric(BenchmarkMetric.ACCURACY, accuracy, "")
        
        return accuracy
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the benchmark and get results.
        
        Returns:
            Benchmark results summary
        """
        self.result.end_timing()
        return self.result.get_summary()


class BenchmarkSuite:
    """Suite of benchmarks for comprehensive evaluation."""
    
    def __init__(self, name: str):
        """
        Initialize the benchmark suite.
        
        Args:
            name: Name of the benchmark suite
        """
        self.name = name
        self.benchmarks = []
        self.results = {}
    
    def add_benchmark(self, benchmark: Any) -> None:
        """
        Add a benchmark to the suite.
        
        Args:
            benchmark: Benchmark instance
        """
        self.benchmarks.append(benchmark)
    
    def run_all(self) -> Dict[str, Any]:
        """
        Run all benchmarks in the suite.
        
        Returns:
            Dictionary with all benchmark results
        """
        for benchmark in self.benchmarks:
            result = benchmark.finalize()
            self.results[benchmark.name] = result
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all benchmark results.
        
        Returns:
            Dictionary with benchmark summary
        """
        return {
            "name": self.name,
            "num_benchmarks": len(self.benchmarks),
            "results": self.results
        }