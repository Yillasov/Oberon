"""
Performance monitoring for neuromorphic hardware.

This module provides tools to monitor and visualize the performance of
neuromorphic systems in real-time.
"""
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Callable, Deque
from collections import deque
from enum import Enum
import threading
import json


class MonitorMetric(Enum):
    """Metrics that can be monitored in real-time."""
    SPIKE_RATE = 0
    POWER_USAGE = 1
    TEMPERATURE = 2
    MEMORY_USAGE = 3
    LATENCY = 4
    THROUGHPUT = 5


class MetricBuffer:
    """Buffer for storing time-series metric data."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the metric buffer.
        
        Args:
            max_size: Maximum number of data points to store
        """
        self.max_size = max_size
        self.timestamps = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
    
    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        """
        Add a data point to the buffer.
        
        Args:
            value: Metric value
            timestamp: Timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.timestamps.append(timestamp)
        self.values.append(value)
    
    def get_data(self, window: Optional[float] = None) -> Tuple[List[float], List[float]]:
        """
        Get data from the buffer.
        
        Args:
            window: Time window in seconds (None for all data)
            
        Returns:
            Tuple of (timestamps, values)
        """
        if not self.timestamps:
            return [], []
        
        if window is None:
            return list(self.timestamps), list(self.values)
        
        current_time = time.time()
        cutoff_time = current_time - window
        
        # Find index of first timestamp within window
        start_idx = 0
        for i, ts in enumerate(self.timestamps):
            if ts >= cutoff_time:
                start_idx = i
                break
        
        return list(self.timestamps)[start_idx:], list(self.values)[start_idx:]
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics for the buffered data.
        
        Returns:
            Dictionary with statistics
        """
        if not self.values:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "count": 0
            }
        
        values = np.array(self.values)
        return {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "count": len(values)
        }


class PerformanceMonitor:
    """Real-time performance monitor for neuromorphic systems."""
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize the performance monitor.
        
        Args:
            sampling_interval: Time between samples in seconds
        """
        self.sampling_interval = sampling_interval
        self.metrics = {}
        self.data_sources = {}
        self.running = False
        self.monitor_thread = None
    
    def register_metric(self, metric: MonitorMetric, 
                       data_source: Callable[[], float],
                       buffer_size: int = 1000) -> None:
        """
        Register a metric to monitor.
        
        Args:
            metric: Type of metric
            data_source: Function that returns the current metric value
            buffer_size: Size of the buffer for this metric
        """
        self.metrics[metric] = MetricBuffer(max_size=buffer_size)
        self.data_sources[metric] = data_source
    
    def start(self) -> None:
        """Start the monitoring thread."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self) -> None:
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            start_time = time.time()
            
            # Sample all metrics
            for metric, data_source in self.data_sources.items():
                try:
                    value = data_source()
                    self.metrics[metric].add(value, start_time)
                except Exception as e:
                    print(f"Error sampling metric {metric}: {e}")
            
            # Sleep until next sample
            elapsed = time.time() - start_time
            sleep_time = max(0, self.sampling_interval - elapsed)
            time.sleep(sleep_time)
    
    def get_metric_data(self, metric: MonitorMetric, 
                       window: Optional[float] = None) -> Tuple[List[float], List[float]]:
        """
        Get data for a specific metric.
        
        Args:
            metric: Type of metric
            window: Time window in seconds (None for all data)
            
        Returns:
            Tuple of (timestamps, values)
        """
        if metric not in self.metrics:
            return [], []
        
        return self.metrics[metric].get_data(window)
    
    def get_metric_statistics(self, metric: MonitorMetric) -> Dict[str, float]:
        """
        Get statistics for a specific metric.
        
        Args:
            metric: Type of metric
            
        Returns:
            Dictionary with statistics
        """
        if metric not in self.metrics:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "count": 0
            }
        
        return self.metrics[metric].get_statistics()
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all metrics.
        
        Returns:
            Dictionary with statistics for all metrics
        """
        result = {}
        for metric in self.metrics:
            result[metric.name] = self.get_metric_statistics(metric)
        
        return result
    
    def export_data(self, file_path: str) -> bool:
        """
        Export monitoring data to a JSON file.
        
        Args:
            file_path: Path to save the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {}
            for metric in self.metrics:
                timestamps, values = self.get_metric_data(metric)
                data[metric.name] = {
                    "timestamps": timestamps,
                    "values": values,
                    "statistics": self.get_metric_statistics(metric)
                }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False


class AlertCondition:
    """Condition for triggering alerts based on metric values."""
    
    def __init__(self, metric: MonitorMetric, 
                threshold: float, 
                comparison: str = "above",
                window: float = 10.0):
        """
        Initialize the alert condition.
        
        Args:
            metric: Metric to monitor
            threshold: Threshold value
            comparison: Type of comparison ("above", "below", "equal")
            window: Time window to consider in seconds
        """
        self.metric = metric
        self.threshold = threshold
        self.comparison = comparison
        self.window = window
    
    def check(self, monitor: PerformanceMonitor) -> bool:
        """
        Check if the condition is met.
        
        Args:
            monitor: Performance monitor instance
            
        Returns:
            True if the condition is met, False otherwise
        """
        stats = monitor.get_metric_statistics(self.metric)
        value = stats["mean"]
        
        if self.comparison == "above":
            return value > self.threshold
        elif self.comparison == "below":
            return value < self.threshold
        elif self.comparison == "equal":
            return abs(value - self.threshold) < 0.001
        
        return False


class AlertManager:
    """Manager for performance alerts."""
    
    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize the alert manager.
        
        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor
        self.conditions = []
        self.callbacks = []
        self.running = False
        self.alert_thread = None
        self.check_interval = 5.0  # seconds
    
    def add_condition(self, condition: AlertCondition) -> None:
        """
        Add an alert condition.
        
        Args:
            condition: Alert condition
        """
        self.conditions.append(condition)
    
    def add_callback(self, callback: Callable[[AlertCondition], None]) -> None:
        """
        Add an alert callback.
        
        Args:
            callback: Function to call when an alert is triggered
        """
        self.callbacks.append(callback)
    
    def start(self) -> None:
        """Start the alert manager."""
        if self.running:
            return
        
        self.running = True
        self.alert_thread = threading.Thread(target=self._alert_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
    
    def stop(self) -> None:
        """Stop the alert manager."""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join(timeout=2.0)
            self.alert_thread = None
    
    def _alert_loop(self) -> None:
        """Main alert checking loop."""
        while self.running:
            for condition in self.conditions:
                if condition.check(self.monitor):
                    for callback in self.callbacks:
                        callback(condition)
            
            time.sleep(self.check_interval)