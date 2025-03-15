"""
Utilities to monitor neuromorphic hardware in operation.
"""
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from ..config.hardware_config import HardwarePlatform

class HardwareMonitor:
    """Simple monitor for neuromorphic hardware during operation."""
    
    def __init__(self, hardware_address: str, platform: HardwarePlatform):
        """
        Initialize hardware monitor.
        
        Args:
            hardware_address: Address of hardware to monitor
            platform: Hardware platform type
        """
        self.hardware_address = hardware_address
        self.platform = platform
        self.logger = logging.getLogger("HardwareMonitor")
        self.monitoring = False
        self.monitor_thread = None
        self.poll_interval = 1.0  # seconds
        self.metrics = {
            "temperature": 0.0,
            "power_consumption": 0.0,
            "utilization": 0.0,
            "spike_rate": 0.0,
            "error_count": 0
        }
        self.callbacks = []
        
    def start_monitoring(self, poll_interval: float = 1.0):
        """
        Start monitoring hardware.
        
        Args:
            poll_interval: Time between polls in seconds
        """
        if self.monitoring:
            return
            
        self.poll_interval = poll_interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info(f"Started monitoring {self.platform.value} at {self.hardware_address}")
        
    def stop_monitoring(self):
        """Stop monitoring hardware."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info(f"Stopped monitoring {self.hardware_address}")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Poll hardware metrics
                self._poll_metrics()
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(self.metrics)
                    except Exception as e:
                        self.logger.error(f"Error in callback: {e}")
                
                # Check for critical conditions
                self._check_alerts()
                
            except Exception as e:
                self.logger.error(f"Error monitoring hardware: {e}")
                self.metrics["error_count"] += 1
            
            time.sleep(self.poll_interval)
    
    def _poll_metrics(self):
        """Poll hardware for metrics."""
        # In a real implementation, this would use hardware-specific APIs
        # to get actual metrics from the neuromorphic hardware
        
        if self.platform == HardwarePlatform.LOIHI:
            # Simulate Loihi metrics
            self.metrics.update({
                "temperature": 45.0 + 5.0 * (0.5 - (time.time() % 10) / 10),
                "power_consumption": 2.5 + 0.5 * (time.time() % 5) / 5,
                "utilization": 0.3 + 0.2 * (time.time() % 7) / 7,
                "spike_rate": 10000 + 5000 * (time.time() % 3) / 3,
                "cores_active": 48,
                "timestamp": time.time()
            })
        elif self.platform == HardwarePlatform.SPINNAKER:
            # Simulate SpiNNaker metrics
            self.metrics.update({
                "temperature": 50.0 + 3.0 * (0.5 - (time.time() % 8) / 8),
                "power_consumption": 5.0 + 1.0 * (time.time() % 6) / 6,
                "utilization": 0.5 + 0.3 * (time.time() % 5) / 5,
                "spike_rate": 20000 + 8000 * (time.time() % 4) / 4,
                "cores_active": 16,
                "timestamp": time.time()
            })
        else:
            # Generic metrics for other platforms
            self.metrics.update({
                "temperature": 40.0 + 2.0 * (0.5 - (time.time() % 6) / 6),
                "power_consumption": 1.0 + 0.2 * (time.time() % 4) / 4,
                "utilization": 0.4 + 0.1 * (time.time() % 3) / 3,
                "spike_rate": 5000 + 2000 * (time.time() % 5) / 5,
                "timestamp": time.time()
            })
    
    def _check_alerts(self):
        """Check for alert conditions."""
        # Temperature alert
        if self.metrics.get("temperature", 0) > 70.0:
            self.logger.warning(f"HIGH TEMPERATURE: {self.metrics['temperature']}°C")
        
        # Utilization alert
        if self.metrics.get("utilization", 0) > 0.9:
            self.logger.warning(f"HIGH UTILIZATION: {self.metrics['utilization'] * 100}%")
        
        # Error count alert
        if self.metrics.get("error_count", 0) > 5:
            self.logger.error(f"MULTIPLE ERRORS: {self.metrics['error_count']} errors detected")
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register callback for metric updates.
        
        Args:
            callback: Function to call with updated metrics
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current hardware metrics.
        
        Returns:
            Dictionary of current metrics
        """
        return self.metrics.copy()


class HardwareMonitorManager:
    """Manages multiple hardware monitors."""
    
    def __init__(self):
        """Initialize hardware monitor manager."""
        self.monitors = {}
        self.logger = logging.getLogger("HardwareMonitorManager")
    
    def add_monitor(self, hardware_id: str, hardware_address: str, 
                   platform: HardwarePlatform) -> HardwareMonitor:
        """
        Add a hardware monitor.
        
        Args:
            hardware_id: Unique identifier for the hardware
            hardware_address: Address of hardware to monitor
            platform: Hardware platform type
            
        Returns:
            Created hardware monitor
        """
        if hardware_id in self.monitors:
            return self.monitors[hardware_id]
            
        monitor = HardwareMonitor(hardware_address, platform)
        self.monitors[hardware_id] = monitor
        return monitor
    
    def start_all(self, poll_interval: float = 1.0):
        """
        Start all monitors.
        
        Args:
            poll_interval: Time between polls in seconds
        """
        for hardware_id, monitor in self.monitors.items():
            monitor.start_monitoring(poll_interval)
            self.logger.info(f"Started monitoring {hardware_id}")
    
    def stop_all(self):
        """Stop all monitors."""
        for hardware_id, monitor in self.monitors.items():
            monitor.stop_monitoring()
            self.logger.info(f"Stopped monitoring {hardware_id}")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics from all monitored hardware.
        
        Returns:
            Dictionary of hardware IDs to metrics
        """
        return {
            hardware_id: monitor.get_current_metrics()
            for hardware_id, monitor in self.monitors.items()
        }


# Simple usage example
def monitor_example():
    """Example of monitoring hardware."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor manager
    manager = HardwareMonitorManager()
    
    # Add monitors for different hardware
    manager.add_monitor("loihi-1", "loihi.example.com:22", HardwarePlatform.LOIHI)
    manager.add_monitor("spinnaker-1", "spinnaker.example.com:22", HardwarePlatform.SPINNAKER)
    
    # Define a callback for metrics
    def print_metrics(metrics):
        print(f"Temperature: {metrics['temperature']}°C, "
              f"Power: {metrics['power_consumption']}W, "
              f"Utilization: {metrics['utilization']*100:.1f}%")
    
    # Register callback for one monitor
    loihi_monitor = manager.monitors["loihi-1"]
    loihi_monitor.register_callback(print_metrics)
    
    # Start all monitors
    manager.start_all(poll_interval=2.0)
    
    try:
        # Run for a while
        time.sleep(10)
        
        # Get all metrics
        all_metrics = manager.get_all_metrics()
        print("\nAll hardware metrics:")
        for hw_id, metrics in all_metrics.items():
            print(f"{hw_id}: {metrics}")
            
    finally:
        # Stop all monitors
        manager.stop_all()


if __name__ == "__main__":
    monitor_example()