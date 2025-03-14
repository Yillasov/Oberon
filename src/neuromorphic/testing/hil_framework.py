"""
Basic Hardware-in-the-Loop (HIL) testing framework for neuromorphic systems.
"""
import numpy as np
import time
import threading
from typing import Dict, Any, List, Optional, Callable
import socket
import json


class HILInterface:
    """Simple Hardware-in-the-Loop interface for neuromorphic testing."""
    
    def __init__(self, digital_twin, interface_type="socket", port=5555, hardware_platform=None):
        """
        Initialize HIL interface.
        
        Args:
            digital_twin: Digital twin simulation instance
            interface_type: Type of hardware interface ("socket", "serial", "mock", or "direct")
            port: Communication port for socket/serial interfaces
            hardware_platform: Optional hardware platform driver instance
        """
        self.digital_twin = digital_twin
        self.interface_type = interface_type
        self.port = port
        self.hardware_platform = hardware_platform
        self.connection = None
        self.is_running = False
        self.last_sync_time = 0
        self.sync_interval = 10  # ms
        self.hardware_buffer = []
        self.simulation_buffer = []
        self.real_time_factor = 1.0
        
        self.running = False
        self.hil_thread = None
        self.hardware_connected = False
        
        # Communication buffers
        self.input_buffer = {}
        self.output_buffer = {}
        self.last_hardware_time = 0
        
        # Initialize interface
        if interface_type == "socket":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        elif interface_type == "serial":
            try:
                import serial
                self.serial = serial.Serial(port=f"COM{port}", baudrate=115200, timeout=0.1)
            except ImportError:
                print("PySerial not installed. Falling back to mock interface.")
                self.interface_type = "mock"
        
        # Register callback with digital twin
        if digital_twin:
            digital_twin.register_callback(self._simulation_callback)
    
    def _simulation_callback(self, state: Dict[str, Any]) -> None:
        """Process simulation state updates."""
        # Extract relevant data for hardware
        sensor_data = {
            "accelerometer": state["sensor_outputs"]["accelerometer"].tolist(),
            "gyroscope": state["sensor_outputs"]["gyroscope"].tolist(),
            "pressure": state["sensor_outputs"]["pressure"],
            "temperature": state["sensor_outputs"]["temperature"],
            "time": state["time"]
        }
        
        # Store in output buffer to be sent to hardware
        self.output_buffer = sensor_data
    
    def _hardware_communication_loop(self) -> None:
        """Main hardware communication loop."""
        last_time = time.time()
        
        while self.running:
            try:
                if self.interface_type == "socket":
                    self._handle_socket_communication()
                elif self.interface_type == "serial":
                    self._handle_serial_communication()
                else:  # Mock interface
                    self._handle_mock_communication()
                
                # Limit update rate
                elapsed = time.time() - last_time
                if elapsed < 0.01:  # 100Hz max
                    time.sleep(0.01 - elapsed)
                last_time = time.time()
                
            except Exception as e:
                print(f"HIL communication error: {e}")
                time.sleep(0.1)
    
    def _handle_socket_communication(self) -> None:
        """Handle socket-based hardware communication."""
        if not hasattr(self, 'client_socket'):
            # Wait for connection
            self.socket.settimeout(0.1)
            try:
                self.socket.listen(1)
                self.client_socket, _ = self.socket.accept()
                self.client_socket.settimeout(0.1)
                self.hardware_connected = True
                print("Hardware connected via socket")
            except socket.timeout:
                return
        
        # Send data to hardware
        try:
            self.client_socket.sendall((json.dumps(self.output_buffer) + '\n').encode())
            
            # Receive data from hardware
            data = self.client_socket.recv(1024).decode().strip()
            if data:
                self.input_buffer = json.loads(data)
                self._process_hardware_input()
        except (socket.timeout, ConnectionResetError):
            pass
    
    def _handle_serial_communication(self) -> None:
        """Handle serial-based hardware communication."""
        # Send data to hardware
        self.serial.write((json.dumps(self.output_buffer) + '\n').encode())
        
        # Receive data from hardware
        if self.serial.in_waiting:
            data = self.serial.readline().decode().strip()
            if data:
                try:
                    self.input_buffer = json.loads(data)
                    self._process_hardware_input()
                    self.hardware_connected = True
                except json.JSONDecodeError:
                    pass
    
    def _handle_mock_communication(self) -> None:
        """Handle mock hardware communication for testing."""
        # Simulate hardware processing delay
        time.sleep(0.01)
        
        # Create mock hardware response
        mock_response = {
            "control_outputs": {
                "aileron": np.sin(time.time()) * 0.2,
                "elevator": np.cos(time.time()) * 0.1,
                "rudder": 0.0,
                "throttle": 0.5
            },
            "hardware_time": time.time(),
            "status": "OK"
        }
        
        self.input_buffer = mock_response
        self._process_hardware_input()
        self.hardware_connected = True
    
    def _process_hardware_input(self) -> None:
        """Process input received from hardware."""
        if not self.input_buffer:
            return
            
        # Extract control outputs from hardware
        if "control_outputs" in self.input_buffer:
            control_outputs = self.input_buffer["control_outputs"]
            
            # Update digital twin with hardware control inputs
            if self.digital_twin and self.digital_twin.running:
                self.digital_twin.set_control_inputs(control_outputs)
        
        # Track hardware timing
        if "hardware_time" in self.input_buffer:
            current_time = self.input_buffer["hardware_time"]
            self.last_hardware_time = current_time
    
    def start(self) -> None:
        """Start the HIL interface."""
        if not self.running:
            self.running = True
            
            # Initialize socket server if needed
            if self.interface_type == "socket":
                try:
                    self.socket.bind(('localhost', self.port))
                    print(f"HIL interface listening on port {self.port}")
                except OSError as e:
                    print(f"Socket binding error: {e}")
                    self.running = False
                    return
            
            # Start communication thread
            self.hil_thread = threading.Thread(target=self._hardware_communication_loop)
            self.hil_thread.daemon = True
            self.hil_thread.start()
    
    def stop(self) -> None:
        """Stop the HIL interface."""
        self.running = False
        
        if self.hil_thread:
            self.hil_thread.join(timeout=1.0)
        
        # Close connections
        if self.interface_type == "socket":
            if hasattr(self, 'client_socket'):
                self.client_socket.close()
            self.socket.close()
        elif self.interface_type == "serial":
            self.serial.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Get HIL interface status."""
        return {
            "running": self.running,
            "hardware_connected": self.hardware_connected,
            "interface_type": self.interface_type,
            "last_hardware_time": self.last_hardware_time,
            "latency": time.time() - self.last_hardware_time if self.last_hardware_time > 0 else 0
        }


def run_hil_test(digital_twin, test_duration=30.0, interface_type="mock"):
    """Run a simple HIL test with the digital twin."""
    # Start digital twin if not already running
    was_running = digital_twin.running
    if not was_running:
        digital_twin.start()
    
    # Create and start HIL interface
    hil = HILInterface(digital_twin, interface_type=interface_type)
    hil.start()
    
    print(f"Starting HIL test for {test_duration} seconds...")
    start_time = time.time()
    
    # Monitor test progress
    while time.time() - start_time < test_duration:
        status = hil.get_status()
        if status["hardware_connected"]:
            print(f"Hardware connected, latency: {status['latency']*1000:.1f}ms")
        else:
            print("Waiting for hardware connection...")
        time.sleep(1.0)
    
    # Stop HIL interface
    hil.stop()
    
    # Stop digital twin if it wasn't running before
    if not was_running:
        digital_twin.stop()
    
    print("HIL test completed")
    return {
        "test_duration": test_duration,
        "hardware_connected": hil.hardware_connected,
        "final_status": hil.get_status()
    }


def connect_hardware(self, hardware_platform=None):
    """
    Connect to physical neuromorphic hardware.
    
    Args:
        hardware_platform: Hardware platform driver instance
        
    Returns:
        True if connection successful, False otherwise
    """
    if hardware_platform:
        self.hardware_platform = hardware_platform
        
    if not self.hardware_platform:
        print("No hardware platform specified")
        return False
        
    try:
        # Connect to the hardware platform
        success = self.hardware_platform.connect()
        if success:
            platform_name = self.hardware_platform.get_hardware_info()['platform']
            print(f"Connected to {platform_name}")
            
            # Configure platform-specific settings
            if "Loihi" in platform_name:
                self._configure_loihi_specific()
            elif "SpiNNaker" in platform_name:
                self._configure_spinnaker_specific()
            elif "TrueNorth" in platform_name:
                self._configure_truenorth_specific()
                
        return success
    except Exception as e:
        print(f"Failed to connect to hardware: {e}")
        return False

def _configure_loihi_specific(self):
    """Configure Loihi-specific settings."""
    config = {
        "real_time_mode": True,
        "host_sync_mode": "barrier",
        "enable_snips": True
    }
    self.hardware_platform.configure(config)
    
def _configure_spinnaker_specific(self):
    """Configure SpiNNaker-specific settings."""
    config = {
        "real_time_mode": True,
        "timestep": 1.0,
        "use_live_packet_gatherer": True,
        "use_live_spike_injector": True
    }
    self.hardware_platform.configure(config)
    
def _configure_truenorth_specific(self):
    """Configure TrueNorth-specific settings."""
    config = {
        "real_time_mode": True,
        "use_direct_memory_access": True,
        "binary_mode": True
    }
    self.hardware_platform.configure(config)

def run_hardware_step(self, input_data, duration):
    """
    Run a single step on the hardware platform.
    
    Args:
        input_data: Input data for the hardware
        duration: Duration to run in milliseconds
        
    Returns:
        Output data from the hardware
    """
    if not self.hardware_platform or not self.hardware_platform.is_connected():
        return None
        
    # Convert input data to appropriate format for the hardware platform
    platform_name = self.hardware_platform.get_hardware_info()['platform']
    
    if "SpiNNaker" in platform_name:
        # SpiNNaker uses spike times format
        input_spikes = self._convert_to_spinnaker_format(input_data)
    elif "TrueNorth" in platform_name:
        # TrueNorth uses binary spike format
        input_spikes = self._convert_to_truenorth_format(input_data)
    else:
        # Default conversion
        input_spikes = self._convert_to_spikes(input_data)
    
    # Run the hardware for the specified duration
    output_spikes = self.hardware_platform.run(input_spikes, duration)
    
    # Convert output spikes back to standard format
    return self._convert_from_spikes(output_spikes)

def _convert_to_spinnaker_format(self, data):
    """Convert data to SpiNNaker spike format."""
    if isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            # Convert to spike times dictionary
            spike_times = {}
            for i, neuron_idx in enumerate(data):
                if neuron_idx not in spike_times:
                    spike_times[neuron_idx] = []
                spike_times[neuron_idx].append(0)  # Spike at time 0
            return spike_times
        elif len(data.shape) == 2:
            # Convert 2D array to spike times dictionary
            spike_times = {}
            for i in range(data.shape[0]):
                if np.any(data[i]):
                    spike_times[i] = np.where(data[i] > 0)[0].tolist()
            return spike_times
    return data

def _convert_to_truenorth_format(self, data):
    """Convert data to TrueNorth binary spike format."""
    if isinstance(data, np.ndarray):
        # TrueNorth requires binary spikes
        return (data > 0.5).astype(np.uint8)
    return data

def _convert_to_spikes(self, data):
    """Convert input data to spike format."""
    if isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            return data
        elif len(data.shape) == 2:
            threshold = 0.5
            return np.where(data > threshold, 1, 0).astype(np.uint8)
    return np.array(data)

def _convert_from_spikes(self, spikes):
    """Convert spike data to standard format."""
    return spikes
