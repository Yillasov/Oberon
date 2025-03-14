"""
Real-time visualization of control-airframe dynamics.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, Any, Optional
import threading
import queue


class RealtimeVisualizer:
    """Simple 2D real-time visualizer for airframe dynamics."""
    
    def __init__(self, update_rate: float = 30.0):
        """
        Initialize real-time visualizer.
        
        Args:
            update_rate: Visualization update rate in Hz
        """
        self.update_rate = update_rate
        self.data_queue = queue.Queue()
        self.running = False
        
        # Initialize plots
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Setup airframe view
        self.ax1.set_xlim(-50, 50)
        self.ax1.set_ylim(-50, 50)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True)
        self.ax1.set_title('Airframe Position and Attitude')
        
        # Setup control response view
        self.ax2.set_xlim(0, 10)  # 10 seconds window
        self.ax2.set_ylim(-1, 1)
        self.ax2.grid(True)
        self.ax2.set_title('Control Surface Positions')
        
        # Initialize visualization elements
        self._init_plot_elements()
        
        # Data buffers
        self.time_buffer = []
        self.control_buffers = {
            'aileron': [],
            'elevator': [],
            'rudder': []
        }
    
    def _init_plot_elements(self):
        """Initialize plot elements."""
        # Airframe representation (simple aircraft symbol)
        self.aircraft_body, = self.ax1.plot([], [], 'b-', lw=2)
        self.aircraft_wing, = self.ax1.plot([], [], 'b-', lw=2)
        self.aircraft_tail, = self.ax1.plot([], [], 'b-', lw=2)
        
        # Control surface plots
        self.control_lines = {}
        for control, color in [('aileron', 'r'), ('elevator', 'g'), ('rudder', 'b')]:
            line, = self.ax2.plot([], [], f'{color}-', label=control)
            self.control_lines[control] = line
        self.ax2.legend()
    
    def _update_aircraft_shape(self, position, attitude):
        """Update aircraft shape based on position and attitude."""
        # Basic aircraft shape (simplified)
        length = 10.0
        wingspan = 8.0
        
        # Apply rotation
        cos_yaw = np.cos(attitude[2])
        sin_yaw = np.sin(attitude[2])
        
        # Fuselage
        nose = position + np.array([length/2 * cos_yaw, length/2 * sin_yaw])
        tail = position - np.array([length/2 * cos_yaw, length/2 * sin_yaw])
        
        # Wings
        wing_left = position + np.array([-wingspan/2 * sin_yaw, wingspan/2 * cos_yaw])
        wing_right = position - np.array([-wingspan/2 * sin_yaw, wingspan/2 * cos_yaw])
        
        # Tail
        tail_span = wingspan * 0.4
        tail_left = tail + np.array([-tail_span/2 * sin_yaw, tail_span/2 * cos_yaw])
        tail_right = tail - np.array([-tail_span/2 * sin_yaw, tail_span/2 * cos_yaw])
        
        # Update plot data
        self.aircraft_body.set_data([tail[0], position[0], nose[0]], 
                                  [tail[1], position[1], nose[1]])
        self.aircraft_wing.set_data([wing_left[0], wing_right[0]], 
                                  [wing_left[1], wing_right[1]])
        self.aircraft_tail.set_data([tail_left[0], tail_right[0]], 
                                  [tail_left[1], tail_right[1]])
    
    def update(self, state: Dict[str, Any]):
        """Update visualization with new state data."""
        if not self.running:
            return
            
        self.data_queue.put(state)
    
    def _update_visualization(self):
        """Update visualization elements."""
        while self.running:
            try:
                state = self.data_queue.get(timeout=1.0/self.update_rate)
                
                # Extract position and attitude
                position = np.array(state["position"][:2])  # 2D position
                attitude = np.array(state["attitude"])
                
                # Update aircraft visualization
                self._update_aircraft_shape(position, attitude)
                
                # Update control plots
                current_time = state["time"]
                control_inputs = state.get("control_inputs", {})
                
                # Update buffers
                self.time_buffer.append(current_time)
                for control in self.control_buffers:
                    self.control_buffers[control].append(
                        control_inputs.get(control, 0.0))
                
                # Trim buffers to window size
                window_size = int(10 * self.update_rate)  # 10 seconds
                if len(self.time_buffer) > window_size:
                    self.time_buffer = self.time_buffer[-window_size:]
                    for control in self.control_buffers:
                        self.control_buffers[control] = \
                            self.control_buffers[control][-window_size:]
                
                # Update control plots
                for control, line in self.control_lines.items():
                    line.set_data(self.time_buffer, self.control_buffers[control])
                
                # Update axis limits for control plot
                if self.time_buffer:
                    self.ax2.set_xlim(
                        max(0, self.time_buffer[-1] - 10),
                        max(10, self.time_buffer[-1]))
                
                # Redraw
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Visualization error: {e}")
    
    def start(self):
        """Start the visualization."""
        if not self.running:
            self.running = True
            self.viz_thread = threading.Thread(target=self._update_visualization)
            self.viz_thread.daemon = True
            self.viz_thread.start()
    
    def stop(self):
        """Stop the visualization."""
        self.running = False
        if hasattr(self, 'viz_thread'):
            self.viz_thread.join(timeout=1.0)
        plt.close(self.fig)


def create_realtime_display(digital_twin) -> RealtimeVisualizer:
    """Create and connect a realtime display to a digital twin."""
    visualizer = RealtimeVisualizer()
    visualizer.start()
    
    # Register callback with digital twin
    digital_twin.register_callback(visualizer.update)
    
    return visualizer