"""
Tools to train spiking neural networks on simulation data before deployment.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import time
import pickle
from ..control.snn_controller import SNNController

class SimulationTrainer:
    """Trains SNN controllers using simulation data."""
    
    def __init__(self, controller: SNNController, learning_rate: float = 0.01):
        """
        Initialize the simulation trainer.
        
        Args:
            controller: SNN controller to train
            learning_rate: Learning rate for weight updates
        """
        self.controller = controller
        self.learning_rate = learning_rate
        self.training_history = {
            'loss': [],
            'accuracy': []
        }
        
    def train_step(self, input_data: np.ndarray, target_output: np.ndarray) -> float:
        """
        Perform a single training step.
        
        Args:
            input_data: Input spike train (timesteps x input_size)
            target_output: Target output (timesteps x output_size)
            
        Returns:
            Loss value for this step
        """
        self.controller.reset()
        total_loss = 0.0
        output_spikes = []
        
        # Forward pass through time
        for t in range(len(input_data)):
            # Run controller for one timestep
            output_spike = self.controller.step(input_data[t])
            output_spikes.append(output_spike)
            
            # Calculate loss (mean squared error)
            loss = np.mean((output_spike - target_output[t])**2)
            total_loss += loss
        
        # Convert to numpy array for easier indexing
        output_spikes = np.array(output_spikes)
        
        # Simplified STDP-inspired learning
        self._update_weights(input_data, output_spikes, target_output)
        
        # Return average loss
        return total_loss / len(input_data)
    
    def _update_weights(self, inputs: np.ndarray, outputs: np.ndarray, targets: np.ndarray):
        """
        Update weights using a simplified STDP-inspired rule.
        
        Args:
            inputs: Input spike train
            outputs: Output spike train
            targets: Target output
        """
        # Calculate error
        error = targets - outputs
        
        # Update hidden-to-output weights
        for t in range(len(inputs)):
            # Get hidden layer activity
            hidden_activity = self.controller.spike_history[t][:self.controller.hidden_size]
            
            # Update weights based on error and presynaptic activity
            for i in range(self.controller.output_size):
                delta = error[t, i] * hidden_activity
                self.controller.weights_ho[i, :] += self.learning_rate * delta
        
        # Update input-to-hidden weights (simplified backpropagation)
        for t in range(len(inputs)):
            # Approximate hidden layer error
            hidden_error = np.dot(self.controller.weights_ho.T, error[t])
            
            # Update weights based on error and input activity
            for i in range(self.controller.hidden_size):
                delta = hidden_error[i] * inputs[t]
                self.controller.weights_ih[i, :] += self.learning_rate * delta
    
    def train(self, training_data: List[Tuple[np.ndarray, np.ndarray]], 
             epochs: int = 10, batch_size: int = 1) -> Dict[str, List[float]]:
        """
        Train the controller on simulation data.
        
        Args:
            training_data: List of (input, target) tuples
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        print(f"Training SNN controller for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            start_time = time.time()
            
            # Shuffle training data
            np.random.shuffle(training_data)
            
            # Process each training sample
            for i, (input_data, target_output) in enumerate(training_data):
                loss = self.train_step(input_data, target_output)
                epoch_loss += loss
                
                if (i + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(training_data)}, Loss: {loss:.4f}")
            
            # Calculate epoch statistics
            avg_loss = epoch_loss / len(training_data)
            self.training_history['loss'].append(avg_loss)
            
            # Evaluate accuracy on training data
            accuracy = self.evaluate(training_data)
            self.training_history['accuracy'].append(accuracy)
            
            print(f"Epoch {epoch+1}/{epochs} completed in {time.time() - start_time:.2f}s, "
                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return self.training_history
    
    def evaluate(self, evaluation_data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Evaluate the controller on data.
        
        Args:
            evaluation_data: List of (input, target) tuples
            
        Returns:
            Accuracy score (0-1)
        """
        correct = 0
        total = 0
        
        for input_data, target_output in evaluation_data:
            self.controller.reset()
            
            # Run through the sequence
            for t in range(len(input_data)):
                output = self.controller.step(input_data[t])
                
                # Count correct predictions (binary accuracy)
                correct += np.sum((output > 0.5) == (target_output[t] > 0.5))
                total += output.size
        
        return correct / total if total > 0 else 0.0
    
    def save_model(self, filepath: str):
        """
        Save the trained controller.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'weights_ih': self.controller.weights_ih,
            'weights_ho': self.controller.weights_ho,
            'thresholds': self.controller.thresholds,
            'decay_factors': self.controller.decay_factors,
            'input_size': self.controller.input_size,
            'hidden_size': self.controller.hidden_size,
            'output_size': self.controller.output_size,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained controller.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Update controller parameters
        self.controller.weights_ih = model_data['weights_ih']
        self.controller.weights_ho = model_data['weights_ho']
        self.controller.thresholds = model_data['thresholds']
        self.controller.decay_factors = model_data['decay_factors']
        
        if 'training_history' in model_data:
            self.training_history = model_data['training_history']
        
        print(f"Model loaded from {filepath}")


def generate_training_data(simulator, scenarios: List[Dict[str, Any]], 
                          timesteps: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate training data from simulation scenarios.
    
    Args:
        simulator: Simulation environment
        scenarios: List of scenario configurations
        timesteps: Number of timesteps per scenario
        
    Returns:
        List of (input, target) tuples
    """
    training_data = []
    
    for scenario in scenarios:
        # Configure simulator with scenario
        simulator.configure(scenario)
        
        # Run simulation
        sim_data = simulator.run(timesteps)
        
        # Extract input and target data
        input_spikes = sim_data['sensor_data']
        target_outputs = sim_data['optimal_control']
        
        # Convert to spike format if needed
        if not isinstance(input_spikes, np.ndarray):
            input_spikes = np.array(input_spikes)
        if not isinstance(target_outputs, np.ndarray):
            target_outputs = np.array(target_outputs)
        
        # Add to training data
        training_data.append((input_spikes, target_outputs))
    
    return training_data