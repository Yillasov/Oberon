"""
Utilities to match simulation parameters with real hardware characteristics.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

def match_neuron_parameters(sim_params: Dict[str, Any], 
                           hardware_platform: str) -> Dict[str, Any]:
    """
    Match simulation neuron parameters to hardware-specific parameters.
    
    Args:
        sim_params: Simulation neuron parameters
        hardware_platform: Target hardware platform name
        
    Returns:
        Hardware-specific neuron parameters
    """
    if "loihi" in hardware_platform.lower():
        return _match_loihi_neuron_params(sim_params)
    elif "spinnaker" in hardware_platform.lower():
        return _match_spinnaker_neuron_params(sim_params)
    elif "truenorth" in hardware_platform.lower():
        return _match_truenorth_neuron_params(sim_params)
    else:
        return sim_params  # Return unchanged for unknown platforms

def _match_loihi_neuron_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Match neuron parameters to Loihi hardware constraints."""
    hw_params = params.copy()
    
    # Loihi uses 12-bit signed integer for decay
    if "decay" in hw_params:
        hw_params["decay"] = max(min(int(hw_params["decay"] * 4096), 2047), -2048)
        
    # Loihi uses 12-bit unsigned integer for threshold
    if "threshold" in hw_params:
        hw_params["threshold"] = max(min(int(hw_params["threshold"] * 4096), 4095), 0)
        
    # Loihi uses 12-bit signed integer for bias
    if "bias" in hw_params:
        hw_params["bias"] = max(min(int(hw_params["bias"] * 4096), 2047), -2048)
    
    return hw_params

def _match_spinnaker_neuron_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Match neuron parameters to SpiNNaker hardware constraints."""
    hw_params = params.copy()
    
    # SpiNNaker uses 32-bit floating point, but with limited precision
    if "tau_m" in hw_params:  # Membrane time constant
        hw_params["tau_m"] = max(hw_params["tau_m"], 0.1)
        
    if "tau_refrac" in hw_params:  # Refractory period
        hw_params["tau_refrac"] = max(hw_params["tau_refrac"], 0.0)
        
    if "v_reset" in hw_params:  # Reset voltage
        hw_params["v_reset"] = float(hw_params["v_reset"])
        
    if "v_rest" in hw_params:  # Resting voltage
        hw_params["v_rest"] = float(hw_params["v_rest"])
    
    return hw_params

def _match_truenorth_neuron_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Match neuron parameters to TrueNorth hardware constraints."""
    hw_params = {}
    
    # TrueNorth uses 9-bit unsigned integer for threshold
    if "threshold" in params:
        hw_params["threshold"] = max(min(int(params["threshold"] * 256), 511), 0)
        
    # TrueNorth uses 8-bit signed integer for leak
    if "leak" in params:
        hw_params["leak"] = max(min(int(params["leak"] * 128), 127), -128)
        
    # TrueNorth uses 9-bit signed integer for reset
    if "reset" in params:
        hw_params["reset"] = max(min(int(params["reset"] * 256), 255), -256)
    
    return hw_params

def match_weight_parameters(weights: np.ndarray, 
                           hardware_platform: str) -> np.ndarray:
    """
    Match simulation weights to hardware-specific weight representation.
    
    Args:
        weights: Simulation weight matrix
        hardware_platform: Target hardware platform name
        
    Returns:
        Hardware-specific weight matrix
    """
    if "loihi" in hardware_platform.lower():
        return _match_loihi_weights(weights)
    elif "spinnaker" in hardware_platform.lower():
        return _match_spinnaker_weights(weights)
    elif "truenorth" in hardware_platform.lower():
        return _match_truenorth_weights(weights)
    else:
        return weights  # Return unchanged for unknown platforms

def _match_loihi_weights(weights: np.ndarray) -> np.ndarray:
    """Match weights to Loihi hardware constraints."""
    # Loihi uses 8-bit signed weights
    return np.clip(weights * 128, -128, 127).astype(np.int8)

def _match_spinnaker_weights(weights: np.ndarray) -> np.ndarray:
    """Match weights to SpiNNaker hardware constraints."""
    # SpiNNaker typically uses 16-bit fixed-point weights (Q8.8)
    return np.clip(weights * 256, -32768, 32767).astype(np.int16)

def _match_truenorth_weights(weights: np.ndarray) -> np.ndarray:
    """Match weights to TrueNorth hardware constraints."""
    # TrueNorth uses binary weights
    return (weights > 0.5).astype(np.uint8)

def estimate_hardware_resources(network_params: Dict[str, Any], 
                               hardware_platform: str) -> Dict[str, Any]:
    """
    Estimate hardware resources required for a network.
    
    Args:
        network_params: Network parameters including neuron count and connectivity
        hardware_platform: Target hardware platform name
        
    Returns:
        Dictionary of estimated hardware resources
    """
    neuron_count = network_params.get("neuron_count", 0)
    synapse_count = network_params.get("synapse_count", 0)
    
    if "loihi" in hardware_platform.lower():
        # Loihi has 128 cores with 1024 neurons per core
        cores_needed = (neuron_count + 1023) // 1024
        return {
            "cores_used": cores_needed,
            "neurons_used": neuron_count,
            "synapses_used": synapse_count,
            "utilization": min(cores_needed / 128, 1.0)
        }
    elif "spinnaker" in hardware_platform.lower():
        # SpiNNaker has ~100 neurons per core
        cores_needed = (neuron_count + 99) // 100
        return {
            "cores_used": cores_needed,
            "neurons_used": neuron_count,
            "synapses_used": synapse_count,
            "sdram_used": synapse_count * 4 + neuron_count * 32  # Rough estimate
        }
    elif "truenorth" in hardware_platform.lower():
        # TrueNorth has 4096 cores with 256 neurons per core
        cores_needed = (neuron_count + 255) // 256
        return {
            "cores_used": cores_needed,
            "neurons_used": neuron_count,
            "axons_used": min(synapse_count, 256 * cores_needed),
            "utilization": min(cores_needed / 4096, 1.0)
        }
    else:
        return {
            "neurons_used": neuron_count,
            "synapses_used": synapse_count
        }