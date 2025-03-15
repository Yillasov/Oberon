"""
Tools to package and deploy neuromorphic models to hardware.
"""
import os
import json
import shutil
import zipfile
import pickle
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from ..config.hardware_config import HardwareConfig, HardwarePlatform

class ModelPackager:
    """Packages neuromorphic models for deployment to hardware."""
    
    def __init__(self, output_dir: str = "/Users/yessine/Oberon/models"):
        """
        Initialize model packager.
        
        Args:
            output_dir: Directory for packaged models
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger("ModelPackager")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def package_model(self, model_data: Dict[str, Any], 
                     model_name: str,
                     target_platform: HardwarePlatform,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Package a model for deployment.
        
        Args:
            model_data: Model weights, parameters, and structure
            model_name: Name of the model
            target_platform: Target hardware platform
            metadata: Additional metadata
            
        Returns:
            Path to packaged model
        """
        # Create timestamp for versioning
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_name}_{timestamp}"
        
        # Create model directory
        model_dir = os.path.join(self.output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "model_name": model_name,
            "model_id": model_id,
            "target_platform": target_platform.value,
            "created_at": timestamp,
            "version": "1.0"
        })
        
        # Save model data
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create hardware-specific configuration
        hw_config = HardwareConfig(target_platform)
        config_path = os.path.join(model_dir, "hardware_config.json")
        hw_config.save_config(config_path)
        
        # Create deployment package (zip)
        package_path = os.path.join(self.output_dir, f"{model_id}.zip")
        with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, model_dir)
                    zipf.write(file_path, arcname)
        
        self.logger.info(f"Model packaged successfully: {package_path}")
        return package_path


class HardwareDeployer:
    """Deploys packaged models to neuromorphic hardware."""
    
    def __init__(self):
        """Initialize hardware deployer."""
        self.logger = logging.getLogger("HardwareDeployer")
        self.deployment_history = []
    
    def deploy_model(self, package_path: str, 
                    hardware_address: str,
                    credentials: Optional[Dict[str, str]] = None) -> bool:
        """
        Deploy a packaged model to hardware.
        
        Args:
            package_path: Path to packaged model
            hardware_address: Address of target hardware
            credentials: Authentication credentials
            
        Returns:
            True if deployment was successful
        """
        if not os.path.exists(package_path):
            self.logger.error(f"Package not found: {package_path}")
            return False
        
        # Extract package to temporary directory
        temp_dir = os.path.join(os.path.dirname(package_path), "temp_deploy")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Extract package
            with zipfile.ZipFile(package_path, "r") as zipf:
                zipf.extractall(temp_dir)
            
            # Load metadata
            metadata_path = os.path.join(temp_dir, "metadata.json")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Determine target platform
            target_platform = metadata.get("target_platform", "simulation")
            
            # Deploy based on platform
            success = self._deploy_to_platform(temp_dir, target_platform, hardware_address, credentials)
            
            if success:
                # Record successful deployment
                self.deployment_history.append({
                    "package": os.path.basename(package_path),
                    "hardware": hardware_address,
                    "platform": target_platform,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "success"
                })
                self.logger.info(f"Model deployed successfully to {hardware_address}")
            else:
                self.logger.error(f"Deployment failed to {hardware_address}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Deployment error: {e}")
            return False
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _deploy_to_platform(self, model_dir: str, 
                           platform: str, 
                           hardware_address: str,
                           credentials: Optional[Dict[str, str]]) -> bool:
        """
        Deploy model to specific hardware platform.
        
        Args:
            model_dir: Directory containing extracted model
            platform: Target platform
            hardware_address: Hardware address
            credentials: Authentication credentials
            
        Returns:
            True if deployment was successful
        """
        # Load model data
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Load hardware configuration
        config_path = os.path.join(model_dir, "hardware_config.json")
        hw_config = HardwareConfig()
        hw_config.load_config(config_path)
        
        if platform == HardwarePlatform.LOIHI.value:
            return self._deploy_to_loihi(model_data, hw_config, hardware_address, credentials)
        elif platform == HardwarePlatform.SPINNAKER.value:
            return self._deploy_to_spinnaker(model_data, hw_config, hardware_address, credentials)
        elif platform == HardwarePlatform.TRUENORTH.value:
            return self._deploy_to_truenorth(model_data, hw_config, hardware_address, credentials)
        else:
            self.logger.warning(f"Unsupported platform: {platform}, using simulation")
            return self._deploy_to_simulation(model_data, hw_config)
    
    def _deploy_to_loihi(self, model_data: Dict[str, Any], 
                        hw_config: HardwareConfig,
                        hardware_address: str,
                        credentials: Optional[Dict[str, str]]) -> bool:
        """Deploy to Loihi hardware."""
        self.logger.info(f"Deploying to Loihi at {hardware_address}")
        
        try:
            # In a real implementation, this would use the NxSDK API
            # to deploy the model to Loihi hardware
            
            # Example pseudocode:
            # import nxsdk
            # board = nxsdk.connect(hardware_address)
            # compiler = nxsdk.Compiler()
            # compiled_network = compiler.compile(model_data, hw_config.config_data)
            # board.deploy(compiled_network)
            
            # Simulate successful deployment
            time.sleep(2)  # Simulate deployment time
            return True
            
        except Exception as e:
            self.logger.error(f"Loihi deployment error: {e}")
            return False
    
    def _deploy_to_spinnaker(self, model_data: Dict[str, Any], 
                            hw_config: HardwareConfig,
                            hardware_address: str,
                            credentials: Optional[Dict[str, str]]) -> bool:
        """Deploy to SpiNNaker hardware."""
        self.logger.info(f"Deploying to SpiNNaker at {hardware_address}")
        
        try:
            # In a real implementation, this would use the PyNN/SpiNNaker API
            # to deploy the model to SpiNNaker hardware
            
            # Example pseudocode:
            # import pyNN.spiNNaker as sim
            # sim.setup(timestep=hw_config.get("runtime", "timestep_us") / 1000.0)
            # sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)
            # network = sim.Population(...)
            # sim.run(...)
            
            # Simulate successful deployment
            time.sleep(2)  # Simulate deployment time
            return True
            
        except Exception as e:
            self.logger.error(f"SpiNNaker deployment error: {e}")
            return False
    
    def _deploy_to_truenorth(self, model_data: Dict[str, Any], 
                            hw_config: HardwareConfig,
                            hardware_address: str,
                            credentials: Optional[Dict[str, str]]) -> bool:
        """Deploy to TrueNorth hardware."""
        self.logger.info(f"Deploying to TrueNorth at {hardware_address}")
        
        try:
            # In a real implementation, this would use the TrueNorth API
            # to deploy the model to TrueNorth hardware
            
            # Simulate successful deployment
            time.sleep(2)  # Simulate deployment time
            return True
            
        except Exception as e:
            self.logger.error(f"TrueNorth deployment error: {e}")
            return False
    
    def _deploy_to_simulation(self, model_data: Dict[str, Any], 
                             hw_config: HardwareConfig) -> bool:
        """Deploy to simulation environment."""
        self.logger.info("Deploying to simulation environment")
        
        try:
            # In a real implementation, this would configure a simulation
            # environment with the model
            
            # Simulate successful deployment
            time.sleep(1)  # Simulate deployment time
            return True
            
        except Exception as e:
            self.logger.error(f"Simulation deployment error: {e}")
            return False
    
    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """
        Get deployment history.
        
        Returns:
            List of deployment records
        """
        return self.deployment_history


# Simple usage example
def package_and_deploy_example():
    """Example of packaging and deploying a model."""
    # Create sample model data
    model_data = {
        "weights_ih": [[0.1, 0.2], [0.3, 0.4]],
        "weights_ho": [[0.5, 0.6]],
        "thresholds": [1.0, 1.0, 1.0],
        "neuron_count": 3,
        "synapse_count": 4
    }
    
    # Package the model
    packager = ModelPackager()
    package_path = packager.package_model(
        model_data=model_data,
        model_name="test_model",
        target_platform=HardwarePlatform.LOIHI,
        metadata={"description": "Test model for deployment"}
    )
    
    # Deploy the model
    deployer = HardwareDeployer()
    success = deployer.deploy_model(
        package_path=package_path,
        hardware_address="loihi.example.com:22",
        credentials={"username": "user", "password": "pass"}
    )
    
    print(f"Deployment {'successful' if success else 'failed'}")
    
    # Get deployment history
    history = deployer.get_deployment_history()
    print(f"Deployment history: {history}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    package_and_deploy_example()