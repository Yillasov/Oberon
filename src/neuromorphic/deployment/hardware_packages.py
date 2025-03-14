"""
Deployment packages for various hardware configurations.
"""
import os
import json
import shutil
import zipfile
import platform
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging


class HardwareType(Enum):
    """Supported hardware types for deployment."""
    STANDARD_CPU = "standard_cpu"
    NEUROMORPHIC_ACCELERATOR = "neuromorphic_accelerator"
    FPGA = "fpga"
    GPU_ACCELERATED = "gpu_accelerated"
    EMBEDDED = "embedded"


@dataclass
class HardwareRequirements:
    """Hardware requirements for deployment."""
    min_cpu_cores: int = 1
    min_memory_gb: float = 1.0
    min_storage_gb: float = 5.0
    required_features: Set[str] = field(default_factory=set)
    recommended_os: List[str] = field(default_factory=list)
    
    def is_compatible(self, system_specs: Dict[str, Any]) -> bool:
        """Check if hardware meets requirements."""
        if system_specs.get("cpu_cores", 0) < self.min_cpu_cores:
            return False
        if system_specs.get("memory_gb", 0) < self.min_memory_gb:
            return False
        if system_specs.get("storage_gb", 0) < self.min_storage_gb:
            return False
            
        # Check required features
        for feature in self.required_features:
            if feature not in system_specs.get("features", []):
                return False
                
        return True


@dataclass
class DeploymentPackage:
    """Deployment package configuration."""
    name: str
    hardware_type: HardwareType
    requirements: HardwareRequirements
    components: List[str]
    config_template: str
    version: str = "1.0.0"
    description: str = ""


class DeploymentManager:
    """Manager for hardware-specific deployment packages."""
    
    def __init__(self, base_path: str = "/Users/yessine/Oberon"):
        self.base_path = base_path
        self.packages: Dict[str, DeploymentPackage] = {}
        self.logger = logging.getLogger("deployment_manager")
        
        # Create deployment directories
        self.packages_dir = os.path.join(base_path, "deployment", "packages")
        self.templates_dir = os.path.join(base_path, "deployment", "templates")
        self.output_dir = os.path.join(base_path, "deployment", "output")
        
        os.makedirs(self.packages_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load predefined packages
        self._load_predefined_packages()
    
    def _load_predefined_packages(self):
        """Load predefined deployment packages."""
        # Standard CPU package
        self.register_package(DeploymentPackage(
            name="standard_cpu",
            hardware_type=HardwareType.STANDARD_CPU,
            requirements=HardwareRequirements(
                min_cpu_cores=2,
                min_memory_gb=4.0,
                min_storage_gb=10.0,
                recommended_os=["Linux", "macOS", "Windows"]
            ),
            components=[
                "system_core",
                "component_registry",
                "event_bus",
                "config_loader",
                "health_monitor",
                "validation_framework",
                "system_lifecycle"
            ],
            config_template="standard_config.yaml",
            description="Standard deployment for CPU-based systems"
        ))
        
        # Neuromorphic accelerator package
        self.register_package(DeploymentPackage(
            name="neuromorphic_accelerator",
            hardware_type=HardwareType.NEUROMORPHIC_ACCELERATOR,
            requirements=HardwareRequirements(
                min_cpu_cores=4,
                min_memory_gb=8.0,
                min_storage_gb=20.0,
                required_features={"neuromorphic_hardware"},
                recommended_os=["Linux"]
            ),
            components=[
                "system_core",
                "component_registry",
                "event_bus",
                "config_loader",
                "health_monitor",
                "validation_framework",
                "system_lifecycle",
                "neuromorphic_runtime"
            ],
            config_template="neuromorphic_config.yaml",
            description="Deployment for systems with neuromorphic accelerators"
        ))
        
        # FPGA package
        self.register_package(DeploymentPackage(
            name="fpga",
            hardware_type=HardwareType.FPGA,
            requirements=HardwareRequirements(
                min_cpu_cores=2,
                min_memory_gb=4.0,
                min_storage_gb=10.0,
                required_features={"fpga_support"},
                recommended_os=["Linux"]
            ),
            components=[
                "system_core",
                "component_registry",
                "event_bus",
                "config_loader",
                "health_monitor",
                "validation_framework",
                "system_lifecycle",
                "fpga_runtime"
            ],
            config_template="fpga_config.yaml",
            description="Deployment for FPGA-based systems"
        ))
        
        # GPU accelerated package
        self.register_package(DeploymentPackage(
            name="gpu_accelerated",
            hardware_type=HardwareType.GPU_ACCELERATED,
            requirements=HardwareRequirements(
                min_cpu_cores=4,
                min_memory_gb=16.0,
                min_storage_gb=50.0,
                required_features={"cuda_support"},
                recommended_os=["Linux", "Windows"]
            ),
            components=[
                "system_core",
                "component_registry",
                "event_bus",
                "config_loader",
                "health_monitor",
                "validation_framework",
                "system_lifecycle",
                "gpu_runtime"
            ],
            config_template="gpu_config.yaml",
            description="Deployment for GPU-accelerated systems"
        ))
        
        # Embedded package
        self.register_package(DeploymentPackage(
            name="embedded",
            hardware_type=HardwareType.EMBEDDED,
            requirements=HardwareRequirements(
                min_cpu_cores=1,
                min_memory_gb=1.0,
                min_storage_gb=2.0,
                recommended_os=["Linux"]
            ),
            components=[
                "system_core",
                "component_registry",
                "event_bus",
                "config_loader",
                "health_monitor",
                "system_lifecycle"
            ],
            config_template="embedded_config.yaml",
            description="Minimal deployment for embedded systems"
        ))
    
    def register_package(self, package: DeploymentPackage) -> bool:
        """Register a deployment package."""
        if package.name in self.packages:
            self.logger.warning(f"Package {package.name} already registered")
            return False
            
        self.packages[package.name] = package
        return True
    
    def get_package(self, name: str) -> Optional[DeploymentPackage]:
        """Get a deployment package by name."""
        return self.packages.get(name)
    
    def list_packages(self) -> List[Dict[str, Any]]:
        """List all available deployment packages."""
        return [
            {
                "name": pkg.name,
                "hardware_type": pkg.hardware_type.value,
                "description": pkg.description,
                "version": pkg.version
            }
            for pkg in self.packages.values()
        ]
    
    def detect_system_specs(self) -> Dict[str, Any]:
        """Detect current system specifications."""
        specs = {
            "cpu_cores": os.cpu_count() or 1,
            "platform": platform.system(),
            "platform_version": platform.version(),
            "machine": platform.machine(),
            "features": []
        }
        
        # Try to detect memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            specs["memory_gb"] = mem.total / (1024 ** 3)
            
            # Detect storage
            disk = psutil.disk_usage('/')
            specs["storage_gb"] = disk.total / (1024 ** 3)
        except ImportError:
            # Fallback values
            specs["memory_gb"] = 4.0
            specs["storage_gb"] = 20.0
        
        # Detect GPU/CUDA
        try:
            import torch
            if torch.cuda.is_available():
                specs["features"].append("cuda_support")
                specs["gpu_count"] = torch.cuda.device_count()
                specs["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except ImportError:
            pass
        
        return specs
    
    def find_compatible_packages(self) -> List[str]:
        """Find packages compatible with current system."""
        specs = self.detect_system_specs()
        compatible = []
        
        for name, package in self.packages.items():
            if package.requirements.is_compatible(specs):
                compatible.append(name)
                
        return compatible
    
    def create_deployment(self, package_name: str, output_name: Optional[str] = None) -> Optional[str]:
        """Create deployment package for specified hardware."""
        package = self.packages.get(package_name)
        if not package:
            self.logger.error(f"Package {package_name} not found")
            return None
            
        if not output_name:
            output_name = f"{package_name}_deployment"
            
        # Create deployment directory
        deploy_dir = os.path.join(self.output_dir, output_name)
        os.makedirs(deploy_dir, exist_ok=True)
        
        # Copy component files
        src_dir = os.path.join(self.base_path, "src", "neuromorphic", "integration")
        for component in package.components:
            src_file = os.path.join(src_dir, f"{component}.py")
            if os.path.exists(src_file):
                shutil.copy2(src_file, deploy_dir)
            else:
                self.logger.warning(f"Component file not found: {src_file}")
        
        # Copy config template
        template_file = os.path.join(self.templates_dir, package.config_template)
        if os.path.exists(template_file):
            shutil.copy2(template_file, os.path.join(deploy_dir, "config.yaml"))
        else:
            # Create basic config if template doesn't exist
            with open(os.path.join(deploy_dir, "config.yaml"), "w") as f:
                f.write(f"# Configuration for {package.name} deployment\n")
                f.write("system:\n")
                f.write(f"  hardware_type: {package.hardware_type.value}\n")
                f.write(f"  version: {package.version}\n")
        
        # Create package metadata
        metadata = {
            "name": package.name,
            "version": package.version,
            "description": package.description,
            "hardware_type": package.hardware_type.value,
            "components": package.components,
            "requirements": {
                "min_cpu_cores": package.requirements.min_cpu_cores,
                "min_memory_gb": package.requirements.min_memory_gb,
                "min_storage_gb": package.requirements.min_storage_gb,
                "required_features": list(package.requirements.required_features),
                "recommended_os": package.requirements.recommended_os
            }
        }
        
        with open(os.path.join(deploy_dir, "package.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create deployment archive
        archive_path = os.path.join(self.output_dir, f"{output_name}.zip")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(deploy_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, deploy_dir))
        
        self.logger.info(f"Created deployment package: {archive_path}")
        return archive_path