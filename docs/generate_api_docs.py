#!/usr/bin/env python3
"""
Script to automatically generate API documentation for the Neuromorphic Biomimetic UCAV SDK.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Directory to save generated API docs
API_DIR = Path(__file__).parent / "api"
SRC_DIR = Path(__file__).parent.parent / "src"

# Create API directory if it doesn't exist
API_DIR.mkdir(exist_ok=True)


def generate_module_rst(module_path, package_name):
    """Generate RST file for a module."""
    module_name = module_path.stem
    if module_name == "__init__":
        return None
    
    rel_path = module_path.relative_to(SRC_DIR)
    import_path = f"{package_name}.{rel_path.parent.as_posix().replace('/', '.')}"
    if import_path.endswith("."):
        import_path = import_path[:-1]
    
    module_import = f"{import_path}.{module_name}"
    
    # Create RST content
    rst_content = f"""{module_name}
{'=' * len(module_name)}

.. automodule:: {module_import}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    
    # Determine output path
    output_dir = API_DIR / rel_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{module_name}.rst"
    
    # Write RST file
    with open(output_file, "w") as f:
        f.write(rst_content)
    
    return module_import, output_file


def generate_package_rst(package_path, package_name):
    """Generate RST file for a package."""
    package_rel_path = package_path.relative_to(SRC_DIR)
    import_path = f"{package_name}.{package_rel_path.as_posix().replace('/', '.')}"
    if import_path.endswith("."):
        import_path = import_path[:-1]
    
    # Get package name from path
    package_name = package_path.name
    
    # Find all modules in the package
    modules = []
    for item in package_path.glob("*.py"):
        if item.is_file() and item.stem != "__init__":
            modules.append(item.stem)
    
    # Find all subpackages
    subpackages = []
    for item in package_path.iterdir():
        if item.is_dir() and (item / "__init__.py").exists():
            subpackages.append(item.name)
    
    # Create RST content
    rst_content = f"""{package_name}
{'=' * len(package_name)}

.. automodule:: {import_path}
   :members:
   :undoc-members:
   :show-inheritance:

"""
    
    if subpackages:
        rst_content += "Subpackages\n-----------\n\n"
        rst_content += ".. toctree::\n   :maxdepth: 1\n\n"
        for subpackage in sorted(subpackages):
            rst_content += f"   {package_name}/{subpackage}/index\n"
        rst_content += "\n"
    
    if modules:
        rst_content += "Modules\n-------\n\n"
        rst_content += ".. toctree::\n   :maxdepth: 1\n\n"
        for module in sorted(modules):
            rst_content += f"   {package_name}/{module}\n"
    
    # Determine output path
    output_dir = API_DIR / package_rel_path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "index.rst"
    
    # Write RST file
    with open(output_file, "w") as f:
        f.write(rst_content)
    
    return import_path, output_file


def generate_api_index():
    """Generate the main API index file."""
    # Find all top-level packages
    packages = []
    for item in SRC_DIR.iterdir():
        if item.is_dir() and (item / "__init__.py").exists():
            packages.append(item.name)
    
    # Create RST content
    rst_content = """API Reference
=============

This section provides detailed API documentation for the Neuromorphic Biomimetic UCAV SDK.

.. toctree::
   :maxdepth: 2

"""
    for package in sorted(packages):
        rst_content += f"   {package}/index\n"
    
    # Write RST file
    with open(API_DIR / "index.rst", "w") as f:
        f.write(rst_content)


def main():
    """Main function to generate API documentation."""
    print("Generating API documentation...")
    
    # Get the package name
    package_name = "src"
    
    # Process all Python files
    for root, dirs, files in os.walk(SRC_DIR):
        root_path = Path(root)
        
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        
        # Process package directories
        if "__init__.py" in files:
            generate_package_rst(root_path, package_name)
        
        # Process module files
        for file in files:
            if file.endswith(".py"):
                generate_module_rst(root_path / file, package_name)
    
    # Generate main API index
    generate_api_index()
    
    print("API documentation generation complete.")


if __name__ == "__main__":
    main()