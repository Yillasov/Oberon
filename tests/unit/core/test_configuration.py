"""
Unit tests for the Configuration class.
"""
import os
import json
import yaml
import pytest
from pathlib import Path
from src.core import Configuration, ConfigurationError


class TestConfiguration:
    """Tests for the Configuration class."""
    
    def test_basic_config_operations(self):
        """Test basic configuration operations."""
        config = Configuration()
        
        # Test setting and getting values
        config.set("test.key", "value")
        assert config.get("test.key") == "value"
        
        # Test default values
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", "default") == "default"
        
        # Test nested values
        config.set("nested.key1.key2", "nested_value")
        assert config.get("nested.key1.key2") == "nested_value"
        
        # Test updating values
        config.update({"test": {"key": "updated_value"}})
        assert config.get("test.key") == "updated_value"
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Configuration()
        config.set("key1", "value1")
        config.set("key2.nested", "value2")
        
        config_dict = config.to_dict()
        assert config_dict["key1"] == "value1"
        assert config_dict["key2"]["nested"] == "value2"
    
    def test_config_clear(self):
        """Test clearing configuration."""
        config = Configuration({"key": "value"})
        assert config.get("key") == "value"
        
        config.clear()
        assert config.get("key") is None
    
    def test_load_json_config(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_data = {"test": {"key": "value"}}
        config_file = tmp_path / "config.json"
        
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        config = Configuration()
        config.load_file(config_file)
        
        assert config.get("test.key") == "value"
    
    def test_load_yaml_config(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_data = {"test": {"key": "value"}}
        config_file = tmp_path / "config.yaml"
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config = Configuration()
        config.load_file(config_file)
        
        assert config.get("test.key") == "value"
    
    def test_save_config(self, tmp_path):
        """Test saving configuration to file."""
        config = Configuration({"test": {"key": "value"}})
        
        # Test saving as JSON
        json_file = tmp_path / "config.json"
        config.save_file(json_file)
        assert json_file.exists()
        
        # Test saving as YAML
        yaml_file = tmp_path / "config.yaml"
        config.save_file(yaml_file)
        assert yaml_file.exists()
        
        # Test loading the saved files
        json_config = Configuration()
        json_config.load_file(json_file)
        assert json_config.get("test.key") == "value"
        
        yaml_config = Configuration()
        yaml_config.load_file(yaml_file)
        assert yaml_config.get("test.key") == "value"
    
    def test_invalid_file_format(self, tmp_path):
        """Test handling of invalid file formats."""
        config = Configuration()
        invalid_file = tmp_path / "config.txt"
        
        with pytest.raises(ConfigurationError):
            config.save_file(invalid_file)
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        config = Configuration()
        
        with pytest.raises(ConfigurationError):
            config.load_file("/nonexistent/file.json")