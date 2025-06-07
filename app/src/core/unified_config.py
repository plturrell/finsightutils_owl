"""
Unified Configuration System

This module provides a single, cohesive configuration system for the OWL platform,
combining environment variables, configuration files, and defaults into a simple,
elegant interface that aligns with Jony Ive's philosophy of simplicity and purpose.
"""
import os
import sys
import yaml
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger("owl.config")

# Configuration models
class APIConfig(BaseModel):
    """API configuration"""
    host: str = Field("0.0.0.0", description="Host to bind the API to")
    port: int = Field(8000, description="Port to bind the API to")
    root_path: str = Field("", description="Root path for API")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    cors_allow_credentials: bool = Field(True, description="Allow credentials in CORS")
    max_upload_size: int = Field(20 * 1024 * 1024, description="Maximum upload size in bytes")
    
    class Config:
        env_prefix = "API_"

class ProcessingConfig(BaseModel):
    """Document processing configuration"""
    cache_dir: Optional[str] = Field(None, description="Cache directory")
    base_uri: str = Field("http://example.org/ontology/", description="Base URI for ontology")
    include_provenance: bool = Field(True, description="Include provenance in output")
    max_concurrent_tasks: int = Field(5, description="Maximum concurrent tasks")
    task_timeout: int = Field(600, description="Task timeout in seconds")
    
    class Config:
        env_prefix = "PROCESSING_"

class CudaConfig(BaseModel):
    """CUDA configuration"""
    enabled: bool = Field(True, description="Enable CUDA acceleration")
    device_id: int = Field(0, description="CUDA device ID")
    memory_limit: Optional[int] = Field(None, description="GPU memory limit in MB")
    
    class Config:
        env_prefix = "CUDA_"

class IntegrationConfig(BaseModel):
    """Integration configuration"""
    deplot_enabled: bool = Field(True, description="Enable DePlot integration")
    nam_enabled: bool = Field(True, description="Enable NAM integration")
    dvrl_enabled: bool = Field(True, description="Enable DVRL integration")
    optlist_enabled: bool = Field(True, description="Enable Opt_list integration")
    tools_dir: str = Field("/app/tools", description="Directory for integrated tools")
    
    class Config:
        env_prefix = "INTEGRATION_"

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field("INFO", description="Logging level")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file: Optional[str] = Field(None, description="Log file path")
    
    class Config:
        env_prefix = "LOGGING_"

class UIConfig(BaseModel):
    """UI configuration"""
    theme: str = Field("light", description="UI theme")
    animation_enabled: bool = Field(True, description="Enable UI animations")
    custom_css: Optional[str] = Field(None, description="Custom CSS file path")
    
    class Config:
        env_prefix = "UI_"

class UnifiedConfig(BaseModel):
    """Unified configuration for the OWL platform"""
    api: APIConfig = Field(default_factory=APIConfig, description="API configuration")
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig, description="Processing configuration")
    cuda: CudaConfig = Field(default_factory=CudaConfig, description="CUDA configuration")
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig, description="Integration configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    ui: UIConfig = Field(default_factory=UIConfig, description="UI configuration")
    debug: bool = Field(False, description="Debug mode")
    environment: str = Field("production", description="Environment (development, testing, production)")

    class Config:
        env_prefix = ""

# Singleton configuration instance
_config_instance = None

def load_config(
    config_path: Optional[str] = None, 
    env_file: Optional[str] = None,
    environment: Optional[str] = None
) -> UnifiedConfig:
    """
    Load the unified configuration
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        env_file: Path to environment file
        environment: Environment name (overrides the one in config file)
        
    Returns:
        Unified configuration instance
    """
    global _config_instance
    
    if _config_instance is not None:
        return _config_instance
    
    # Start with default config
    config_dict = {}
    
    # Load configuration file if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, "r") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    config_dict = json.load(f)
                else:
                    logger.warning(f"Unknown configuration file format: {config_path.suffix}")
    
    # Load environment file if provided
    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            logger.info(f"Loading environment from {env_path}")
            _load_env_file(env_path)
    
    # Override environment if provided
    if environment:
        config_dict["environment"] = environment
    
    # Create config instance
    _config_instance = UnifiedConfig(**config_dict)
    
    # Override with environment variables
    _apply_env_overrides(_config_instance)
    
    return _config_instance

def get_config() -> UnifiedConfig:
    """
    Get the unified configuration instance
    
    Returns:
        Unified configuration instance
    """
    global _config_instance
    
    if _config_instance is None:
        return load_config()
    
    return _config_instance

def _load_env_file(env_path: Path) -> None:
    """
    Load environment variables from file
    
    Args:
        env_path: Path to environment file
    """
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip("'\"")

def _apply_env_overrides(config: UnifiedConfig) -> None:
    """
    Apply environment variable overrides to config
    
    Args:
        config: Configuration instance to update
    """
    # Helper function to set nested attributes
    def set_nested_attr(obj, path, value):
        parts = path.split("__")
        for i, part in enumerate(parts[:-1]):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return
        if hasattr(obj, parts[-1]):
            attr = parts[-1]
            current_value = getattr(obj, attr)
            # Convert value to the appropriate type
            if isinstance(current_value, bool):
                if value.lower() in ["true", "1", "yes", "y"]:
                    value = True
                elif value.lower() in ["false", "0", "no", "n"]:
                    value = False
                else:
                    return
            elif isinstance(current_value, int):
                try:
                    value = int(value)
                except ValueError:
                    return
            elif isinstance(current_value, float):
                try:
                    value = float(value)
                except ValueError:
                    return
            elif isinstance(current_value, list):
                try:
                    value = value.split(",")
                except:
                    return
            setattr(obj, attr, value)
    
    # Apply all environment variables that start with OWL_
    prefix = "OWL_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to lowercase
            config_key = key[len(prefix):].lower()
            # Replace _ with __ for nested attributes
            config_key = config_key.replace("_", "__")
            set_nested_attr(config, config_key, value)
    
    # Apply special case for environment variables
    if "ENVIRONMENT" in os.environ:
        config.environment = os.environ["ENVIRONMENT"]
    
    if "DEBUG" in os.environ:
        debug_value = os.environ["DEBUG"].lower()
        if debug_value in ["true", "1", "yes", "y"]:
            config.debug = True
        elif debug_value in ["false", "0", "no", "n"]:
            config.debug = False