"""
Configuration module for the Mistral Docs Assistant.
"""
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelName(str, Enum):
    """Enum for supported Mistral model names."""
    SMALL = "mistral-small"
    MEDIUM = "mistral-medium"

# Base default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    # Document processing
    "chunk_size": 1000,
    "chunk_overlap": 20,
    
    # Mistral AI settings
    "model_name": ModelName.MEDIUM.value,
    "temperature": 0.5,
    "max_tokens": 1024,
    
    # Retrieval settings
    "top_k": 5,
    "search_type": "similarity",
    
    # Application settings
    "debug_mode": False,
    "log_level": "INFO",
}

def get_config_path() -> Path:
    """Return the path to the configuration file."""
    config_dir: Path = Path.home() / ".mistral_docs_assistant"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "config.json"

def load_config() -> Dict[str, Any]:
    """Load configuration from file or return default."""
    config_path: Path = get_config_path()
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                user_config: Dict[str, Any] = json.load(f)
            
            # Merge with defaults to ensure all keys exist
            config: Dict[str, Any] = DEFAULT_CONFIG.copy()
            config.update(user_config)
            return validate_config(config)
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return DEFAULT_CONFIG
    else:
        return DEFAULT_CONFIG

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file."""
    config_path: Path = get_config_path()
    
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def get_api_key() -> Optional[str]:
    """Get Mistral API key from environment or config."""
    api_key: Optional[str] = os.getenv("MISTRAL_API_KEY")
    
    if not api_key:
        config: Dict[str, Any] = load_config()
        api_key = config.get("mistral_api_key")
    
    return api_key

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration values."""
    validated: Dict[str, Any] = DEFAULT_CONFIG.copy()
    
    # Validate and sanitize values
    if "chunk_size" in config:
        chunk_size: int = int(config["chunk_size"])
        validated["chunk_size"] = max(100, min(5000, chunk_size))
        if chunk_size > config.get("max_tokens", validated["max_tokens"]):
            raise ValueError("chunk_size cannot exceed max_tokens")
    
    if "chunk_overlap" in config:
        chunk_overlap: int = int(config["chunk_overlap"])
        validated["chunk_overlap"] = max(0, min(500, chunk_overlap))
    
    if "model_name" in config:
        model_name: str = config["model_name"]
        valid_models: List[str] = [ModelName.SMALL.value, ModelName.MEDIUM.value]
        validated["model_name"] = model_name if model_name in valid_models else DEFAULT_CONFIG["model_name"]
    
    if "temperature" in config:
        temperature: float = float(config["temperature"])
        validated["temperature"] = max(0.0, min(1.0, temperature))
    
    if "max_tokens" in config:
        max_tokens: int = int(config["max_tokens"])
        validated["max_tokens"] = max(1, min(4096, max_tokens))
    
    if "top_k" in config:
        top_k: int = int(config["top_k"])
        validated["top_k"] = max(1, min(20, top_k))
    
    return validated