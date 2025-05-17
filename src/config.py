"""
Configuration module for the Mistral Docs Assistant.
"""
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base default configuration
DEFAULT_CONFIG = {
    # Document processing
    "chunk_size": 1000,
    "chunk_overlap": 20,
    
    # Mistral AI settings
    "model_name": "mistral-medium",
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
    # Use a dedicated config directory in the user's home directory
    config_dir = Path.home() / ".mistral_docs_assistant"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "config.json"

def load_config() -> Dict[str, Any]:
    """Load configuration from file or return default."""
    config_path = get_config_path()
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
            
            # Merge with defaults to ensure all keys exist
            config = DEFAULT_CONFIG.copy()
            config.update(user_config)
            return config
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return DEFAULT_CONFIG
    else:
        return DEFAULT_CONFIG

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file."""
    config_path = get_config_path()
    
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def get_api_key() -> Optional[str]:
    """Get Mistral API key from environment or config."""
    # First try environment variable
    api_key = os.getenv("MISTRAL_API_KEY")
    
    # If not in environment, try config file
    if not api_key:
        config = load_config()
        api_key = config.get("mistral_api_key")
    
    return api_key

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration values."""
    validated = DEFAULT_CONFIG.copy()
    
    # Validate and sanitize values
    if "chunk_size" in config:
        chunk_size = config["chunk_size"]
        validated["chunk_size"] = max(100, min(5000, int(chunk_size)))
    
    if "chunk_overlap" in config:
        chunk_overlap = config["chunk_overlap"]
        validated["chunk_overlap"] = max(0, min(500, int(chunk_overlap)))
    
    if "model_name" in config:
        model_name = config["model_name"]
        valid_models = ["mistral-small", "mistral-medium"]
        validated["model_name"] = model_name if model_name in valid_models else DEFAULT_CONFIG["model_name"]
    
    if "temperature" in config:
        temperature = config["temperature"]
        validated["temperature"] = max(0.0, min(1.0, float(temperature)))
    
    if "max_tokens" in config:
        max_tokens = config["max_tokens"]
        validated["max_tokens"] = max(1, min(4096, int(max_tokens)))
    
    if "top_k" in config:
        top_k = config["top_k"]
        validated["top_k"] = max(1, min(20, int(top_k)))
    
    return validated
