"""
Model provider module for Mistral Docs Assistant.
This module handles different LLM providers and manages switching between them.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Union
import time
from functools import lru_cache
from abc import ABC, abstractmethod

# LangChain imports
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings

# Provider-specific imports
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings

# Optional imports - will be used if available
try:
    from langchain_openai import ChatOpenAI
    from langchain_openai.embeddings import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_community.llms import HuggingFaceHub
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from langchain_community.llms import LlamaCpp
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry of available models and their configurations."""
    
    # Default configurations for different models
    DEFAULT_CONFIGS = {
        # Mistral models
        "mistral-small": {
            "provider": "mistral",
            "embedding_model": "mistral-embed",
            "temperature": 0.5,
            "max_tokens": 1024,
        },
        "mistral-medium": {
            "provider": "mistral",
            "embedding_model": "mistral-embed",
            "temperature": 0.5,
            "max_tokens": 2048,
        },
        "mistral-large": {
            "provider": "mistral",
            "embedding_model": "mistral-embed",
            "temperature": 0.5,
            "max_tokens": 4096,
        },
        
        # OpenAI models
        "gpt-3.5-turbo": {
            "provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "gpt-4": {
            "provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        
        # Hugging Face models
        "huggingface/mistralai/Mistral-7B-Instruct-v0.2": {
            "provider": "huggingface",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        
        # Local models
        "local/llama-2-7b": {
            "provider": "local",
            "model_path": "models/llama-2-7b-chat.gguf",
            "embedding_model": "local/all-MiniLM-L6-v2",
            "temperature": 0.7,
            "max_tokens": 1024,
            "context_window": 4096,
            "n_gpu_layers": 0,  # GPU acceleration if > 0
            "n_batch": 512,
        },
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_name in cls.DEFAULT_CONFIGS:
            return cls.DEFAULT_CONFIGS[model_name].copy()
        else:
            raise ValueError(f"Model {model_name} not found in registry")
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of all available models based on installed packages."""
        available_models = []
        
        # Always include Mistral models since they're required
        available_models.extend([
            "mistral-small", 
            "mistral-medium", 
            "mistral-large"
        ])
        
        # Add OpenAI models if available
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            available_models.extend([
                "gpt-3.5-turbo",
                "gpt-4"
            ])
        
        # Add HuggingFace models if available
        if HUGGINGFACE_AVAILABLE and os.getenv("HUGGINGFACE_API_KEY"):
            available_models.append("huggingface/mistralai/Mistral-7B-Instruct-v0.2")
        
        # Add local models if available
        if LOCAL_LLM_AVAILABLE and os.path.exists("models/llama-2-7b-chat.gguf"):
            available_models.append("local/llama-2-7b")
        
        return available_models
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers based on installed packages."""
        providers = ["mistral"]  # Always available
        
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            providers.append("openai")
            
        if HUGGINGFACE_AVAILABLE and os.getenv("HUGGINGFACE_API_KEY"):
            providers.append("huggingface")
            
        if LOCAL_LLM_AVAILABLE:
            providers.append("local")
            
        return providers


class BaseLLMProvider(ABC):
    """Base interface for LLM providers."""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.name = "base"
        
    @abstractmethod
    def get_llm(self) -> LLM:
        """Return a LangChain compatible LLM."""
        pass
        
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Return a LangChain compatible embeddings model."""
        pass
    
    @property
    def available(self) -> bool:
        """Check if this provider is available for use."""
        return True


class MistralProvider(BaseLLMProvider):
    """Mistral AI provider implementation."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.name = "mistral"
        self.api_key = os.getenv("MISTRAL_API_KEY")
        
        if not self.api_key:
            raise ValueError("Missing MISTRAL_API_KEY environment variable")
    
    def get_llm(self) -> ChatMistralAI:
        """Return a Mistral LLM instance."""
        model_name = self.model_config.get("model", "mistral-medium")
        temperature = self.model_config.get("temperature", 0.5)
        
        return ChatMistralAI(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature
        )
    
    def get_embeddings(self) -> MistralAIEmbeddings:
        """Return Mistral embeddings model."""
        return MistralAIEmbeddings(api_key=self.api_key)
    
    @property
    def available(self) -> bool:
        """Check if Mistral API is available and not rate limited."""
        return bool(self.api_key)  # Basic check - could add rate limit detection


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.name = "openai"
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI packages not installed. Install with: pip install langchain-openai")
    
    def get_llm(self) -> ChatOpenAI:
        """Return an OpenAI LLM instance."""
        model_name = self.model_config.get("model", "gpt-3.5-turbo")
        temperature = self.model_config.get("temperature", 0.7)
        max_tokens = self.model_config.get("max_tokens", 2048)
        
        return ChatOpenAI(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def get_embeddings(self) -> OpenAIEmbeddings:
        """Return OpenAI embeddings model."""
        embedding_model = self.model_config.get("embedding_model", "text-embedding-3-small")
        
        return OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.api_key
        )
    
    @property
    def available(self) -> bool:
        """Check if OpenAI API is available."""
        return bool(self.api_key) and OPENAI_AVAILABLE


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Hub provider implementation."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.name = "huggingface"
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Missing HUGGINGFACE_API_KEY environment variable")
        
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("HuggingFace packages not installed. Install with: pip install langchain-community")
    
    def get_llm(self) -> HuggingFaceHub:
        """Return a HuggingFace LLM instance."""
        model_name = self.model_config.get("model", "mistralai/Mistral-7B-Instruct-v0.2")
        temperature = self.model_config.get("temperature", 0.7)
        max_tokens = self.model_config.get("max_tokens", 1024)
        
        # Extract repo_id from full model name if needed
        if "/" in model_name and model_name.startswith("huggingface/"):
            repo_id = model_name[len("huggingface/"):]
        else:
            repo_id = model_name
        
        return HuggingFaceHub(
            repo_id=repo_id,
            huggingfacehub_api_token=self.api_key,
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": max_tokens
            }
        )
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Return HuggingFace embeddings model."""
        embedding_model = self.model_config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
        
        return HuggingFaceEmbeddings(
            model_name=embedding_model,
            huggingfacehub_api_token=self.api_key
        )
    
    @property
    def available(self) -> bool:
        """Check if HuggingFace API is available."""
        return bool(self.api_key) and HUGGINGFACE_AVAILABLE


class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider implementation using llama.cpp."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.name = "local"
        
        if not LOCAL_LLM_AVAILABLE:
            raise ImportError("Local LLM packages not installed. Install with: pip install llama-cpp-python langchain-community")
        
        self.model_path = self.model_config.get("model_path")
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError(f"Model file not found at {self.model_path}")
    
    def get_llm(self) -> LlamaCpp:
        """Return a local LLM instance."""
        model_path = self.model_config.get("model_path")
        n_gpu_layers = self.model_config.get("n_gpu_layers", 0)
        n_batch = self.model_config.get("n_batch", 512)
        temperature = self.model_config.get("temperature", 0.7)
        max_tokens = self.model_config.get("max_tokens", 1024)
        context_window = self.model_config.get("context_window", 4096)
        
        return LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            n_ctx=context_window
        )
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Return local embeddings model."""
        embedding_model = self.model_config.get("embedding_model", "local/all-MiniLM-L6-v2")
        
        if embedding_model.startswith("local/"):
            # Use a local embedding model
            model_name = embedding_model[len("local/"):]
            return HuggingFaceEmbeddings(model_name=model_name)
        else:
            # Fallback to a hosted model
            return HuggingFaceEmbeddings(model_name=embedding_model)
    
    @property
    def available(self) -> bool:
        """Check if local model is available."""
        return LOCAL_LLM_AVAILABLE and os.path.exists(self.model_config.get("model_path", ""))


class ModelManager:
    """Manager for handling multiple LLM providers and caching."""
    
    _PROVIDER_MAP = {
        "mistral": MistralProvider,
        "openai": OpenAIProvider,
        "huggingface": HuggingFaceProvider,
        "local": LocalLLMProvider
    }
    
    def __init__(self):
        self.fallback_order = ["mistral", "openai", "huggingface", "local"]
        self.cache_size = 100  # Default cache size
        
    def get_provider(self, model_name: str) -> BaseLLMProvider:
        """Get the provider instance for the specified model."""
        try:
            model_config = ModelRegistry.get_model_config(model_name)
            provider_name = model_config.get("provider", "mistral")
            
            if provider_name not in self._PROVIDER_MAP:
                raise ValueError(f"Unknown provider: {provider_name}")
            
            provider_class = self._PROVIDER_MAP[provider_name]
            return provider_class(model_config)
        
        except Exception as e:
            logger.error(f"Error creating provider for model {model_name}: {str(e)}")
            raise
    
    def get_default_model(self) -> str:
        """Get the default model based on available providers."""
        available_models = ModelRegistry.get_available_models()
        
        # Default preference order
        preferred_models = [
            "mistral-medium",  # First choice
            "mistral-small",   # Second choice
            "gpt-3.5-turbo",   # Third choice
            "local/llama-2-7b"  # Fourth choice
        ]
        
        for model in preferred_models:
            if model in available_models:
                return model
        
        # Fallback to first available model
        return available_models[0] if available_models else "mistral-small"
    
    def set_fallback_order(self, order: List[str]) -> None:
        """Set the order of provider fallbacks."""
        available_providers = ModelRegistry.get_available_providers()
        
        # Validate the provided order
        for provider in order:
            if provider not in self._PROVIDER_MAP:
                raise ValueError(f"Unknown provider: {provider}")
        
        # Filter out unavailable providers
        valid_order = [p for p in order if p in available_providers]
        
        # Add any available providers that weren't in the list
        for provider in available_providers:
            if provider not in valid_order:
                valid_order.append(provider)
        
        self.fallback_order = valid_order
    
    def get_llm_with_fallback(self, primary_model: str) -> Tuple[LLM, str]:
        """Get an LLM instance, falling back to alternatives if needed."""
        tried_models = []
        
        # Try the primary model first
        try:
            provider = self.get_provider(primary_model)
            if provider.available:
                return provider.get_llm(), primary_model
            tried_models.append(primary_model)
        except Exception as e:
            logger.warning(f"Failed to use primary model {primary_model}: {str(e)}")
            tried_models.append(primary_model)
        
        # If primary model fails, try alternatives by provider
        primary_config = ModelRegistry.get_model_config(primary_model)
        primary_provider = primary_config.get("provider")
        
        # Determine fallback order starting after the primary provider
        if primary_provider in self.fallback_order:
            idx = self.fallback_order.index(primary_provider)
            ordered_fallbacks = self.fallback_order[idx+1:] + self.fallback_order[:idx]
        else:
            ordered_fallbacks = self.fallback_order
        
        # Try each fallback provider
        for provider_name in ordered_fallbacks:
            for model_name in ModelRegistry.get_available_models():
                if model_name in tried_models:
                    continue
                    
                model_config = ModelRegistry.get_model_config(model_name)
                if model_config.get("provider") != provider_name:
                    continue
                
                try:
                    provider = self.get_provider(model_name)
                    if provider.available:
                        return provider.get_llm(), model_name
                except Exception as e:
                    logger.warning(f"Fallback to {model_name} failed: {str(e)}")
                
                tried_models.append(model_name)
        
        # If all fallbacks fail, raise error
        raise RuntimeError("All LLM providers failed or are unavailable")

    @lru_cache(maxsize=100)
    def get_cached_response(self, model_name: str, input_text: str) -> Optional[str]:
        """Get a cached response if available (using lru_cache decorator)."""
        # This method is intentionally empty as the caching is handled by the decorator
        # This is just a placeholder to make the caching mechanism explicit
        return None
    
    def set_cache_size(self, size: int) -> None:
        """Dynamically adjust the LRU cache size."""
        if size < 0:
            raise ValueError("Cache size cannot be negative")
        
        # Create a new cached function with updated cache size
        self.get_cached_response = lru_cache(maxsize=size)(self.get_cached_response.__wrapped__)
        self.cache_size = size


# Create a singleton instance of the model manager
model_manager = ModelManager()


def get_model_list() -> List[Dict[str, str]]:
    """Get a list of available models with metadata for the UI."""
    models = []
    available_models = ModelRegistry.get_available_models()
    
    for model_name in available_models:
        try:
            config = ModelRegistry.get_model_config(model_name)
            provider = config.get("provider", "unknown")
            
            models.append({
                "name": model_name,
                "provider": provider,
                "display_name": model_name.split("/")[-1] if "/" in model_name else model_name,
                "is_local": provider == "local"
            })
        except:
            pass  # Skip models with errors
    
    return models
