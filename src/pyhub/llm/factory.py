"""LLM factory for creating LLM instances."""

import logging
from typing import Any, Dict, Optional, Type

from .base import BaseLLM
from .cache import BaseCache, FileCache
from .exceptions import ProviderNotFoundError
from .settings import Settings
from .types import LLMModelType

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances."""
    
    _providers: Dict[str, Type[BaseLLM]] = {}
    _settings: Optional[Settings] = None
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLM]) -> None:
        """Register a provider class.
        
        Args:
            name: Provider name (e.g., 'openai', 'anthropic')
            provider_class: Provider class
        """
        cls._providers[name.lower()] = provider_class
        logger.debug(f"Registered provider: {name}")
    
    @classmethod
    def get_provider(cls, name: str) -> Type[BaseLLM]:
        """Get a provider class by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider class
            
        Raises:
            ProviderNotFoundError: If provider not found
        """
        provider = cls._providers.get(name.lower())
        if provider is None:
            available = ", ".join(cls._providers.keys())
            raise ProviderNotFoundError(f"Provider '{name}' not found. Available: {available}")
        return provider
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List available providers.
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def get_settings(cls) -> Settings:
        """Get or create settings instance."""
        if cls._settings is None:
            cls._settings = Settings()
        return cls._settings
    
    @classmethod
    def create(
        cls,
        model: LLMModelType,
        api_key: Optional[str] = None,
        cache: Optional[BaseCache] = None,
        use_cache: bool = True,
        **kwargs
    ) -> BaseLLM:
        """Create an LLM instance.
        
        Args:
            model: Model name
            api_key: API key (optional, can be loaded from settings)
            cache: Cache instance (optional)
            use_cache: Whether to use caching
            **kwargs: Additional arguments for the LLM
            
        Returns:
            LLM instance
            
        Raises:
            ProviderNotFoundError: If provider cannot be determined
        """
        # Auto-detect provider from model name
        provider_name = cls._detect_provider(model)
        
        # Get provider class
        provider_class = cls.get_provider(provider_name)
        
        # Get settings
        settings = cls.get_settings()
        
        # Get API key from settings if not provided
        if api_key is None:
            api_key = settings.get_api_key(provider_name)
        
        # Setup cache if requested
        if use_cache and cache is None:
            cache_dir = settings.get("cache_dir", "~/.pyhub/cache")
            cache = FileCache(cache_dir)
        
        # Create instance
        return provider_class(
            model=model,
            api_key=api_key,
            cache=cache if use_cache else None,
            **kwargs
        )
    
    @classmethod
    def _detect_provider(cls, model: str) -> str:
        """Detect provider from model name.
        
        Args:
            model: Model name
            
        Returns:
            Provider name
            
        Raises:
            ProviderNotFoundError: If provider cannot be detected
        """
        model_lower = model.lower()
        
        # Google models (check specific model first)
        if model_lower == "text-embedding-004" or model_lower.startswith("gemini"):
            return "google"
        
        # OpenAI models
        elif any(model_lower.startswith(prefix) for prefix in ["gpt-", "o1", "text-embedding", "chatgpt"]):
            return "openai"
        
        # Anthropic models
        elif model_lower.startswith("claude"):
            return "anthropic"
        
        # Upstage models
        elif any(model_lower.startswith(prefix) for prefix in ["solar", "embedding-"]):
            return "upstage"
        
        # Ollama models (various)
        elif any(substr in model_lower for substr in ["llama", "mistral", "qwen", "gemma"]):
            return "ollama"
        
        else:
            raise ProviderNotFoundError(
                f"Cannot detect provider for model '{model}'. "
                "Please specify provider explicitly or use a known model name."
            )


# Register providers on import
def _register_builtin_providers():
    """Register built-in providers."""
    try:
        from .providers.openai import OpenAILLM
        LLMFactory.register_provider("openai", OpenAILLM)
    except ImportError:
        logger.debug("OpenAI provider not available")
    
    try:
        from .providers.anthropic import AnthropicLLM
        LLMFactory.register_provider("anthropic", AnthropicLLM)
    except ImportError:
        logger.debug("Anthropic provider not available")
    
    try:
        from .providers.google import GoogleLLM
        LLMFactory.register_provider("google", GoogleLLM)
    except ImportError:
        logger.debug("Google provider not available")
    
    try:
        from .providers.ollama import OllamaLLM
        LLMFactory.register_provider("ollama", OllamaLLM)
    except ImportError:
        logger.debug("Ollama provider not available")
    
    try:
        from .providers.upstage import UpstageLLM
        LLMFactory.register_provider("upstage", UpstageLLM)
    except ImportError:
        logger.debug("Upstage provider not available")


# Register on module import
_register_builtin_providers()


# Main factory class for public API
class LLM:
    """Main LLM factory class."""
    
    @staticmethod
    def create(model: LLMModelType, **kwargs) -> BaseLLM:
        """Create an LLM instance.
        
        Args:
            model: Model name
            **kwargs: Additional arguments
            
        Returns:
            LLM instance
        """
        return LLMFactory.create(model, **kwargs)
    
    @staticmethod
    def list_providers() -> list[str]:
        """List available providers."""
        return LLMFactory.list_providers()
    
    @staticmethod
    def register_provider(name: str, provider_class: Type[BaseLLM]) -> None:
        """Register a custom provider."""
        LLMFactory.register_provider(name, provider_class)