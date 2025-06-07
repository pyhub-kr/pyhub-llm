"""Base cache interface for PyHub LLM."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> None:
        """Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            timeout: Optional timeout in seconds
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was deleted, False if not found
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key exists
        """
        pass
    
    def get_or_set(self, key: str, default_factory: callable, timeout: Optional[int] = None) -> Any:
        """Get a value from cache or set it using a factory function.
        
        Args:
            key: Cache key
            default_factory: Function to call if key not found
            timeout: Optional timeout in seconds
            
        Returns:
            Cached or newly created value
        """
        value = self.get(key)
        if value is None:
            value = default_factory()
            self.set(key, value, timeout)
        return value