"""Base cache interface"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCache(ABC):
    """Abstract base class for cache implementations"""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all values from cache"""
        pass

    def get_or_set(self, key: str, callable_fn, ttl: Optional[int] = None) -> Any:
        """Get value from cache, or set it using callable if not found"""
        value = self.get(key)
        if value is None:
            value = callable_fn()
            self.set(key, value, ttl)
        return value

    async def get_async(self, key: str, default: Any = None) -> Any:
        """Async version of get (default implementation calls sync version)"""
        return self.get(key, default)

    async def set_async(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Async version of set (default implementation calls sync version)"""
        self.set(key, value, ttl)
