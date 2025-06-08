"""In-memory cache implementation"""

import time
from typing import Any, Optional

from pyhub.llm.cache.base import BaseCache


class MemoryCache(BaseCache):
    """Simple in-memory cache with TTL support"""

    def __init__(self):
        self._cache = {}
        self._expiry = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        # Check if key exists and hasn't expired
        if key in self._cache:
            expiry = self._expiry.get(key)
            if expiry is None or time.time() < expiry:
                return self._cache[key]
            else:
                # Remove expired entry
                del self._cache[key]
                del self._expiry[key]
        return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL in seconds"""
        self._cache[key] = value
        if ttl:
            self._expiry[key] = time.time() + ttl
        elif key in self._expiry:
            del self._expiry[key]

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if key in self._cache:
            del self._cache[key]
            if key in self._expiry:
                del self._expiry[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all values from cache"""
        self._cache.clear()
        self._expiry.clear()
