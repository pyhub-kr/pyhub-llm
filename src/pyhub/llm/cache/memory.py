"""In-memory cache implementation."""

import time
from typing import Any, Dict, Optional, Tuple

from .base import BaseCache


class MemoryCache(BaseCache):
    """Simple in-memory cache implementation."""
    
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, Optional[float]]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        
        # Check expiry
        if expiry is not None and time.time() > expiry:
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> None:
        """Set a value in cache."""
        expiry = None
        if timeout is not None:
            expiry = time.time() + timeout
        
        self._cache[key] = (value, expiry)
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        # Check with expiry
        value = self.get(key)
        return value is not None
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self._cache.items()
            if expiry is not None and current_time > expiry
        ]
        for key in expired_keys:
            del self._cache[key]