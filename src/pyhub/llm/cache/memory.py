"""In-memory cache implementation"""

import logging
import time
from typing import Any, Optional

from pyhub.llm.cache.base import BaseCache

logger = logging.getLogger(__name__)


class MemoryCache(BaseCache):
    """Simple in-memory cache with TTL support"""

    def __init__(self, ttl: Optional[int] = None, debug: bool = False):
        """
        Initialize MemoryCache.

        Args:
            ttl: Default TTL in seconds. None means no expiry by default.
            debug: Enable debug logging for cache operations.

        Raises:
            ValueError: If TTL is negative.
        """
        if ttl is not None and ttl < 0:
            raise ValueError("TTL must be non-negative")

        self._cache = {}
        self._expiry = {}
        self._default_ttl = ttl
        self._debug = debug
        self._stats = {"hits": 0, "misses": 0, "sets": 0}

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        # Check if key exists and hasn't expired
        if key in self._cache:
            expiry = self._expiry.get(key)
            if expiry is None or time.time() < expiry:
                if self._debug:
                    logger.debug(f"Cache HIT: {key}")
                self._stats["hits"] += 1
                return self._cache[key]
            else:
                # Remove expired entry
                del self._cache[key]
                del self._expiry[key]
                if self._debug:
                    logger.debug(f"Cache MISS (expired): {key}")
                self._stats["misses"] += 1
        else:
            if self._debug:
                logger.debug(f"Cache MISS: {key}")
            self._stats["misses"] += 1
        return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL in seconds.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds. If None, uses default TTL. If 0, no expiry.

        Raises:
            ValueError: If TTL is negative.
        """
        if ttl is not None and ttl < 0:
            raise ValueError("TTL must be non-negative")

        self._cache[key] = value
        self._stats["sets"] += 1
        
        if self._debug:
            logger.debug(f"Cache SET: {key}")

        # Determine effective TTL
        effective_ttl = ttl if ttl is not None else self._default_ttl

        if effective_ttl:
            self._expiry[key] = time.time() + effective_ttl
            if self._debug:
                logger.debug(f"  TTL: {effective_ttl} seconds")
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
        if self._debug:
            logger.debug("Cache CLEARED")
    
    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "size": len(self._cache)
        }
