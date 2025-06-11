"""File-based cache implementation"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from pyhub.llm.cache.base import BaseCache

logger = logging.getLogger(__name__)


class FileCache(BaseCache):
    """File-based cache implementation with TTL support"""

    def __init__(self, cache_dir: Optional[str] = None, ttl: Optional[int] = None, debug: bool = False):
        """Initialize file cache

        Args:
            cache_dir: Directory to store cache files. Defaults to .cache/pyhub-llm
            ttl: Default TTL in seconds. None means no expiry by default.
            debug: Enable debug logging for cache operations.

        Raises:
            ValueError: If TTL is negative.
        """
        if ttl is not None and ttl < 0:
            raise ValueError("TTL must be non-negative")

        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), ".cache", "pyhub-llm")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._default_ttl = ttl
        self._debug = debug
        self._stats = {"hits": 0, "misses": 0, "sets": 0}

    def _get_cache_file(self, key: str) -> Path:
        """Get the cache file path for a key"""
        # Use a simple file naming scheme
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            if self._debug:
                logger.debug(f"Cache MISS: {key} (file not found)")
            self._stats["misses"] += 1
            return default

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Check if expired
            if "expiry" in data and data["expiry"] is not None:
                if time.time() > data["expiry"]:
                    # Remove expired file
                    cache_file.unlink()
                    if self._debug:
                        logger.debug(f"Cache MISS: {key} (expired)")
                    self._stats["misses"] += 1
                    return default

            if self._debug:
                logger.debug(f"Cache HIT: {key}")
            self._stats["hits"] += 1
            return data.get("value", default)
        except (json.JSONDecodeError, IOError):
            # Remove corrupted file
            cache_file.unlink(missing_ok=True)
            if self._debug:
                logger.debug(f"Cache MISS: {key} (corrupted file)")
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

        cache_file = self._get_cache_file(key)

        # Determine effective TTL
        effective_ttl = ttl if ttl is not None else self._default_ttl

        data = {"value": value, "expiry": time.time() + effective_ttl if effective_ttl else None}

        try:
            from pyhub.llm.json import JSONEncoder

            with open(cache_file, "w") as f:
                json.dump(data, f, cls=JSONEncoder)
            
            self._stats["sets"] += 1
            if self._debug:
                logger.debug(f"Cache SET: {key}")
                if effective_ttl:
                    logger.debug(f"  TTL: {effective_ttl} seconds")
        except (IOError, TypeError) as e:
            # If we can't serialize or write, just ignore
            if self._debug:
                logger.debug(f"Cache SET failed: {key} - {str(e)}")
            pass

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            cache_file.unlink()
            return True
        return False

    def clear(self) -> None:
        """Clear all cache files"""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink(missing_ok=True)
            count += 1
        if self._debug:
            logger.debug(f"Cache CLEARED: {count} files removed")
    
    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        # Count cache files
        cache_files = list(self.cache_dir.glob("*.json"))
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "size": len(cache_files)
        }
