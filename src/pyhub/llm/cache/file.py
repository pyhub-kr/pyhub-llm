"""File-based cache implementation"""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from pyhub.llm.cache.base import BaseCache


class FileCache(BaseCache):
    """File-based cache implementation with TTL support"""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize file cache

        Args:
            cache_dir: Directory to store cache files. Defaults to .cache/pyhub-llm
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), ".cache", "pyhub-llm")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_file(self, key: str) -> Path:
        """Get the cache file path for a key"""
        # Use a simple file naming scheme
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            return default

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Check if expired
            if "expiry" in data and data["expiry"] is not None:
                if time.time() > data["expiry"]:
                    # Remove expired file
                    cache_file.unlink()
                    return default

            return data.get("value", default)
        except (json.JSONDecodeError, IOError):
            # Remove corrupted file
            cache_file.unlink(missing_ok=True)
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL in seconds"""
        cache_file = self._get_cache_file(key)

        data = {"value": value, "expiry": time.time() + ttl if ttl else None}

        try:
            from pyhub.llm.json import JSONEncoder

            with open(cache_file, "w") as f:
                json.dump(data, f, cls=JSONEncoder)
        except (IOError, TypeError):
            # If we can't serialize or write, just ignore
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
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink(missing_ok=True)
