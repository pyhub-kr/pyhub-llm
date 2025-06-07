"""File-based cache implementation."""

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

from .base import BaseCache


class FileCache(BaseCache):
    """File-based cache implementation."""
    
    def __init__(self, cache_dir: str = "~/.pyhub/cache"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a key."""
        # Use SHA256 to ensure valid filenames
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_meta_path(self, key: str) -> Path:
        """Get the metadata file path for a key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        if not cache_path.exists():
            return None
        
        # Check metadata for expiry
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                if 'expires_at' in meta and meta['expires_at'] is not None:
                    if time.time() > meta['expires_at']:
                        # Expired
                        cache_path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                        return None
            except:
                # If metadata is corrupted, ignore it
                pass
        
        # Load cached value
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            # If cache file is corrupted, delete it
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None
    
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> None:
        """Set a value in cache."""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        # Save value
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f)
        
        # Save metadata
        meta = {
            'created_at': time.time(),
            'expires_at': time.time() + timeout if timeout else None
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        existed = cache_path.exists()
        cache_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        
        return existed
    
    def clear(self) -> None:
        """Clear all cached values."""
        for file in self.cache_dir.glob("*.cache"):
            file.unlink()
        for file in self.cache_dir.glob("*.meta"):
            file.unlink()
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        # Use get to check with expiry
        return self.get(key) is not None