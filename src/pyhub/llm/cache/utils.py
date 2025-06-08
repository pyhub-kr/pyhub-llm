"""Cache utility functions"""

from typing import Any, Optional, Callable
from pyhub.llm.cache.base import BaseCache


__all__ = [
    'cache_make_key_and_get',
    'cache_make_key_and_get_async', 
    'cache_set',
    'cache_set_async',
]


def cache_make_key_and_get(
    key_prefix: str,
    key_parts: dict,
    cache_alias: str = "default",
    enable_cache: bool = False,
    default: Any = None
) -> tuple[str, Any]:
    """Generate cache key and get value from cache"""
    cache_key = generate_cache_key(key_prefix, **key_parts)
    
    # If caching is not enabled, return None
    if not enable_cache:
        return cache_key, None
    
    # TODO: Implement actual cache lookup based on cache_alias
    # For now, just return None (no cached value)
    return cache_key, None


async def cache_make_key_and_get_async(
    key_prefix: str,
    key_parts: dict,
    cache_alias: str = "default",
    enable_cache: bool = False,
    default: Any = None
) -> tuple[str, Any]:
    """Generate cache key and get value from cache (async version)"""
    cache_key = generate_cache_key(key_prefix, **key_parts)
    
    # If caching is not enabled, return None
    if not enable_cache:
        return cache_key, None
    
    # TODO: Implement actual cache lookup based on cache_alias
    # For now, just return None (no cached value)
    return cache_key, None


def cache_set(
    key: str,
    value: Any,
    cache_alias: str = "default",
    enable_cache: bool = False,
    ttl: Optional[int] = None
) -> None:
    """Set value in cache"""
    # If caching is not enabled, do nothing
    if not enable_cache:
        return
    
    # TODO: Implement actual cache set based on cache_alias
    # For now, just do nothing
    pass


async def cache_set_async(
    key: str,
    value: Any,
    cache_alias: str = "default",
    enable_cache: bool = False,
    ttl: Optional[int] = None
) -> None:
    """Set value in cache (async version)"""
    # If caching is not enabled, do nothing
    if not enable_cache:
        return
    
    # TODO: Implement actual cache set based on cache_alias
    # For now, just do nothing
    pass


def generate_cache_key(prefix: str, **kwargs) -> str:
    """Generate a cache key from prefix and keyword arguments"""
    import hashlib
    import json
    from pyhub.llm.json import JSONEncoder
    
    # Sort kwargs for consistent key generation
    sorted_data = json.dumps(kwargs, sort_keys=True, cls=JSONEncoder)
    hash_suffix = hashlib.md5(sorted_data.encode()).hexdigest()[:8]
    
    return f"{prefix}:{hash_suffix}"