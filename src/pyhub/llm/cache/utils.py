"""Cache utilities for PyHub LLM."""

import hashlib
import json
from typing import Any, Dict


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from arguments.
    
    Args:
        prefix: Key prefix
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Generated cache key
    """
    # Create a dictionary with all arguments
    key_data = {
        'prefix': prefix,
        'args': args,
        'kwargs': kwargs
    }
    
    # Convert to JSON for consistent serialization
    key_json = json.dumps(key_data, sort_keys=True, default=str)
    
    # Generate hash
    key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]
    
    return f"{prefix}:{key_hash}"