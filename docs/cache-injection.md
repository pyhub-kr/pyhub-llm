# Cache Injection in pyhub-llm

## Overview

pyhub-llm implements a flexible caching system that allows users to inject their own cache backends at the LLM instance level. This approach provides maximum control over caching behavior without requiring complex configuration files.

## Cache Key Generation

### Algorithm
- **Hash Function**: SHA-256 (cryptographically secure)
- **Key Length**: 16 characters from the hexadecimal digest
- **Format**: `{namespace}:{hash_suffix}`
- **Example**: `openai:a1b2c3d4e5f6g7h8`

### Why 16 Characters?

The 16-character hash provides an excellent balance between collision resistance and practical key length:

- **Collision Probability**: ~1 in 10^19 for 16 hex characters
- **Storage Efficiency**: Shorter keys are more efficient for cache storage
- **Readability**: Keys remain manageable for debugging and logging

For most applications handling millions of cached responses, the probability of collision remains negligible.

## Cache Injection Pattern

### Basic Usage

```python
from pyhub.llm import LLM
from pyhub.llm.cache import MemoryCache, FileCache

# Create cache instance
cache = MemoryCache()

# Inject cache at LLM creation
llm = LLM.create("gpt-4o", cache=cache)

# Use with caching enabled
response = llm.ask("Hello", enable_cache=True)
```

### Multiple LLMs with Different Caches

```python
# Memory cache for fast, temporary storage
memory_cache = MemoryCache()
llm1 = LLM.create("gpt-4o", cache=memory_cache)

# File cache for persistent storage
file_cache = FileCache("/path/to/cache")
llm2 = LLM.create("claude-3-5-sonnet-latest", cache=file_cache)
```

### Shared Cache Between LLMs

```python
# Share cache across multiple LLMs
shared_cache = MemoryCache()
llm1 = LLM.create("gpt-4o", cache=shared_cache)
llm2 = LLM.create("gpt-4o-mini", cache=shared_cache)
```

## Custom Cache Implementation

You can implement your own cache backend by inheriting from `BaseCache`:

```python
from pyhub.llm.cache.base import BaseCache

class RedisCache(BaseCache):
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def get(self, key: str, default=None):
        value = self.redis.get(key)
        return value if value is not None else default
    
    def set(self, key: str, value, ttl=None):
        self.redis.set(key, value, ex=ttl)
    
    def delete(self, key: str):
        self.redis.delete(key)
    
    def clear(self):
        # Implement based on your needs
        pass

# Use custom cache
redis_cache = RedisCache(redis_client)
llm = LLM.create("gpt-4o", cache=redis_cache)
```

## Cache Behavior

### What Gets Cached
- LLM text responses
- Token usage information (set to 0 for cached responses)
- Embedding results

### What Doesn't Get Cached
- Streaming responses (planned for future)
- Tool/function calling intermediate results

### Cache Key Components
The cache key includes all parameters that affect the response:
- Model name
- Messages/prompt
- Temperature
- Max tokens
- System prompt
- Any other model-specific parameters

## Best Practices

1. **Explicit Caching**: Always explicitly enable caching with `enable_cache=True` for each call
2. **Cache Isolation**: Use different cache instances for different use cases
3. **TTL Management**: Implement TTL in your cache backend for automatic expiration
4. **Error Handling**: Cache errors should not affect LLM functionality

## Migration from Settings-Based Cache

If you were using the old settings-based cache configuration:

```python
# Old approach (no longer supported)
llm_settings.use_default_cache = True
llm_settings.cache_backend = "file"

# New approach
from pyhub.llm.cache import FileCache
cache = FileCache()
llm = LLM.create("gpt-4o", cache=cache)
```

This new approach provides better control and flexibility without the complexity of global settings.