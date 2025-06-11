import json
import time

import pytest

from pyhub.llm.cache import BaseCache, FileCache
from pyhub.llm.types import Message


class TestMemoryCache:
    """Test MemoryCache implementation."""

    def test_set_and_get(self, memory_cache):
        """Test basic set and get operations."""
        memory_cache.set("key1", "value1")
        assert memory_cache.get("key1") == "value1"
    
    def test_default_ttl_in_constructor(self):
        """Test that MemoryCache accepts default TTL in constructor."""
        from pyhub.llm.cache import MemoryCache
        
        # Should be able to create cache with default TTL
        cache = MemoryCache(ttl=3600)
        assert cache._default_ttl == 3600
        
        # Should be able to create cache without TTL
        cache_no_ttl = MemoryCache()
        assert cache_no_ttl._default_ttl is None
    
    def test_default_ttl_applied_in_set(self):
        """Test that default TTL is applied when not specified in set()."""
        from pyhub.llm.cache import MemoryCache
        
        cache = MemoryCache(ttl=2)  # 2 second default TTL
        cache.set("key1", "value1")  # No TTL specified, should use default
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(2.1)
        
        # Should be expired due to default TTL
        assert cache.get("key1") is None
    
    def test_set_ttl_overrides_default(self):
        """Test that set() TTL overrides default TTL."""
        from pyhub.llm.cache import MemoryCache
        
        cache = MemoryCache(ttl=10)  # 10 second default TTL
        cache.set("key1", "value1", ttl=1)  # Override with 1 second TTL
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for 1 second TTL
        time.sleep(1.1)
        
        # Should be expired (not waiting for 10 second default)
        assert cache.get("key1") is None
    
    def test_set_ttl_zero_overrides_default(self):
        """Test that set() with ttl=0 overrides default TTL (no expiry)."""
        from pyhub.llm.cache import MemoryCache
        
        cache = MemoryCache(ttl=1)  # 1 second default TTL
        cache.set("key1", "value1", ttl=0)  # Override with no expiry
        
        # Wait longer than default TTL
        time.sleep(1.5)
        
        # Should still be available (no expiry)
        assert cache.get("key1") == "value1"
    
    def test_ttl_validation(self):
        """Test TTL value validation."""
        from pyhub.llm.cache import MemoryCache
        
        # Negative TTL should raise ValueError
        with pytest.raises(ValueError, match="TTL must be non-negative"):
            MemoryCache(ttl=-1)
        
        cache = MemoryCache()
        with pytest.raises(ValueError, match="TTL must be non-negative"):
            cache.set("key1", "value1", ttl=-1)

    def test_get_nonexistent_key(self, memory_cache):
        """Test getting a non-existent key."""
        assert memory_cache.get("nonexistent") is None

    def test_delete(self, memory_cache):
        """Test delete operation."""
        memory_cache.set("key1", "value1")
        assert memory_cache.delete("key1") is True
        assert memory_cache.get("key1") is None

        # Delete non-existent key
        assert memory_cache.delete("nonexistent") is False

    def test_clear(self, memory_cache):
        """Test clear operation."""
        memory_cache.set("key1", "value1")
        memory_cache.set("key2", "value2")

        memory_cache.clear()

        assert memory_cache.get("key1") is None
        assert memory_cache.get("key2") is None

    def test_ttl_expiration(self, memory_cache):
        """Test TTL expiration."""
        memory_cache.set("key1", "value1", ttl=1)  # 1 second TTL

        # Should be available immediately
        assert memory_cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert memory_cache.get("key1") is None

    def test_get_or_set(self, memory_cache):
        """Test get_or_set operation."""

        # Key doesn't exist, should call factory
        def factory():
            return "generated_value"

        result = memory_cache.get_or_set("key1", factory)
        assert result == "generated_value"
        assert memory_cache.get("key1") == "generated_value"

        # Key exists, should not call factory
        def factory2():
            raise Exception("Should not be called")

        result = memory_cache.get_or_set("key1", factory2)
        assert result == "generated_value"

    def test_complex_data_types(self, memory_cache):
        """Test caching complex data types."""
        # List
        memory_cache.set("list_key", [1, 2, 3])
        assert memory_cache.get("list_key") == [1, 2, 3]

        # Dict
        memory_cache.set("dict_key", {"a": 1, "b": 2})
        assert memory_cache.get("dict_key") == {"a": 1, "b": 2}

        # Custom object
        obj = Message(role="user", content="Hello")
        memory_cache.set("obj_key", obj)
        retrieved = memory_cache.get("obj_key")
        assert retrieved.role == "user"
        assert retrieved.content == "Hello"

    def test_none_value(self, memory_cache):
        """Test caching None value."""
        memory_cache.set("none_key", None)
        # Should return None (the cached value), not None (key not found)
        assert memory_cache.get("none_key") is None
        assert "none_key" in memory_cache._cache


class TestFileCache:
    """Test FileCache implementation."""

    def test_initialization(self, temp_cache_dir):
        """Test FileCache initialization."""
        cache = FileCache(str(temp_cache_dir))
        assert cache.cache_dir == temp_cache_dir
        assert cache.cache_dir.exists()
    
    def test_default_ttl_in_constructor(self, temp_cache_dir):
        """Test that FileCache accepts default TTL in constructor."""
        # Should be able to create cache with default TTL
        cache = FileCache(cache_dir=str(temp_cache_dir), ttl=3600)
        assert cache._default_ttl == 3600
        
        # Should be able to create cache without TTL
        cache_no_ttl = FileCache(cache_dir=str(temp_cache_dir))
        assert cache_no_ttl._default_ttl is None
    
    def test_default_ttl_applied_in_set(self, temp_cache_dir):
        """Test that default TTL is applied when not specified in set()."""
        cache = FileCache(cache_dir=str(temp_cache_dir), ttl=2)  # 2 second default TTL
        cache.set("key1", "value1")  # No TTL specified, should use default
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(2.1)
        
        # Should be expired due to default TTL
        assert cache.get("key1") is None
    
    def test_set_ttl_overrides_default(self, temp_cache_dir):
        """Test that set() TTL overrides default TTL."""
        cache = FileCache(cache_dir=str(temp_cache_dir), ttl=10)  # 10 second default TTL
        cache.set("key1", "value1", ttl=1)  # Override with 1 second TTL
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for 1 second TTL
        time.sleep(1.1)
        
        # Should be expired (not waiting for 10 second default)
        assert cache.get("key1") is None
    
    def test_ttl_validation(self, temp_cache_dir):
        """Test TTL value validation."""
        # Negative TTL should raise ValueError
        with pytest.raises(ValueError, match="TTL must be non-negative"):
            FileCache(cache_dir=str(temp_cache_dir), ttl=-1)
        
        cache = FileCache(cache_dir=str(temp_cache_dir))
        with pytest.raises(ValueError, match="TTL must be non-negative"):
            cache.set("key1", "value1", ttl=-1)

    def test_create_directory_if_not_exists(self, tmp_path):
        """Test cache directory creation."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        cache = FileCache(str(cache_dir))
        assert cache_dir.exists()

    def test_set_and_get(self, file_cache):
        """Test basic set and get operations."""
        file_cache.set("key1", "value1")
        assert file_cache.get("key1") == "value1"

    def test_get_nonexistent_key(self, file_cache):
        """Test getting a non-existent key."""
        assert file_cache.get("nonexistent") is None

    def test_delete(self, file_cache):
        """Test delete operation."""
        file_cache.set("key1", "value1")
        assert file_cache.delete("key1") is True
        assert file_cache.get("key1") is None

        # Delete non-existent key
        assert file_cache.delete("nonexistent") is False

    def test_clear(self, file_cache):
        """Test clear operation."""
        file_cache.set("key1", "value1")
        file_cache.set("key2", "value2")

        file_cache.clear()

        assert file_cache.get("key1") is None
        assert file_cache.get("key2") is None

        # Cache directory should still exist
        assert file_cache.cache_dir.exists()

    def test_ttl_expiration(self, file_cache):
        """Test TTL expiration."""
        file_cache.set("key1", "value1", ttl=1)  # 1 second TTL

        # Should be available immediately
        assert file_cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert file_cache.get("key1") is None

    def test_key_hashing(self, file_cache):
        """Test key sanitization for file names."""
        # Keys with special characters should be sanitized
        special_key = "cache:key/with/special"
        file_cache.set(special_key, "value")

        # Should create a sanitized filename with .json extension
        cache_files = list(file_cache.cache_dir.glob("*.json"))
        assert len(cache_files) == 1
        # FileCache replaces '/' and ':' with '_'
        assert cache_files[0].stem == "cache_key_with_special"

    def test_complex_data_types(self, file_cache):
        """Test caching complex data types."""
        # List
        file_cache.set("list_key", [1, 2, 3])
        assert file_cache.get("list_key") == [1, 2, 3]

        # Dict
        file_cache.set("dict_key", {"a": 1, "b": 2})
        assert file_cache.get("dict_key") == {"a": 1, "b": 2}

        # Custom object - Message objects are serialized as dicts
        obj = Message(role="user", content="Hello")
        file_cache.set("obj_key", obj)
        retrieved = file_cache.get("obj_key")
        assert isinstance(retrieved, dict)
        assert retrieved["role"] == "user"
        assert retrieved["content"] == "Hello"

    def test_pickle_vs_json_serialization(self, file_cache):
        """Test that appropriate serialization is used."""
        # JSON-serializable data
        json_data = {"key": "value", "number": 42}
        file_cache.set("json_key", json_data)

        # Non-JSON-serializable data
        obj = Message(role="user", content="Hello")
        file_cache.set("pickle_key", obj)

        # Both should be retrievable
        assert file_cache.get("json_key") == json_data
        retrieved_obj = file_cache.get("pickle_key")
        # Message objects are serialized as dicts
        assert isinstance(retrieved_obj, dict)
        assert retrieved_obj["role"] == "user"

    def test_corrupted_cache_file(self, file_cache):
        """Test handling of corrupted cache files."""
        file_cache.set("key1", "value1")

        # Corrupt the cache file
        cache_files = list(file_cache.cache_dir.glob("*.json"))
        assert len(cache_files) == 1

        with open(cache_files[0], "w") as f:
            f.write("corrupted data")

        # Should return None for corrupted file
        assert file_cache.get("key1") is None

    def test_metadata_file(self, file_cache):
        """Test that FileCache includes metadata in the JSON file."""
        file_cache.set("key1", "value1", ttl=3600)

        # FileCache stores metadata in the same JSON file, not separate .meta files
        cache_files = list(file_cache.cache_dir.glob("*.json"))
        assert len(cache_files) == 1

        # Check that the JSON file contains expiry metadata
        with open(cache_files[0], "r") as f:
            data = json.load(f)

        assert "value" in data
        assert "expiry" in data
        assert data["value"] == "value1"
        assert data["expiry"] is not None  # TTL was set


class TestCacheKeyGeneration:
    """Test cache key generation for LLM operations."""

    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        from pyhub.llm.cache import generate_cache_key

        key1 = generate_cache_key("question", q="What is AI?")
        key2 = generate_cache_key("question", q="What is AI?")
        key3 = generate_cache_key("question", q="What is ML?")

        # Same input should generate same key
        assert key1 == key2

        # Different input should generate different key
        assert key1 != key3

    def test_generate_cache_key_with_kwargs(self):
        """Test cache key generation with keyword arguments."""
        from pyhub.llm.cache import generate_cache_key

        key1 = generate_cache_key("question", q="What is AI?", temperature=0.7)
        key2 = generate_cache_key("question", q="What is AI?", temperature=0.7)
        key3 = generate_cache_key("question", q="What is AI?", temperature=0.5)

        # Same kwargs should generate same key
        assert key1 == key2

        # Different kwargs should generate different key
        assert key1 != key3

    def test_generate_cache_key_messages(self):
        """Test cache key generation for message lists."""
        from pyhub.llm.cache import generate_cache_key

        messages1 = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        messages2 = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        messages3 = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hey!"),
        ]

        key1 = generate_cache_key("messages", messages=messages1)
        key2 = generate_cache_key("messages", messages=messages2)
        key3 = generate_cache_key("messages", messages=messages3)

        # Same messages should generate same key
        assert key1 == key2

        # Different messages should generate different key
        assert key1 != key3

    def test_generate_cache_key_order_independence(self):
        """Test that kwargs order doesn't affect key."""
        from pyhub.llm.cache import generate_cache_key

        key1 = generate_cache_key("test", value="value", a=1, b=2, c=3)
        key2 = generate_cache_key("test", value="value", c=3, a=1, b=2)

        # Order of kwargs shouldn't matter
        assert key1 == key2


class TestBaseCache:
    """Test BaseCache abstract interface."""

    def test_cannot_instantiate_base_cache(self):
        """Test that BaseCache cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCache()  # type: ignore

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement all methods."""

        class IncompleteCache(BaseCache):
            def get(self, key: str):
                return None

            # Missing set, delete, clear methods

        with pytest.raises(TypeError):
            IncompleteCache()  # type: ignore

    def test_get_or_set_default_implementation(self):
        """Test default get_or_set implementation."""

        class MinimalCache(BaseCache):
            def __init__(self):
                self.data = {}

            def get(self, key: str):
                return self.data.get(key)

            def set(self, key: str, value, ttl=None):
                self.data[key] = value

            def delete(self, key: str) -> bool:
                if key in self.data:
                    del self.data[key]
                    return True
                return False

            def clear(self):
                self.data.clear()

            def exists(self, key: str) -> bool:
                return key in self.data

        cache = MinimalCache()

        # Test get_or_set with non-existent key
        result = cache.get_or_set("key1", lambda: "generated")
        assert result == "generated"
        assert cache.get("key1") == "generated"

        # Test get_or_set with existing key
        result = cache.get_or_set("key1", lambda: "should not be called")
        assert result == "generated"


class TestCacheDebugging:
    """Test cache debugging and statistics features."""
    
    def test_memory_cache_debug_mode(self):
        """Test MemoryCache debug mode logging."""
        from pyhub.llm.cache import MemoryCache
        
        # Create cache with debug enabled
        cache = MemoryCache(debug=True)
        
        # Test cache miss logging
        result = cache.get("nonexistent")
        assert result is None
        
        # Test cache set
        cache.set("key1", "value1")
        
        # Test cache hit logging
        result = cache.get("key1")
        assert result == "value1"
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        from pyhub.llm.cache import MemoryCache
        
        cache = MemoryCache(debug=True)
        
        # Initial stats should be zero
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
        assert cache.stats["sets"] == 0
        assert cache.stats["hit_rate"] == 0.0
        
        # Cache miss
        cache.get("key1")
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0
        
        # Cache set
        cache.set("key1", "value1")
        assert cache.stats["sets"] == 1
        
        # Cache hit
        cache.get("key1")
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1
        assert cache.stats["hit_rate"] == 0.5  # 1 hit / (1 hit + 1 miss)
        
        # Multiple operations
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        cache.set("key2", "value2")
        cache.get("key2")  # hit
        
        assert cache.stats["hits"] == 3
        assert cache.stats["misses"] == 2
        assert cache.stats["sets"] == 2
        assert cache.stats["hit_rate"] == 0.6  # 3 hits / (3 hits + 2 misses)
    
    def test_file_cache_debug_mode(self, temp_cache_dir):
        """Test FileCache debug mode."""
        cache = FileCache(cache_dir=str(temp_cache_dir), debug=True)
        
        # Test operations
        cache.get("nonexistent")  # miss
        cache.set("key1", "value1")
        cache.get("key1")  # hit
        
        # Check stats
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1
        assert cache.stats["sets"] == 1


class TestReadmeExamples:
    """Test examples from README.md to ensure they work correctly."""
    
    def test_memory_cache_with_ttl_from_readme(self):
        """Test MemoryCache example from README.md."""
        from pyhub.llm.cache import MemoryCache
        
        # Example from README.md
        memory_cache = MemoryCache(ttl=3600)  # 1시간 TTL
        
        # Verify it works as expected
        assert memory_cache._default_ttl == 3600
        
        # Test set/get operations
        memory_cache.set("test_key", "test_value")
        assert memory_cache.get("test_key") == "test_value"
        
        # Test TTL override
        memory_cache.set("short_ttl", "expires quickly", ttl=1)
        assert memory_cache.get("short_ttl") == "expires quickly"
        
        time.sleep(1.1)
        assert memory_cache.get("short_ttl") is None
    
    def test_file_cache_with_ttl_from_readme(self, temp_cache_dir):
        """Test FileCache example from README.md."""
        # Example from README.md
        file_cache = FileCache(cache_dir=".cache", ttl=7200)  # 2시간 TTL
        
        # Verify it works as expected
        assert file_cache._default_ttl == 7200
        
        # Test set/get operations
        file_cache.set("test_key", "test_value")
        assert file_cache.get("test_key") == "test_value"
        
        # Cleanup
        file_cache.clear()
    
    def test_cache_usage_patterns_from_readme(self, temp_cache_dir):
        """Test various cache usage patterns shown in README.md."""
        from pyhub.llm.cache import MemoryCache, FileCache
        
        # Pattern 1: Memory cache with custom TTL
        memory_cache = MemoryCache(ttl=3600)
        memory_cache.set("key1", "value1")  # Uses default TTL
        memory_cache.set("key2", "value2", ttl=60)  # Override TTL
        
        assert memory_cache.get("key1") == "value1"
        assert memory_cache.get("key2") == "value2"
        
        # Pattern 2: File cache without default TTL
        file_cache = FileCache(cache_dir=str(temp_cache_dir))
        assert file_cache._default_ttl is None
        
        file_cache.set("permanent", "no expiry")  # No TTL
        file_cache.set("temporary", "expires", ttl=1)  # With TTL
        
        assert file_cache.get("permanent") == "no expiry"
        assert file_cache.get("temporary") == "expires"
        
        time.sleep(1.1)
        assert file_cache.get("permanent") == "no expiry"  # Still exists
        assert file_cache.get("temporary") is None  # Expired
