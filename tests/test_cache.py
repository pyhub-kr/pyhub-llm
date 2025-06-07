import pytest
import time
import json
import pickle
from pathlib import Path
from unittest.mock import Mock, patch
from pyhub.llm.cache import BaseCache, MemoryCache, FileCache
from pyhub.llm.types import Message


class TestMemoryCache:
    """Test MemoryCache implementation."""
    
    def test_set_and_get(self, memory_cache):
        """Test basic set and get operations."""
        memory_cache.set("key1", "value1")
        assert memory_cache.get("key1") == "value1"
    
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
        memory_cache.set("key1", "value1", timeout=1)  # 1 second TTL
        
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
        file_cache.set("key1", "value1", timeout=1)  # 1 second TTL
        
        # Should be available immediately
        assert file_cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert file_cache.get("key1") is None
    
    def test_key_hashing(self, file_cache):
        """Test key hashing for file names."""
        # Long keys should be hashed
        long_key = "a" * 300
        file_cache.set(long_key, "value")
        
        # Should create a hashed filename
        cache_files = list(file_cache.cache_dir.glob("*.cache"))
        assert len(cache_files) == 1
        assert len(cache_files[0].stem) == 64  # SHA256 hash length
    
    def test_complex_data_types(self, file_cache):
        """Test caching complex data types."""
        # List
        file_cache.set("list_key", [1, 2, 3])
        assert file_cache.get("list_key") == [1, 2, 3]
        
        # Dict
        file_cache.set("dict_key", {"a": 1, "b": 2})
        assert file_cache.get("dict_key") == {"a": 1, "b": 2}
        
        # Custom object
        obj = Message(role="user", content="Hello")
        file_cache.set("obj_key", obj)
        retrieved = file_cache.get("obj_key")
        assert retrieved.role == "user"
        assert retrieved.content == "Hello"
    
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
        assert retrieved_obj.role == "user"
    
    def test_corrupted_cache_file(self, file_cache):
        """Test handling of corrupted cache files."""
        file_cache.set("key1", "value1")
        
        # Corrupt the cache file
        cache_files = list(file_cache.cache_dir.glob("*.cache"))
        assert len(cache_files) == 1
        
        with open(cache_files[0], 'w') as f:
            f.write("corrupted data")
        
        # Should return None for corrupted file
        assert file_cache.get("key1") is None
    
    def test_metadata_file(self, file_cache):
        """Test metadata file creation and usage."""
        file_cache.set("key1", "value1", timeout=3600)
        
        # Check metadata file exists
        meta_files = list(file_cache.cache_dir.glob("*.meta"))
        assert len(meta_files) == 1
        
        # Check metadata content
        with open(meta_files[0], 'r') as f:
            metadata = json.load(f)
        
        assert "created_at" in metadata
        assert "expires_at" in metadata
        assert metadata["expires_at"] > metadata["created_at"]


class TestCacheKeyGeneration:
    """Test cache key generation for LLM operations."""
    
    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        from pyhub.llm.cache import generate_cache_key
        
        key1 = generate_cache_key("question", "What is AI?")
        key2 = generate_cache_key("question", "What is AI?")
        key3 = generate_cache_key("question", "What is ML?")
        
        # Same input should generate same key
        assert key1 == key2
        
        # Different input should generate different key
        assert key1 != key3
    
    def test_generate_cache_key_with_kwargs(self):
        """Test cache key generation with keyword arguments."""
        from pyhub.llm.cache import generate_cache_key
        
        key1 = generate_cache_key("question", "What is AI?", temperature=0.7)
        key2 = generate_cache_key("question", "What is AI?", temperature=0.7)
        key3 = generate_cache_key("question", "What is AI?", temperature=0.5)
        
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
        
        key1 = generate_cache_key("messages", messages1)
        key2 = generate_cache_key("messages", messages2)
        key3 = generate_cache_key("messages", messages3)
        
        # Same messages should generate same key
        assert key1 == key2
        
        # Different messages should generate different key
        assert key1 != key3
    
    def test_generate_cache_key_order_independence(self):
        """Test that kwargs order doesn't affect key."""
        from pyhub.llm.cache import generate_cache_key
        
        key1 = generate_cache_key("test", "value", a=1, b=2, c=3)
        key2 = generate_cache_key("test", "value", c=3, a=1, b=2)
        
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
            
            def set(self, key: str, value, timeout=None):
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