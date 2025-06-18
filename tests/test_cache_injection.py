"""Test cache injection functionality."""

import json
from unittest.mock import Mock, patch

import pytest

from pyhub.llm import LLM
from pyhub.llm.cache import FileCache, MemoryCache
from pyhub.llm.cache.base import BaseCache


class MockCache(BaseCache):
    """Mock cache for testing."""

    def __init__(self):
        self.storage = {}
        self.get_called = 0
        self.set_called = 0

    def get(self, key: str, default=None):
        self.get_called += 1
        return self.storage.get(key, default)

    def set(self, key: str, value, ttl=None):
        self.set_called += 1
        self.storage[key] = value

    def delete(self, key: str):
        if key in self.storage:
            del self.storage[key]

    def clear(self):
        self.storage.clear()


class TestCacheInjection:
    """Test cache injection at LLM instance level."""

    def test_llm_accepts_cache_parameter(self):
        """Test that LLM instances can accept cache parameter."""
        cache = MemoryCache()

        # Create LLM with cache
        with patch("openai.OpenAI"):
            from pyhub.llm.openai import OpenAILLM

            llm = OpenAILLM(model="gpt-4o", cache=cache)
            assert llm.cache == cache

    def test_llm_create_with_cache(self):
        """Test LLM.create() accepts cache parameter."""
        cache = MemoryCache()

        with patch("openai.OpenAI"):
            llm = LLM.create("gpt-4o", cache=cache)
            assert llm.cache == cache

    def test_llm_without_cache(self):
        """Test LLM works without cache (default behavior)."""
        with patch("openai.OpenAI"):
            llm = LLM.create("gpt-4o")
            assert llm.cache is None

    @patch("openai.OpenAI")
    def test_cache_hit(self, mock_openai_class):
        """Test cache hit returns cached response."""
        # Setup mock cache
        mock_cache = MockCache()

        # Create cached response (complete ChatCompletion format)
        cached_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Cached response"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Pre-populate cache
        cache_key = "openai:test_cache_key"
        mock_cache.set(cache_key, json.dumps(cached_response))

        # Create LLM with cache
        llm = LLM.create("gpt-4o", cache=mock_cache)

        # Mock the cache key generation to return our test key
        with patch("pyhub.llm.cache.utils.generate_cache_key", return_value=cache_key):
            # Make request (cache is automatically used since llm has cache)
            response = llm.ask("Test question")

            # Should get cached response
            assert response.text == "Cached response"
            assert response.usage.input == 0  # Cached responses have 0 usage
            assert response.usage.output == 0

            # API should not be called
            mock_client = mock_openai_class.return_value
            mock_client.chat.completions.create.assert_not_called()

    @patch("openai.OpenAI")
    def test_cache_miss_and_set(self, mock_openai_class):
        """Test cache miss triggers API call and stores response."""
        # Setup mock cache
        mock_cache = MockCache()

        # Setup mock OpenAI response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Fresh response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump_json.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "Fresh response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

        mock_client.chat.completions.create.return_value = mock_response

        # Create LLM with cache
        llm = LLM.create("gpt-4o", cache=mock_cache)

        # Make request (cache is automatically used since llm has cache)
        response = llm.ask("Test question")

        # Should get fresh response
        assert response.text == "Fresh response"
        assert response.usage.input == 10
        assert response.usage.output == 5

        # Cache should be populated
        assert mock_cache.set_called == 1
        assert len(mock_cache.storage) == 1

    @patch("openai.OpenAI")
    def test_cache_disabled_when_no_cache_injected(self, mock_openai_class):
        """Test that cache is not used when no cache is injected."""
        # Setup mock OpenAI
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client.chat.completions.create.return_value = mock_response

        # Create LLM without cache (cache=None)
        llm = LLM.create("gpt-4o")  # No cache injected

        # Make request (no cache to use)
        _response = llm.ask("Test question")

        # No cache operations should occur
        assert llm.cache is None

        # API should be called
        mock_client.chat.completions.create.assert_called_once()

    def test_multiple_llms_different_caches(self):
        """Test multiple LLMs can have different cache strategies."""
        cache1 = MemoryCache()
        cache2 = MockCache()

        with patch("openai.OpenAI"):
            llm1 = LLM.create("gpt-4o", cache=cache1)
            llm2 = LLM.create("gpt-4o-mini", cache=cache2)

            assert llm1.cache == cache1
            assert llm2.cache == cache2
            assert llm1.cache != llm2.cache

    def test_shared_cache_between_llms(self):
        """Test multiple LLMs can share the same cache."""
        shared_cache = MemoryCache()

        with patch("openai.OpenAI"), patch("anthropic.Anthropic"):
            llm1 = LLM.create("gpt-4o", cache=shared_cache)
            llm2 = LLM.create("claude-3-5-sonnet-latest", cache=shared_cache)

            assert llm1.cache is llm2.cache

    def test_file_cache_injection(self, temp_cache_dir):
        """Test FileCache injection works correctly."""
        file_cache = FileCache(str(temp_cache_dir))

        with patch("openai.OpenAI"):
            llm = LLM.create("gpt-4o", cache=file_cache)
            assert isinstance(llm.cache, FileCache)
            assert llm.cache.cache_dir == temp_cache_dir

    def test_custom_cache_implementation(self):
        """Test custom cache implementation can be injected."""

        class CustomCache(BaseCache):
            def __init__(self):
                self.data = {}

            def get(self, key: str, default=None):
                return self.data.get(key, default)

            def set(self, key: str, value, ttl=None):
                self.data[key] = value

            def delete(self, key: str):
                self.data.pop(key, None)

            def clear(self):
                self.data.clear()

        custom_cache = CustomCache()

        with patch("openai.OpenAI"):
            llm = LLM.create("gpt-4o", cache=custom_cache)
            assert isinstance(llm.cache, CustomCache)

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_async_cache_operations(self, mock_async_openai_class):
        """Test cache works with async operations."""
        mock_cache = MockCache()

        # Setup mock response
        mock_client = Mock()
        mock_async_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Async response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump_json.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "Async response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

        # Make create return a coroutine
        async def mock_create(**kwargs):
            return mock_response

        mock_client.chat.completions.create = mock_create

        # Create LLM with cache
        llm = LLM.create("gpt-4o", cache=mock_cache)

        # Make async request (cache is automatically used since llm has cache)
        response = await llm.ask_async("Test question")

        assert response.text == "Async response"
        assert mock_cache.set_called == 1


class TestCacheKeyGeneration:
    """Test cache key generation with injected cache."""

    @patch("openai.OpenAI")
    def test_cache_key_includes_all_parameters(self, mock_openai_class):
        """Test cache key includes all relevant parameters."""
        mock_cache = MockCache()

        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump_json.return_value = json.dumps(
            {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Response"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
        mock_client.chat.completions.create.return_value = mock_response

        llm = LLM.create("gpt-4o", cache=mock_cache, temperature=0.5, max_tokens=100)

        # Make two identical requests without saving history
        llm.ask("Same question", use_history=False)

        # Check cache was populated
        assert len(mock_cache.storage) == 1
        assert mock_cache.set_called == 1

        # Make same request again
        llm.ask("Same question", use_history=False)

        # Should only call API once (second is cached)
        assert mock_client.chat.completions.create.call_count == 1
        assert mock_cache.get_called == 2  # Called twice for both requests

        # Make request with different parameters
        llm2 = LLM.create("gpt-4o", cache=mock_cache, temperature=0.8, max_tokens=100)
        llm2.ask("Same question")

        # Should call API again (different temperature)
        assert mock_client.chat.completions.create.call_count == 2
