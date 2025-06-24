"""Tests for batch processing functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyhub.llm import LLM
from pyhub.llm.types import Reply


class TestBatchProcessing:
    """Test batch() method functionality."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = LLM.create("gpt-4o-mini")
            llm.history = []
            return llm
    
    @pytest.mark.asyncio
    async def test_batch_independent_mode(self, mock_llm):
        """Test batch processing with independent mode."""
        # Mock the ask_async method
        responses = [
            Reply(text="Python is a programming language"),
            Reply(text="JavaScript is a scripting language"),
            Reply(text="Go is a compiled language")
        ]
        
        mock_llm.ask_async = AsyncMock(side_effect=responses)
        
        prompts = [
            "What is Python?",
            "What is JavaScript?",
            "What is Go?"
        ]
        
        results = await mock_llm.batch(prompts)
        
        assert len(results) == 3
        assert results[0].text == "Python is a programming language"
        assert results[1].text == "JavaScript is a scripting language"
        assert results[2].text == "Go is a compiled language"
        
        # Verify ask_async was called 3 times
        assert mock_llm.ask_async.call_count == 3
    
    @pytest.mark.asyncio
    async def test_batch_sequential_mode(self, mock_llm):
        """Test batch processing with sequential mode."""
        # Mock responses for sequential conversation
        responses = [
            Reply(text="The Fibonacci sequence is 0, 1, 1, 2, 3, 5..."),
            Reply(text="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
            Reply(text="The time complexity is O(2^n)")
        ]
        
        mock_llm.ask_async = AsyncMock(side_effect=responses)
        
        prompts = [
            "Explain fibonacci sequence",
            "Implement it in Python",
            "What's the time complexity?"
        ]
        
        results = await mock_llm.batch(
            prompts, 
            history_mode="sequential",
            use_history=True
        )
        
        assert len(results) == 3
        assert "Fibonacci" in results[0].text
        assert "def fibonacci" in results[1].text
        assert "O(2^n)" in results[2].text
        
        # In sequential mode, each call should use history
        assert mock_llm.ask_async.call_count == 3
    
    @pytest.mark.asyncio
    async def test_batch_shared_mode(self, mock_llm):
        """Test batch processing with shared context mode."""
        # Set up initial context
        mock_llm.history = [
            {"role": "user", "content": "Our products are A, B, C"},
            {"role": "assistant", "content": "I understand you have products A, B, and C."}
        ]
        
        responses = [
            Reply(text="Product A is efficient"),
            Reply(text="Product B is reliable"),
            Reply(text="Product C is affordable")
        ]
        
        mock_llm.ask_async = AsyncMock(side_effect=responses)
        
        prompts = [
            "Benefits of product A?",
            "Benefits of product B?",
            "Benefits of product C?"
        ]
        
        results = await mock_llm.batch(
            prompts,
            history_mode="shared",
            use_history=True
        )
        
        assert len(results) == 3
        assert "efficient" in results[0].text
        assert "reliable" in results[1].text
        assert "affordable" in results[2].text
    
    def test_batch_sync(self, mock_llm):
        """Test synchronous batch processing."""
        responses = [
            Reply(text="Answer 1"),
            Reply(text="Answer 2")
        ]
        
        # Mock the ask_async method, so real batch and batch_sync logic is exercised
        mock_llm.ask_async = AsyncMock(side_effect=responses)
        
        prompts = ["Question 1", "Question 2"]
        results = mock_llm.batch_sync(prompts)
        
        assert len(results) == 2
        assert results[0].text == "Answer 1"
        assert results[1].text == "Answer 2"
    
    @pytest.mark.asyncio
    async def test_batch_sync_in_async_context(self, mock_llm):
        """Test that batch_sync raises error when called from async context."""
        with pytest.raises(RuntimeError, match="cannot be called from a running event loop"):
            mock_llm.batch_sync(["Question"])
    
    @pytest.mark.asyncio
    async def test_batch_with_max_parallel(self, mock_llm):
        """Test batch processing with max_parallel limit."""
        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0
        
        async def mock_ask_with_delay(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate API delay
            concurrent_count -= 1
            return Reply(text="Response")
        
        mock_llm.ask_async = mock_ask_with_delay
        
        # Process 10 prompts with max_parallel=3
        prompts = [f"Question {i}" for i in range(10)]
        results = await mock_llm.batch(prompts, max_parallel=3)
        
        assert len(results) == 10
        assert max_concurrent <= 3  # Should not exceed limit
    
    @pytest.mark.asyncio
    async def test_batch_error_handling_continue(self, mock_llm):
        """Test batch processing continues on error when fail_fast=False."""
        responses = [
            Reply(text="Success 1"),
            Exception("API Error"),
            Reply(text="Success 2")
        ]
        
        async def mock_ask_with_errors(*args, **kwargs):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        
        mock_llm.ask_async = mock_ask_with_errors
        
        prompts = ["Q1", "Q2", "Q3"]
        results = await mock_llm.batch(prompts, fail_fast=False)
        
        assert len(results) == 3
        assert results[0].text == "Success 1"
        assert "Error processing prompt 2" in results[1].text
        assert results[2].text == "Success 2"
    
    @pytest.mark.asyncio
    async def test_batch_error_handling_fail_fast(self, mock_llm):
        """Test batch processing stops on error when fail_fast=True."""
        mock_llm.ask_async = AsyncMock(side_effect=[
            Reply(text="Success 1"),
            Exception("API Error"),
            Reply(text="Should not reach")
        ])
        
        prompts = ["Q1", "Q2", "Q3"]
        
        with pytest.raises(Exception, match="API Error"):
            await mock_llm.batch(prompts, fail_fast=True)
    
    @pytest.mark.asyncio
    async def test_batch_empty_prompts(self, mock_llm):
        """Test batch with empty prompt list."""
        results = await mock_llm.batch([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_batch_with_system_prompt(self, mock_llm):
        """Test batch processing with custom system prompt."""
        mock_llm.ask_async = AsyncMock(return_value=Reply(text="Response"))
        
        await mock_llm.batch(
            ["Question"],
            system_prompt="You are a helpful assistant"
        )
        
        # Verify system prompt was set
        mock_llm.ask_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_invalid_history_mode(self, mock_llm):
        """Test batch with invalid history mode."""
        with pytest.raises(ValueError, match="Invalid history_mode"):
            await mock_llm.batch(["Q1"], history_mode="invalid")
    
    @pytest.mark.asyncio 
    async def test_batch_preserves_original_settings(self, mock_llm):
        """Test that batch preserves original LLM settings."""
        original_system_prompt = mock_llm.system_prompt
        original_temperature = mock_llm.temperature
        original_max_tokens = mock_llm.max_tokens
        
        mock_llm.ask_async = AsyncMock(return_value=Reply(text="Response"))
        
        # Run batch with different settings
        await mock_llm.batch(
            ["Question"],
            system_prompt="Temporary prompt",
            temperature=0.5,
            max_tokens=100
        )
        
        # Verify settings were restored
        assert mock_llm.system_prompt == original_system_prompt
        assert mock_llm.temperature == original_temperature
        assert mock_llm.max_tokens == original_max_tokens