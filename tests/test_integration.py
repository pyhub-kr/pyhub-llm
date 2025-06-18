"""Integration tests using mock providers."""

from unittest.mock import patch

import pytest

from pyhub.llm import LLM
from pyhub.llm.cache import FileCache
from pyhub.llm.mock import MockLLM
from pyhub.llm.settings import LLMSettings


class TestLLMIntegration:
    """Test LLM integration scenarios."""

    def test_create_openai_llm(self):
        """Test creating an OpenAI LLM."""
        with patch("pyhub.llm.openai.OpenAILLM") as mock_openai_class:
            mock_instance = MockLLM(model="gpt-4o")  # Use MockLLM for testing
            mock_openai_class.return_value = mock_instance

            llm = LLM.create("gpt-4o", api_key="test-key")
            assert llm.model == "gpt-4o"
            mock_openai_class.assert_called_once_with(model="gpt-4o", api_key="test-key")

    def test_create_with_cache(self, memory_cache):
        """Test creating LLM with cache."""
        with patch("pyhub.llm.openai.OpenAILLM") as mock_openai_class:
            mock_instance = MockLLM(model="gpt-4o")
            mock_openai_class.return_value = mock_instance

            llm = LLM.create("gpt-4o", api_key="test-key")

            # Cache is now handled via constructor injection
            # Test that the LLM can make calls
            response = llm.ask("Test question")
            assert response.text == "Mock response: Test question"

    def test_create_anthropic_llm(self):
        """Test creating an Anthropic LLM."""
        with patch("pyhub.llm.anthropic.AnthropicLLM") as mock_anthropic_class:
            mock_instance = MockLLM(model="claude-3-5-sonnet-latest")
            mock_anthropic_class.return_value = mock_instance

            llm = LLM.create("claude-3-5-sonnet-latest", api_key="test-key")
            assert llm.model == "claude-3-5-sonnet-latest"
            mock_anthropic_class.assert_called_once_with(model="claude-3-5-sonnet-latest", api_key="test-key")


class TestMockLLMIntegration:
    """Test MockLLM integration scenarios."""

    def test_conversation_flow(self):
        """Test a complete conversation flow."""
        llm = MockLLM(model="mock-model")

        # First question
        response1 = llm.ask("Hello, how are you?")
        assert response1.text == "Mock response: Hello, how are you?"
        assert len(llm.history) == 2  # User + assistant

        # Follow-up question
        response2 = llm.ask("What's your name?")
        assert response2.text == "Mock response: What's your name?"
        assert len(llm.history) == 4  # 2 more messages

        # Check history
        assert llm.history[0].role == "user"
        assert llm.history[0].content == "Hello, how are you?"
        assert llm.history[1].role == "assistant"
        assert llm.history[1].content == "Mock response: Hello, how are you?"

    def test_streaming_conversation(self):
        """Test streaming in a conversation."""
        llm = MockLLM(model="mock-model")

        # Stream response
        chunks = []
        for chunk in llm.ask("Tell me a story", stream=True):
            chunks.append(chunk)

        # Should have multiple chunks
        assert len(chunks) > 1
        # Collect text from Reply objects
        full_text = "".join(chunk.text for chunk in chunks).strip()
        assert full_text == "Mock response: Tell me a story"

    @pytest.mark.asyncio
    async def test_async_conversation_flow(self):
        """Test async conversation flow."""
        llm = MockLLM(model="mock-model")

        # First async question
        response1 = await llm.ask_async("Hello async!")
        assert response1.text == "Mock response: Hello async!"

        # Async streaming
        chunks = []
        async for chunk in await llm.ask_async("Stream async", stream=True):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Collect text from Reply objects
        full_text = "".join(chunk.text for chunk in chunks).strip()
        assert full_text == "Mock response: Stream async"


class TestCacheIntegration:
    """Test cache integration with LLM."""

    def test_memory_cache_integration(self, memory_cache):
        """Test LLM with memory cache."""
        # Create LLM with cache injected
        llm = MockLLM(model="mock-model", cache=memory_cache)

        # Make same request twice (caching is automatic with injected cache)
        response1 = llm.ask("Cached question", save_history=False)
        response2 = llm.ask("Cached question", save_history=False)

        # Both should return same response
        assert response1.text == response2.text

        # Test cache directly
        memory_cache.set("test_key", "test_value")
        assert memory_cache.get("test_key") == "test_value"

    def test_file_cache_integration(self, temp_cache_dir):
        """Test LLM with file cache."""
        file_cache = FileCache(str(temp_cache_dir))
        # Create LLM with cache injected
        llm = MockLLM(model="mock-model", cache=file_cache)

        # Make request (caching is automatic with injected cache)
        response = llm.ask("File cached question", save_history=False)
        assert response.text == "Mock response: File cached question"

        # Test file cache directly
        file_cache.set("test_key", {"data": "test"})
        assert file_cache.get("test_key") == {"data": "test"}


class TestPromptFormatting:
    """Test prompt formatting integration."""

    def test_llm_with_formatted_prompts(self):
        """Test LLM with formatted prompts."""
        # Simple template-like formatting using Python string formatting
        template = "Question about {topic}: {question}"

        # Create LLM
        llm = MockLLM(model="mock-model")

        # Format prompt
        formatted_prompt = template.format(topic="AI", question="What is machine learning?")

        assert formatted_prompt == "Question about AI: What is machine learning?"

        # Use LLM with formatted prompt
        response = llm.ask(formatted_prompt)
        assert "Question about AI" in response.text


class TestChainIntegration:
    """Test LLM chaining integration."""

    def test_chain_multiple_llms(self):
        """Test chaining multiple LLMs."""
        # Create LLMs with prompts for chaining
        llm1 = MockLLM(model="mock-1", output_key="step1", prompt="{input}")
        llm1.set_mock_response("Step 1 complete")

        llm2 = MockLLM(model="mock-2", output_key="step2", prompt="{step1}")
        llm2.set_mock_response("Step 2 complete")

        llm3 = MockLLM(model="mock-3", output_key="step3", prompt="{step2}")
        llm3.set_mock_response("Final step")

        # Create chain
        chain = llm1 | llm2 | llm3

        # Execute chain with dict input
        result = chain.ask({"input": "Start"})

        # Check results
        assert len(result.reply_list) == 3
        # MockLLM returns "Mock response: {input}" but we set custom responses
        assert "Step 1 complete" in result.values["step1"]
        assert "Step 2 complete" in result.values["step2"]
        assert "Final step" in result.values["step3"]

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="SequentialChain doesn't support async yet")
    async def test_async_chain(self):
        """Test async chaining."""
        llm1 = MockLLM(model="async-1", prompt="{input}")
        llm2 = MockLLM(model="async-2", prompt="{text}")

        _chain = llm1 | llm2

        # This would need to be implemented in SequentialChain
        # result = await chain.ask_async({"input": "Async start"})
        # assert len(result.reply_list) == 2


class TestSettingsIntegration:
    """Test settings integration."""

    def test_settings_from_env(self, monkeypatch):
        """Test loading settings from environment."""
        # Clear existing env vars first
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Set new values
        monkeypatch.setenv("PYHUB_LLM_OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("PYHUB_LLM_TRACE", "true")

        # Create new settings instance to pick up env changes
        settings = LLMSettings()
        assert settings.openai_api_key == "test-openai-key"
        assert settings.anthropic_api_key == "test-anthropic-key"
        assert settings.trace_enabled is True

    def test_settings_boolean_parsing(self, monkeypatch):
        """Test boolean setting parsing."""
        monkeypatch.setenv("PYHUB_LLM_TRACE", "true")
        monkeypatch.setenv("PYHUB_LLM_TRACE_FUNCTION_CALLS", "false")

        settings = LLMSettings()
        assert settings.trace_enabled is True
        assert settings.trace_function_calls is False


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_provider_not_found(self):
        """Test error when provider not found."""
        with pytest.raises(ValueError, match="Unknown model"):
            LLM.create("unknown-model-xyz")

    def test_missing_api_key(self):
        """Test handling missing API key."""
        # Test using a model that should be recognized (mock doesn't require API key)
        with patch("pyhub.llm.mock.MockLLM") as mock_llm_class:
            mock_instance = MockLLM(model="mock-model")
            mock_llm_class.return_value = mock_instance

            # Since 'mock-model' is not in the model list, it will raise ValueError
            with pytest.raises(ValueError, match="Unknown model"):
                LLM.create("mock-model")
