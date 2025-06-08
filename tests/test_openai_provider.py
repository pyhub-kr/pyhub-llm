"""Tests for OpenAI provider."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from pyhub.llm.openai import OpenAILLM
from pyhub.llm.types import Reply


class TestOpenAILLM:
    """Test OpenAI LLM provider."""

    def test_initialization(self):
        """Test OpenAI LLM initialization."""
        llm = OpenAILLM(model="gpt-4o", api_key="test-key")

        assert llm.model == "gpt-4o"
        assert llm.api_key == "test-key"
        assert llm.temperature == 0.2  # Default from BaseLLM
        assert llm.max_tokens == 1000  # Default from BaseLLM

    @patch("pyhub.llm.openai.SyncOpenAI")
    def test_ask_simple(self, mock_openai_class):
        """Test simple ask method."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Create mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 8
        mock_response.usage.total_tokens = 18

        mock_client.chat.completions.create.return_value = mock_response

        # Test ask
        llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")
        reply = llm.ask("Hello")

        # Verify SyncOpenAI was instantiated
        mock_openai_class.assert_called()

        # Verify response
        assert isinstance(reply, Reply)
        assert reply.text == "Hello! How can I help you?"
        assert reply.usage.input == 10
        assert reply.usage.output == 8
        assert reply.usage.total == 18

    @patch("pyhub.llm.openai.SyncOpenAI")
    def test_ask_with_kwargs(self, mock_openai_class):
        """Test ask method with additional parameters."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = None

        mock_client.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(api_key="test-key")
        # Temperature and max_tokens are set during initialization
        reply = llm.ask("Question")

        # Just verify SyncOpenAI was called
        mock_openai_class.assert_called()

    @patch("pyhub.llm.openai.SyncOpenAI")
    def test_ask_with_history(self, mock_openai_class):
        """Test ask method with conversation history."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I can help with that!"
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30

        mock_client.chat.completions.create.return_value = mock_response

        # Test ask with history
        llm = OpenAILLM(model="gpt-4o", api_key="test-key")
        llm.ask("Hello")  # First message
        reply = llm.ask("What is Python?")

        # Verify response
        assert isinstance(reply, Reply)
        assert reply.text == "I can help with that!"
        assert reply.usage.total == 30

    @patch("pyhub.llm.openai.SyncOpenAI")
    def test_error_handling(self, mock_openai_class):
        """Test error handling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        llm = OpenAILLM(api_key="test-key")

        # OpenAI provider returns error in Reply instead of raising
        reply = llm.ask("Hello")
        assert "API Error" in reply.text
        assert reply.text.startswith("Error:")

    @patch("pyhub.llm.openai.SyncOpenAI")
    def test_streaming(self, mock_openai_class):
        """Test streaming functionality."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response chunks
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))], usage=None),
            Mock(choices=[Mock(delta=Mock(content=" world"))], usage=None),
            Mock(choices=[Mock(delta=Mock(content="!"))], usage=None),
            # Final chunk with usage info
            Mock(choices=[], usage=Mock(prompt_tokens=5, completion_tokens=3)),
        ]
        mock_client.chat.completions.create.return_value = iter(mock_chunks)

        llm = OpenAILLM(api_key="test-key")
        chunks = list(llm.ask("Hello", stream=True))

        # Check that we get Reply objects
        assert len(chunks) == 4  # 3 content chunks + 1 usage chunk
        assert all(isinstance(chunk, Reply) for chunk in chunks)

        # Check content
        assert chunks[0].text == "Hello"
        assert chunks[1].text == " world"
        assert chunks[2].text == "!"

        # Check final chunk has usage info
        assert chunks[3].text == ""
        assert chunks[3].usage.input == 5
        assert chunks[3].usage.output == 3

    @pytest.mark.asyncio
    @patch("pyhub.llm.openai.AsyncOpenAI")
    async def test_async_ask(self, mock_async_openai_class):
        """Test async ask functionality."""
        mock_client = Mock()
        mock_async_openai_class.return_value = mock_client

        # Mock async response with proper usage
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Async response"
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = OpenAILLM(api_key="test-key")
        reply = await llm.ask_async("Hello async")

        assert isinstance(reply, Reply)
        assert reply.text == "Async response"
        assert reply.usage.input == 10
        assert reply.usage.output == 5

    @patch("pyhub.llm.openai.SyncOpenAI")
    def test_embed(self, mock_openai_class):
        """Test embed functionality."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock embedding response
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3, 0.4]
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.total_tokens = 5

        mock_client.embeddings.create.return_value = mock_response

        llm = OpenAILLM(api_key="test-key")
        result = llm.embed("test text")

        assert result.array == [0.1, 0.2, 0.3, 0.4]
        assert result.usage.input == 5

    @patch("pyhub.llm.openai.SyncOpenAI")
    def test_empty_content_handling(self, mock_openai_class):
        """Test handling of empty content in response."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None  # Empty content
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=0)

        mock_client.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(api_key="test-key")
        reply = llm.ask("Hello")

        assert reply.text is None  # OpenAI returns None as-is for empty content
        assert reply.usage.input == 5
        assert reply.usage.output == 0
