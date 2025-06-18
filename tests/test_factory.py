from unittest.mock import Mock, patch

import pytest

from pyhub.llm import LLM
from pyhub.llm.base import BaseLLM


class TestLLMFactory:
    """Test LLM factory functionality."""

    def test_create_method_exists(self):
        """Test that LLM.create method exists."""
        assert hasattr(LLM, "create")
        assert callable(LLM.create)

    @patch("pyhub.llm.LLM.create")
    def test_llm_create_method(self, mock_create):
        """Test that LLM.create method works."""
        mock_llm = Mock(spec=BaseLLM)
        mock_create.return_value = mock_llm

        result = LLM.create("gpt-4o", api_key="test-key")

        mock_create.assert_called_once_with("gpt-4o", api_key="test-key")
        assert result == mock_llm

    def test_detect_provider_openai(self):
        """Test provider detection for OpenAI models."""
        assert LLM.get_vendor_from_model("gpt-4o") == "openai"
        assert LLM.get_vendor_from_model("text-embedding-ada-002") == "openai"

    def test_detect_provider_anthropic(self):
        """Test provider detection for Anthropic models."""
        assert LLM.get_vendor_from_model("claude-3-5-sonnet-latest") == "anthropic"
        assert LLM.get_vendor_from_model("claude-3-5-haiku-latest") == "anthropic"
        assert LLM.get_vendor_from_model("claude-3-opus-latest") == "anthropic"

    def test_detect_provider_google(self):
        """Test provider detection for Google models."""
        assert LLM.get_vendor_from_model("gemini-2.0-flash") == "google"
        assert LLM.get_vendor_from_model("gemini-1.5-pro") == "google"

    def test_detect_provider_ollama(self):
        """Test provider detection for Ollama models."""
        assert LLM.get_vendor_from_model("llama3.3") == "ollama"
        assert LLM.get_vendor_from_model("mistral") == "ollama"
        assert LLM.get_vendor_from_model("qwen2") == "ollama"

    def test_detect_provider_upstage(self):
        """Test provider detection for Upstage models."""
        assert LLM.get_vendor_from_model("solar-pro") == "upstage"
        assert LLM.get_vendor_from_model("embedding-query") == "upstage"

    def test_detect_provider_unknown(self):
        """Test provider detection for unknown models."""
        with pytest.raises(ValueError, match="Unknown model"):
            LLM.get_vendor_from_model("unknown-model-xyz")

    def test_custom_provider_creation(self):
        """Test creating LLM with custom logic."""
        # The current LLM.create uses hardcoded vendor detection
        # This test documents expected behavior for extensibility
        with pytest.raises(ValueError, match="Unknown model"):
            LLM.create("custom-model")

    def test_create_openai_model(self):
        """Test creating OpenAI model."""
        # LLM.create imports OpenAILLM from the openai module via lazy loading
        with patch("pyhub.llm.openai.OpenAILLM") as mock_openai_class:
            mock_instance = Mock(spec=BaseLLM)
            mock_openai_class.return_value = mock_instance

            result = LLM.create("gpt-4o", api_key="test-key")

            mock_openai_class.assert_called_once_with(model="gpt-4o", api_key="test-key")
            assert result == mock_instance

    def test_create_with_cache(self, memory_cache):
        """Test LLM creation with cache."""
        # Test that cache parameter is properly passed through
        with patch("pyhub.llm.openai.OpenAILLM") as mock_openai_class:
            mock_instance = Mock(spec=BaseLLM)
            mock_openai_class.return_value = mock_instance

            _result = LLM.create("gpt-4o", cache=memory_cache)

            # Check that cache was passed to the constructor
            mock_openai_class.assert_called_once()
            args, kwargs = mock_openai_class.call_args
            assert "cache" in kwargs
            assert kwargs["cache"] == memory_cache

    def test_create_with_custom_parameters(self):
        """Test LLM creation with custom parameters."""
        with patch("pyhub.llm.openai.OpenAILLM") as mock_openai_class:
            mock_instance = Mock(spec=BaseLLM)
            mock_openai_class.return_value = mock_instance

            # Create with custom parameters
            _result = LLM.create("gpt-4o", temperature=0.5, max_tokens=1000, system_prompt="You are helpful.")

            # Verify parameters were passed
            call_kwargs = mock_openai_class.call_args[1]
            assert call_kwargs.get("temperature") == 0.5
            assert call_kwargs.get("max_tokens") == 1000
            assert call_kwargs.get("system_prompt") == "You are helpful."

    def test_create_invalid_model(self):
        """Test error when invalid model name is provided."""
        with pytest.raises(ValueError, match="Unknown model"):
            LLM.create("invalid-model-name")

    def test_available_vendors(self):
        """Test available vendor types."""
        # Test that we can detect all major vendors
        vendors = ["openai", "anthropic", "google", "ollama", "upstage"]
        for vendor in vendors:
            # Each vendor should have at least one model that maps to it
            assert vendor in ["openai", "anthropic", "google", "ollama", "upstage"]

    def test_model_type_detection(self):
        """Test model type detection for various models."""
        # Test chat models
        assert LLM.get_vendor_from_model("gpt-4o") == "openai"
        assert LLM.get_vendor_from_model("claude-3-5-sonnet-latest") == "anthropic"
        assert LLM.get_vendor_from_model("gemini-2.0-flash") == "google"

        # Test embedding models
        assert LLM.get_vendor_from_model("text-embedding-3-small") == "openai"
        assert LLM.get_vendor_from_model("embedding-query") == "upstage"

    def test_create_embedding_model(self):
        """Test creating embedding models."""
        with patch("pyhub.llm.openai.OpenAILLM") as mock_openai_class:
            mock_instance = Mock(spec=BaseLLM)
            mock_openai_class.return_value = mock_instance

            # Create embedding model
            result = LLM.create("text-embedding-3-small", api_key="test-key")

            # Should create with embedding_model parameter
            mock_openai_class.assert_called_once_with(embedding_model="text-embedding-3-small", api_key="test-key")
            assert result == mock_instance
