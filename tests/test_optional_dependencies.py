import sys
from unittest.mock import MagicMock, patch

import pytest


class TestOptionalDependencies:
    """Test that optional dependencies are handled correctly with lazy imports."""

    def test_import_without_optional_deps(self):
        """Test that importing pyhub.llm works without optional dependencies."""
        # This should work since it doesn't import provider classes directly
        from pyhub.llm import LLM, BaseLLM, MockLLM

        # These should be importable
        assert LLM is not None
        assert BaseLLM is not None
        assert MockLLM is not None

    def test_anthropic_import_error(self):
        """Test that AnthropicLLM raises ImportError when anthropic is not installed."""
        with patch.dict(sys.modules, {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic package not installed"):
                from pyhub.llm import AnthropicLLM

                # This should trigger the import
                AnthropicLLM()

    def test_openai_import_error(self):
        """Test that OpenAILLM raises ImportError when openai is not installed."""
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError, match="openai package not installed"):
                from pyhub.llm import OpenAILLM

                # This should trigger the import
                OpenAILLM()

    def test_google_import_error(self):
        """Test that GoogleLLM raises ImportError when google-genai is not installed."""
        with patch.dict(sys.modules, {"google": None}):
            with pytest.raises(ImportError, match="google-genai package not installed"):
                from pyhub.llm import GoogleLLM

                # This should trigger the import
                GoogleLLM()

    def test_ollama_import_error(self):
        """Test that OllamaLLM raises ImportError when ollama is not installed."""
        with patch.dict(sys.modules, {"ollama": None}):
            with pytest.raises(ImportError, match="ollama package not installed"):
                from pyhub.llm import OllamaLLM

                # This should trigger the import
                OllamaLLM()

    def test_upstage_import_error(self):
        """Test that UpstageLLM raises ImportError when openai is not installed."""
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError, match="openai package not installed.*upstage"):
                from pyhub.llm import UpstageLLM

                # This should trigger the import
                UpstageLLM()

    def test_llm_create_with_missing_deps(self):
        """Test that LLM.create raises ImportError for providers with missing dependencies."""
        from pyhub.llm import LLM

        # Test Anthropic
        with patch.dict(sys.modules, {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic package not installed"):
                LLM.create("claude-3-5-haiku-latest")

        # Test OpenAI
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError, match="openai package not installed"):
                LLM.create("gpt-4o-mini")

        # Test Google
        with patch.dict(sys.modules, {"google": None}):
            with pytest.raises(ImportError, match="google-genai package not installed"):
                LLM.create("gemini-2.0-flash")

        # Test Ollama
        with patch.dict(sys.modules, {"ollama": None}):
            with pytest.raises(ImportError, match="ollama package not installed"):
                LLM.create("mistral")

        # Test Upstage
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError, match="openai package not installed.*upstage"):
                LLM.create("solar-mini")

    def test_successful_import_with_mock_deps(self):
        """Test that providers can be imported when dependencies are available."""
        # Mock the dependencies
        mock_anthropic = MagicMock()
        mock_openai = MagicMock()
        mock_google = MagicMock()
        mock_ollama = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "anthropic": mock_anthropic,
                "anthropic.types": MagicMock(),
                "openai": mock_openai,
                "openai.types": MagicMock(),
                "openai.types.chat": MagicMock(),
                "google.generativeai": mock_google,
                "google.generativeai.types": MagicMock(),
                "ollama": mock_ollama,
            },
        ):
            # These imports should work now
            from pyhub.llm import (
                AnthropicLLM,
                GoogleLLM,
                OllamaLLM,
                OpenAILLM,
                UpstageLLM,
            )

            # Verify they're the correct classes
            assert AnthropicLLM.__name__ == "AnthropicLLM"
            assert OpenAILLM.__name__ == "OpenAILLM"
            assert GoogleLLM.__name__ == "GoogleLLM"
            assert OllamaLLM.__name__ == "OllamaLLM"
            assert UpstageLLM.__name__ == "UpstageLLM"

    def test_lazy_import_attribute_error(self):
        """Test that accessing non-existent attributes raises AttributeError."""
        import pyhub.llm

        with pytest.raises(AttributeError, match="has no attribute 'NonExistentLLM'"):
            pyhub.llm.NonExistentLLM
