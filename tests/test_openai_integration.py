"""Integration test for OpenAI provider with factory."""

from unittest.mock import Mock, patch


from pyhub.llm import LLM
from pyhub.llm.openai import OpenAILLM
from pyhub.llm.types import Reply


class TestOpenAIIntegration:
    """Test OpenAI provider integration with factory."""

    @patch("openai.OpenAI")
    def test_create_openai_via_factory(self, mock_openai_class):
        """Test creating OpenAI LLM via factory."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello from GPT-4!"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 4

        mock_client.chat.completions.create.return_value = mock_response

        # Create LLM via factory
        llm = LLM.create("gpt-4o", api_key="test-key")

        # Verify it's an OpenAI instance
        assert isinstance(llm, OpenAILLM)
        assert llm.model == "gpt-4o"

        # Test using the LLM
        reply = llm.ask("Hello")
        assert isinstance(reply, Reply)
        assert reply.text == "Hello from GPT-4!"
        assert reply.usage.input == 5
        assert reply.usage.output == 4

    def test_detect_openai_models(self):
        """Test that various OpenAI models are detected correctly."""
        openai_models = ["gpt-4o", "o1-preview", "text-embedding-ada-002", "gpt-4o"]

        for model in openai_models:
            provider = LLM.get_vendor_from_model(model)
            assert provider == "openai", f"Model {model} should be detected as OpenAI"
