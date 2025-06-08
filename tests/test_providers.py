import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from pyhub.llm import LLM
from pyhub.llm.mock import MockLLM
from pyhub.llm.types import Message


class TestProviderDetection:
    """Test provider detection functionality."""
    
    def test_detect_openai_provider(self):
        """Test detecting OpenAI provider from model names."""
        openai_models = ["gpt-4o", "gpt-4o-mini", "text-embedding-ada-002"]
        for model in openai_models:
            assert LLM.get_vendor_from_model(model) == "openai"
    
    def test_detect_anthropic_provider(self):
        """Test detecting Anthropic provider from model names."""
        anthropic_models = ["claude-3-5-sonnet-latest", "claude-3-opus-latest"]
        for model in anthropic_models:
            assert LLM.get_vendor_from_model(model) == "anthropic"
    
    def test_create_with_correct_provider(self):
        """Test creating LLM instances with correct providers."""
        # Test OpenAI
        with patch('pyhub.llm.OpenAILLM') as mock_openai:
            mock_openai.return_value = Mock()
            llm = LLM.create("gpt-4o")
            mock_openai.assert_called_once()


class TestMockProvider:
    """Test mock provider functionality."""
    
    def test_mock_provider_ask(self):
        """Test mock provider ask method."""
        llm = MockLLM(model="mock-model")
        response = llm.ask("Test question")
        
        assert response.text == "Mock response: Test question"
        assert response.usage.input == 10
        assert response.usage.output == 20
    
    @pytest.mark.asyncio
    async def test_mock_provider_ask_async(self):
        """Test mock provider async ask."""
        llm = MockLLM(model="mock-model")
        response = await llm.ask_async("Async question")
        
        assert response.text == "Mock response: Async question"
    
    def test_mock_provider_streaming(self):
        """Test mock provider streaming."""
        llm = MockLLM(model="mock-model")
        
        chunks = list(llm.ask("Stream test", stream=True))
        assert len(chunks) > 0
        assert "".join(chunks).strip() == "Mock response: Stream test"
    
    def test_mock_provider_messages(self):
        """Test mock provider messages method."""
        llm = MockLLM(model="mock-model")
        
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?")
        ]
        
        response = llm.messages(messages)
        assert response.text == "Mock response: How are you?"
    
    def test_mock_provider_embeddings(self):
        """Test mock provider embeddings."""
        llm = MockLLM(model="mock-model")
        
        # Single text
        result = llm.embed("Test text")
        assert result.array == [0.1, 0.2, 0.3, 0.4]
        assert result.usage.input == 5
        
        # Multiple texts
        result = llm.embed(["Text 1", "Text 2", "Text 3"])
        assert len(result.arrays) == 3
        assert result.usage.input == 15  # 5 * 3
    
    def test_mock_provider_custom_response(self):
        """Test setting custom mock responses."""
        llm = MockLLM(model="mock-model")
        
        # Set custom response
        llm.set_mock_response("Custom response")
        response = llm.ask("Any question")
        assert response.text == "Custom response: Any question"
        
        # Set custom usage
        llm.set_mock_usage(100, 200)
        response = llm.ask("Another question")
        assert response.usage.input == 100
        assert response.usage.output == 200
    
    def test_mock_provider_with_choices(self):
        """Test mock provider with choices parameter."""
        llm = MockLLM(model="mock-model")
        
        choices = ["Yes", "No", "Maybe"]
        response = llm.ask("Is this working?", choices=choices)
        
        assert response.choice == "Yes"
        assert response.choice_index == 0
        assert response.confidence == 0.95
        assert response.is_choice_response


class TestProviderErrorHandling:
    """Test provider error handling."""
    
    def test_provider_not_found(self):
        """Test error when provider not found."""
        with pytest.raises(ValueError, match="Unknown model"):
            LLM.create("unknown-xyz-model")
    
    def test_invalid_model_name(self):
        """Test creating LLM with invalid model name."""
        with pytest.raises(ValueError, match="Unknown model"):
            LLM.create("not-a-real-model")


class TestProviderAutoDetection:
    """Test automatic provider detection."""
    
    def test_detect_openai_models(self):
        """Test detecting OpenAI models."""
        assert LLM.get_vendor_from_model("gpt-4o") == "openai"
        assert LLM.get_vendor_from_model("text-embedding-ada-002") == "openai"
        assert LLM.get_vendor_from_model("gpt-4o") == "openai"
        assert LLM.get_vendor_from_model("o1-mini") == "openai"
    
    def test_detect_anthropic_models(self):
        """Test detecting Anthropic models."""
        assert LLM.get_vendor_from_model("claude-3-5-sonnet-latest") == "anthropic"
        assert LLM.get_vendor_from_model("claude-3-5-haiku-latest") == "anthropic"
        assert LLM.get_vendor_from_model("claude-3-opus-latest") == "anthropic"
    
    def test_detect_google_models(self):
        """Test detecting Google models."""
        assert LLM.get_vendor_from_model("gemini-2.0-flash") == "google"
        assert LLM.get_vendor_from_model("gemini-1.5-pro") == "google"
        assert LLM.get_vendor_from_model("gemini-1.5-flash") == "google"
        assert LLM.get_vendor_from_model("text-embedding-004") == "google"
    
    def test_detect_ollama_models(self):
        """Test detecting Ollama models."""
        assert LLM.get_vendor_from_model("llama3.3") == "ollama"
        assert LLM.get_vendor_from_model("llama3.1") == "ollama"
        assert LLM.get_vendor_from_model("mistral") == "ollama"
        assert LLM.get_vendor_from_model("qwen2") == "ollama"
        assert LLM.get_vendor_from_model("gemma3") == "ollama"
    
    def test_detect_upstage_models(self):
        """Test detecting Upstage models."""
        assert LLM.get_vendor_from_model("solar-pro") == "upstage"
        assert LLM.get_vendor_from_model("solar-mini") == "upstage"
        assert LLM.get_vendor_from_model("embedding-query") == "upstage"
        assert LLM.get_vendor_from_model("embedding-passage") == "upstage"


class TestProviderIntegration:
    """Test provider integration scenarios."""
    
    def test_switching_providers(self):
        """Test switching between providers."""
        # Create different providers using mock instances
        llm1 = MockLLM(model="gpt-4-mock")
        llm2 = MockLLM(model="claude-3-mock")
        llm3 = MockLLM(model="gemini-pro-mock")
        
        # All should be MockLLM instances
        assert all(isinstance(llm, MockLLM) for llm in [llm1, llm2, llm3])
        
        # But with different models
        assert llm1.model == "gpt-4-mock"
        assert llm2.model == "claude-3-mock"
        assert llm3.model == "gemini-pro-mock"
    
    def test_provider_with_api_key(self):
        """Test provider with API key."""
        # Test creating with explicit API key
        with patch('pyhub.llm.OpenAILLM') as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance
            
            llm = LLM.create("gpt-4o", api_key="test-key-123")
            
            mock_openai.assert_called_once_with(model="gpt-4o", api_key="test-key-123")
            assert llm == mock_instance
    
    def test_provider_comparison(self):
        """Test comparing responses from different providers (all mock)."""
        providers = []
        models = ["gpt-4o", "claude-3", "gemini-pro"]
        
        for i, model in enumerate(models):
            llm = MockLLM(model=model)
            llm.set_mock_response(f"Provider {i+1} response")
            providers.append(llm)
        
        question = "What is AI?"
        responses = {}
        
        for llm in providers:
            response = llm.ask(question)
            responses[llm.model] = response.text
        
        # Check all responded
        assert len(responses) == 3
        assert responses["gpt-4o"] == "Provider 1 response: What is AI?"
        assert responses["claude-3"] == "Provider 2 response: What is AI?"
        assert responses["gemini-pro"] == "Provider 3 response: What is AI?"


@pytest.mark.openai
class TestOpenAIProviderMocked:
    """Test OpenAI provider with mocks (no actual API calls)."""
    
    def test_openai_provider_detection(self):
        """Test OpenAI models are detected correctly."""
        openai_models = [
            "gpt-4o", "gpt-4o",
            "text-embedding-ada-002", "text-embedding-3-small"
        ]
        for model in openai_models:
            assert LLM.get_vendor_from_model(model) == "openai"
    
    def test_openai_provider_creation(self):
        """Test creating OpenAI provider."""
        with patch('pyhub.llm.OpenAILLM') as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance
            
            llm = LLM.create("gpt-4o")
            
            mock_openai.assert_called_once()
            assert llm == mock_instance


@pytest.mark.anthropic
class TestAnthropicProviderMocked:
    """Test Anthropic provider with mocks (no actual API calls)."""
    
    def test_anthropic_provider_detection(self):
        """Test Anthropic models are detected correctly."""
        anthropic_models = [
            "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest",
            "claude-3-opus-latest"
        ]
        for model in anthropic_models:
            assert LLM.get_vendor_from_model(model) == "anthropic"
    
    def test_anthropic_provider_creation(self):
        """Test creating Anthropic provider."""
        with patch('pyhub.llm.AnthropicLLM') as mock_anthropic:
            mock_instance = Mock()
            mock_anthropic.return_value = mock_instance
            
            llm = LLM.create("claude-3-5-sonnet-latest")
            
            mock_anthropic.assert_called_once()
            assert llm == mock_instance


@pytest.mark.google
class TestGoogleProviderMocked:
    """Test Google provider with mocks (no actual API calls)."""
    
    def test_google_provider_detection(self):
        """Test Google models are detected correctly."""
        google_models = [
            "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash",
            "text-embedding-004"
        ]
        for model in google_models:
            assert LLM.get_vendor_from_model(model) == "google"


@pytest.mark.ollama
class TestOllamaProviderMocked:
    """Test Ollama provider with mocks (no actual API calls)."""
    
    def test_ollama_provider_detection(self):
        """Test Ollama models are detected correctly."""
        ollama_models = [
            "llama3.3", "llama3.1", "llama3.2",
            "mistral", "qwen2", "gemma3"
        ]
        for model in ollama_models:
            assert LLM.get_vendor_from_model(model) == "ollama"


class TestProviderFeatures:
    """Test provider-specific features with mocks."""
    
    def test_max_tokens_defaults(self):
        """Test default max_tokens for different providers."""
        # Mock providers would have different defaults
        llm = MockLLM(model="mock-model", max_tokens=1000)
        assert llm.max_tokens == 1000
        
        # Test without explicit max_tokens
        llm2 = MockLLM(model="mock-model")
        assert llm2.max_tokens == 1000  # Default from BaseLLM
    
    def test_temperature_defaults(self):
        """Test default temperature for different providers."""
        llm = MockLLM(model="mock-model")
        assert llm.temperature == 0.2  # Default from BaseLLM
        
        # Custom temperature
        llm2 = MockLLM(model="mock-model", temperature=0.7)
        assert llm2.temperature == 0.7
