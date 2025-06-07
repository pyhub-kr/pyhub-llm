import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from pyhub.llm import LLM
from pyhub.llm.factory import LLMFactory
from pyhub.llm.providers.mock import MockLLM
from pyhub.llm.types import Message


class TestProviderRegistration:
    """Test provider registration and factory integration."""
    
    def test_register_mock_provider(self):
        """Test registering mock provider."""
        # Register the mock provider
        LLMFactory.register_provider("mock", MockLLM)
        
        # Check it's registered
        assert "mock" in LLMFactory.list_providers()
    
    def test_create_with_mock_provider(self):
        """Test creating LLM with mock provider."""
        # Register mock provider
        LLMFactory.register_provider("mock", MockLLM)
        
        # Mock the detect provider to return "mock"
        with patch.object(LLMFactory, '_detect_provider', return_value="mock"):
            llm = LLM.create("mock-model")
            assert isinstance(llm, MockLLM)
            assert llm.model == "mock-model"


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
        from pyhub.llm.exceptions import ProviderNotFoundError
        
        with pytest.raises(ProviderNotFoundError):
            LLM.create("unknown-xyz-model")
    
    def test_invalid_provider_class(self):
        """Test registering invalid provider class."""
        # Try to register a class that doesn't inherit from BaseLLM
        class InvalidProvider:
            pass
        
        # This should work - validation happens at runtime
        LLMFactory.register_provider("invalid", InvalidProvider)
        
        # But creating an instance should fail
        with patch.object(LLMFactory, '_detect_provider', return_value="invalid"):
            with pytest.raises(Exception):
                LLM.create("invalid-model")


class TestProviderDetection:
    """Test automatic provider detection."""
    
    def test_detect_openai_models(self):
        """Test detecting OpenAI models."""
        factory = LLMFactory()
        assert factory._detect_provider("gpt-4") == "openai"
        assert factory._detect_provider("gpt-3.5-turbo") == "openai"
        assert factory._detect_provider("text-embedding-ada-002") == "openai"
        assert factory._detect_provider("chatgpt-4o-latest") == "openai"
        assert factory._detect_provider("o1-mini") == "openai"
    
    def test_detect_anthropic_models(self):
        """Test detecting Anthropic models."""
        factory = LLMFactory()
        assert factory._detect_provider("claude-3-opus") == "anthropic"
        assert factory._detect_provider("claude-3-sonnet") == "anthropic"
        assert factory._detect_provider("claude-2") == "anthropic"
        assert factory._detect_provider("claude-3-5-sonnet-20241022") == "anthropic"
    
    def test_detect_google_models(self):
        """Test detecting Google models."""
        factory = LLMFactory()
        assert factory._detect_provider("gemini-pro") == "google"
        assert factory._detect_provider("gemini-1.5-pro") == "google"
        assert factory._detect_provider("gemini-2.0-flash") == "google"
        assert factory._detect_provider("text-embedding-004") == "google"
    
    def test_detect_ollama_models(self):
        """Test detecting Ollama models."""
        factory = LLMFactory()
        assert factory._detect_provider("llama2") == "ollama"
        assert factory._detect_provider("llama3.1") == "ollama"
        assert factory._detect_provider("mistral") == "ollama"
        assert factory._detect_provider("qwen2") == "ollama"
        assert factory._detect_provider("gemma3") == "ollama"
    
    def test_detect_upstage_models(self):
        """Test detecting Upstage models."""
        factory = LLMFactory()
        assert factory._detect_provider("solar-pro") == "upstage"
        assert factory._detect_provider("solar-mini") == "upstage"
        assert factory._detect_provider("embedding-query") == "upstage"
        assert factory._detect_provider("embedding-passage") == "upstage"


class TestProviderIntegration:
    """Test provider integration scenarios."""
    
    def test_switching_providers(self):
        """Test switching between providers."""
        # Register mock provider
        LLMFactory.register_provider("mock", MockLLM)
        
        # Create different "providers" (all mock for testing)
        with patch.object(LLMFactory, '_detect_provider', return_value="mock"):
            llm1 = LLM.create("gpt-4-mock")
            llm2 = LLM.create("claude-3-mock")
            llm3 = LLM.create("gemini-pro-mock")
        
        # All should be MockLLM instances
        assert all(isinstance(llm, MockLLM) for llm in [llm1, llm2, llm3])
        
        # But with different models
        assert llm1.model == "gpt-4-mock"
        assert llm2.model == "claude-3-mock"
        assert llm3.model == "gemini-pro-mock"
    
    def test_provider_with_settings(self):
        """Test provider with settings integration."""
        from pyhub.llm.settings import Settings
        
        # Create settings
        settings = Settings()
        settings.set("mock", {"api_key": "test-key-123"})
        
        # Register mock provider
        LLMFactory.register_provider("mock", MockLLM)
        
        with patch.object(LLMFactory, '_detect_provider', return_value="mock"):
            with patch.object(LLMFactory, 'get_settings', return_value=settings):
                llm = LLM.create("mock-model")
                assert isinstance(llm, MockLLM)
    
    def test_provider_comparison(self):
        """Test comparing responses from different providers (all mock)."""
        # Register mock provider
        LLMFactory.register_provider("mock", MockLLM)
        
        providers = []
        models = ["gpt-4", "claude-3", "gemini-pro"]
        
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
        assert responses["gpt-4"] == "Provider 1 response: What is AI?"
        assert responses["claude-3"] == "Provider 2 response: What is AI?"
        assert responses["gemini-pro"] == "Provider 3 response: What is AI?"


@pytest.mark.openai
class TestOpenAIProviderMocked:
    """Test OpenAI provider with mocks (no actual API calls)."""
    
    def test_openai_provider_detection(self):
        """Test OpenAI models are detected correctly."""
        factory = LLMFactory()
        openai_models = [
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
            "text-embedding-ada-002", "text-embedding-3-small"
        ]
        for model in openai_models:
            assert factory._detect_provider(model) == "openai"
    
    def test_openai_provider_not_available(self):
        """Test handling when OpenAI provider not available."""
        # When provider not registered, should raise error
        from pyhub.llm.exceptions import ProviderNotFoundError
        
        # Clear providers and try to create
        with patch.object(LLMFactory, '_providers', {}):
            with pytest.raises(ProviderNotFoundError):
                LLM.create("gpt-4")


@pytest.mark.anthropic
class TestAnthropicProviderMocked:
    """Test Anthropic provider with mocks (no actual API calls)."""
    
    def test_anthropic_provider_detection(self):
        """Test Anthropic models are detected correctly."""
        factory = LLMFactory()
        anthropic_models = [
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
            "claude-3-5-sonnet-20241022", "claude-2"
        ]
        for model in anthropic_models:
            assert factory._detect_provider(model) == "anthropic"


@pytest.mark.google
class TestGoogleProviderMocked:
    """Test Google provider with mocks (no actual API calls)."""
    
    def test_google_provider_detection(self):
        """Test Google models are detected correctly."""
        factory = LLMFactory()
        google_models = [
            "gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash",
            "text-embedding-004"
        ]
        for model in google_models:
            assert factory._detect_provider(model) == "google"


@pytest.mark.ollama
class TestOllamaProviderMocked:
    """Test Ollama provider with mocks (no actual API calls)."""
    
    def test_ollama_provider_detection(self):
        """Test Ollama models are detected correctly."""
        factory = LLMFactory()
        ollama_models = [
            "llama2", "llama3", "llama3.1", "llama3.2",
            "mistral", "qwen2", "gemma3"
        ]
        for model in ollama_models:
            assert factory._detect_provider(model) == "ollama"


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