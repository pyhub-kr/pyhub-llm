import pytest
from unittest.mock import Mock, patch
from pyhub.llm import LLM
from pyhub.llm.factory import LLMFactory
from pyhub.llm.base import BaseLLM
from pyhub.llm.cache import MemoryCache
from pyhub.llm.exceptions import LLMError, ProviderNotFoundError


class TestLLMFactory:
    """Test LLMFactory functionality."""
    
    def test_create_method_exists(self):
        """Test that LLM.create method exists."""
        assert hasattr(LLM, 'create')
        assert callable(LLM.create)
    
    @patch('pyhub.llm.factory.LLMFactory.create')
    def test_llm_create_delegates_to_factory(self, mock_create):
        """Test that LLM.create delegates to LLMFactory.create."""
        mock_llm = Mock(spec=BaseLLM)
        mock_create.return_value = mock_llm
        
        result = LLM.create("gpt-4", api_key="test-key")
        
        mock_create.assert_called_once_with("gpt-4", api_key="test-key")
        assert result == mock_llm
    
    def test_detect_provider_openai(self):
        """Test provider detection for OpenAI models."""
        factory = LLMFactory()
        assert factory._detect_provider("gpt-4") == "openai"
        assert factory._detect_provider("gpt-3.5-turbo") == "openai"
        assert factory._detect_provider("text-embedding-ada-002") == "openai"
    
    def test_detect_provider_anthropic(self):
        """Test provider detection for Anthropic models."""
        factory = LLMFactory()
        assert factory._detect_provider("claude-3-opus") == "anthropic"
        assert factory._detect_provider("claude-3-sonnet") == "anthropic"
        assert factory._detect_provider("claude-2") == "anthropic"
    
    def test_detect_provider_google(self):
        """Test provider detection for Google models."""
        factory = LLMFactory()
        assert factory._detect_provider("gemini-pro") == "google"
        assert factory._detect_provider("gemini-1.5-pro") == "google"
    
    def test_detect_provider_ollama(self):
        """Test provider detection for Ollama models."""
        factory = LLMFactory()
        assert factory._detect_provider("llama2") == "ollama"
        assert factory._detect_provider("mistral") == "ollama"
        assert factory._detect_provider("qwen2") == "ollama"
    
    def test_detect_provider_upstage(self):
        """Test provider detection for Upstage models."""
        factory = LLMFactory()
        assert factory._detect_provider("solar-pro") == "upstage"
        assert factory._detect_provider("embedding-query") == "upstage"
    
    def test_detect_provider_unknown(self):
        """Test provider detection for unknown models."""
        factory = LLMFactory()
        with pytest.raises(ProviderNotFoundError):
            factory._detect_provider("unknown-model-xyz")
    
    @patch('pyhub.llm.factory.LLMFactory._providers', {})
    def test_register_provider(self):
        """Test provider registration."""
        class CustomLLM(BaseLLM):
            def ask(self, question: str, **kwargs) -> str:
                return "Custom response"
            
            def messages(self, messages, **kwargs) -> str:
                return "Custom messages"
            
            def embed(self, text: str, **kwargs):
                return Mock(embeddings=[[0.1, 0.2]])
        
        # Register provider
        LLMFactory.register_provider("custom", CustomLLM)
        
        assert "custom" in LLMFactory._providers
        assert LLMFactory._providers["custom"] == CustomLLM
    
    def test_create_with_settings(self):
        """Test LLM creation with settings integration."""
        # Mock settings
        mock_settings_instance = Mock()
        mock_settings_instance.get_api_key.return_value = "settings-api-key"
        mock_settings_instance.get.return_value = "~/.pyhub/cache"
        
        # Mock FileCache to avoid real file operations
        with patch('pyhub.llm.factory.FileCache') as mock_file_cache:
            mock_cache_instance = Mock()
            mock_file_cache.return_value = mock_cache_instance
            
            with patch.object(LLMFactory, '_settings', mock_settings_instance):
                with patch('pyhub.llm.factory.LLMFactory._detect_provider') as mock_detect:
                    mock_detect.return_value = "openai"
                    
                    with patch('pyhub.llm.factory.LLMFactory._providers') as mock_providers:
                        mock_llm_class = Mock()
                        mock_llm_instance = Mock(spec=BaseLLM)
                        mock_llm_class.return_value = mock_llm_instance
                        mock_providers.get.return_value = mock_llm_class
                        
                        # Create without explicit API key
                        result = LLMFactory.create("gpt-4")
                        
                        # Should use API key from settings
                        mock_settings_instance.get_api_key.assert_called_once_with("openai")
                        mock_llm_class.assert_called_once()
                        assert result == mock_llm_instance
    
    def test_create_with_cache(self, memory_cache):
        """Test LLM creation with cache."""
        with patch('pyhub.llm.factory.LLMFactory._detect_provider') as mock_detect:
            mock_detect.return_value = "openai"
            
            with patch('pyhub.llm.factory.LLMFactory._providers') as mock_providers:
                mock_llm_class = Mock()
                mock_llm_instance = Mock(spec=BaseLLM)
                mock_llm_class.return_value = mock_llm_instance
                mock_providers.get.return_value = mock_llm_class
                
                # Create with cache
                result = LLMFactory.create("gpt-4", cache=memory_cache)
                
                # Verify cache was passed
                call_kwargs = mock_llm_class.call_args[1]
                assert call_kwargs.get('cache') == memory_cache
    
    def test_create_with_custom_parameters(self):
        """Test LLM creation with custom parameters."""
        with patch('pyhub.llm.factory.LLMFactory._detect_provider') as mock_detect:
            mock_detect.return_value = "openai"
            
            with patch('pyhub.llm.factory.LLMFactory._providers') as mock_providers:
                mock_llm_class = Mock()
                mock_llm_instance = Mock(spec=BaseLLM)
                mock_llm_class.return_value = mock_llm_instance
                mock_providers.get.return_value = mock_llm_class
                
                # Create with custom parameters
                result = LLMFactory.create(
                    "gpt-4",
                    temperature=0.5,
                    max_tokens=1000,
                    system_prompt="You are helpful."
                )
                
                # Verify parameters were passed
                call_kwargs = mock_llm_class.call_args[1]
                assert call_kwargs.get('temperature') == 0.5
                assert call_kwargs.get('max_tokens') == 1000
                assert call_kwargs.get('system_prompt') == "You are helpful."
    
    def test_create_missing_provider_class(self):
        """Test error when provider class is not found."""
        with patch('pyhub.llm.factory.LLMFactory._detect_provider') as mock_detect:
            mock_detect.return_value = "openai"
            
            with patch('pyhub.llm.factory.LLMFactory._providers') as mock_providers:
                mock_providers.get.return_value = None
                
                with pytest.raises(ProviderNotFoundError):
                    LLMFactory.create("gpt-4")
    
    def test_list_providers(self):
        """Test listing available providers."""
        with patch('pyhub.llm.factory.LLMFactory._providers') as mock_providers:
            mock_providers.keys.return_value = ["openai", "anthropic", "google"]
            
            providers = LLMFactory.list_providers()
            assert providers == ["openai", "anthropic", "google"]
    
    def test_model_aliases(self):
        """Test model name aliases."""
        # Common aliases that should map to same provider
        aliases = [
            ("gpt-4-turbo", "openai"),
            ("gpt-4-1106-preview", "openai"),
            ("claude-3-opus-20240229", "anthropic"),
            ("gemini-1.5-pro-latest", "google"),
        ]
        
        factory = LLMFactory()
        for model, expected_provider in aliases:
            assert factory._detect_provider(model) == expected_provider
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-api-key'})
    def test_api_key_priority(self):
        """Test API key priority (explicit > env > settings)."""
        with patch('pyhub.llm.settings.Settings') as mock_settings_class:
            mock_settings = Mock()
            mock_settings.get_api_key.return_value = "settings-api-key"
            mock_settings_class.return_value = mock_settings
            
            with patch('pyhub.llm.factory.LLMFactory._detect_provider') as mock_detect:
                mock_detect.return_value = "openai"
                
                with patch('pyhub.llm.factory.LLMFactory._providers') as mock_providers:
                    mock_llm_class = Mock()
                    mock_llm_instance = Mock(spec=BaseLLM)
                    mock_llm_class.return_value = mock_llm_instance
                    mock_providers.get.return_value = mock_llm_class
                    
                    # Test explicit API key takes priority
                    LLMFactory.create("gpt-4", api_key="explicit-key")
                    call_kwargs = mock_llm_class.call_args[1]
                    assert call_kwargs.get('api_key') == "explicit-key"