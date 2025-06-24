"""Tests for Ollama provider."""

from unittest.mock import AsyncMock, Mock, patch
import sys

import pytest

from pyhub.llm.types import Reply


def mock_ollama():
    """Helper to create mocked ollama module."""
    mock_ollama = Mock()
    
    # Mock client classes
    mock_ollama.Client = Mock()
    # AsyncClient should be a regular Mock that returns an async client instance
    mock_ollama.AsyncClient = Mock()
    
    # Mock response classes
    mock_ollama.ChatResponse = Mock
    mock_ollama.EmbedResponse = Mock
    mock_ollama.ListResponse = Mock
    
    return mock_ollama


class TestOllamaLLM:
    """Test Ollama LLM provider."""

    def test_initialization(self):
        """Test Ollama LLM initialization."""
        mock_ollama_module = mock_ollama()
        
        with patch.dict("sys.modules", {"ollama": mock_ollama_module}):
            # Clear cached module
            if "pyhub.llm.ollama" in sys.modules:
                del sys.modules["pyhub.llm.ollama"]
            
            from pyhub.llm.ollama import OllamaLLM
            
            # Test with default values
            llm = OllamaLLM()
            
            assert llm.model == "mistral:latest"  # Should add :latest
            assert llm.embedding_model == "nomic-embed-text:latest"
            assert llm.temperature == 0.2
            assert llm.output_key == "text"
            assert llm.timeout == 60
            
            # Test with custom values
            llm = OllamaLLM(
                model="llama3.1:70b",
                embedding_model="avr/sfr-embedding-mistral",
                temperature=0.5,
                base_url="http://localhost:11434",
                timeout=120
            )
            
            assert llm.model == "llama3.1:70b"  # Already has tag
            assert llm.embedding_model == "avr/sfr-embedding-mistral:latest"
            assert llm.temperature == 0.5
            assert llm.base_url == "http://localhost:11434"
            assert llm.timeout == 120

    def test_initialization_without_ollama(self):
        """Test initialization when ollama is not installed."""
        with patch.dict("sys.modules", {"ollama": None}):
            # Clear cached module
            if "pyhub.llm.ollama" in sys.modules:
                del sys.modules["pyhub.llm.ollama"]
            
            from pyhub.llm.ollama import OllamaLLM
            
            with pytest.raises(ImportError, match="ollama package not installed"):
                OllamaLLM(model="mistral")

    def test_initialization_from_settings(self):
        """Test initialization with base URL from settings."""
        mock_ollama_module = mock_ollama()
        
        with patch.dict("sys.modules", {"ollama": mock_ollama_module}):
            if "pyhub.llm.ollama" in sys.modules:
                del sys.modules["pyhub.llm.ollama"]
            
            with patch("pyhub.llm.settings.llm_settings") as mock_settings:
                mock_settings.ollama_base_url = "http://custom:11434"
                
                from pyhub.llm.ollama import OllamaLLM
                
                llm = OllamaLLM()
                
                assert llm.base_url == "http://custom:11434"

    def test_check_method(self):
        """Test check method for base URL validation."""
        mock_ollama_module = mock_ollama()
        
        # Mock the list response for model checking
        mock_client = Mock()
        mock_list_response = Mock()
        mock_list_response.models = [
            Mock(model="mistral:latest"),
            Mock(model="llama3.1:latest"),
            Mock(model="nomic-embed-text:latest")  # Add embedding model
        ]
        mock_client.list.return_value = mock_list_response
        mock_ollama_module.Client.return_value = mock_client
        
        with patch.dict("sys.modules", {"ollama": mock_ollama_module}):
            if "pyhub.llm.ollama" in sys.modules:
                del sys.modules["pyhub.llm.ollama"]
            
            from pyhub.llm.ollama import OllamaLLM
            
            # Test with valid base URL
            llm = OllamaLLM(base_url="http://localhost:11434")
            errors = llm.check()
            assert len(errors) == 0
            
            # Test without base URL (uses default)
            with patch("pyhub.llm.settings.llm_settings") as mock_settings:
                mock_settings.ollama_base_url = "http://localhost:11434"
                llm = OllamaLLM()
                errors = llm.check()
                assert len(errors) == 0
            
            # Test with no base URL - test model not found error instead
            # Create a new instance with a non-existent model
            llm = OllamaLLM(model="nonexistent:latest", base_url="http://localhost:11434")
            errors = llm.check()
            assert len(errors) == 1
            assert "nonexistent:latest" in errors[0]["msg"]
            assert "not found" in errors[0]["msg"]

    def test_ask_simple(self):
        """Test simple ask method."""
        mock_ollama_module = mock_ollama()
        mock_client = Mock()
        mock_ollama_module.Client.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.message.content = "Hello from Ollama!"
        # Ollama response has usage with prompt_tokens and completion_tokens
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 8
        
        mock_client.chat.return_value = mock_response
        
        with patch.dict("sys.modules", {"ollama": mock_ollama_module}):
            if "pyhub.llm.ollama" in sys.modules:
                del sys.modules["pyhub.llm.ollama"]
            
            from pyhub.llm.ollama import OllamaLLM
            
            # Test ask
            llm = OllamaLLM(model="mistral:latest", base_url="http://localhost:11434")
            reply = llm.ask("Hello Ollama")
            
            # Verify client creation - Ollama client doesn't take timeout in constructor
            mock_ollama_module.Client.assert_called_with(
                host="http://localhost:11434"
            )
            
            # Verify API call
            mock_client.chat.assert_called_once()
            call_kwargs = mock_client.chat.call_args[1]
            
            assert call_kwargs["model"] == "mistral:latest"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hello Ollama"}]
            assert call_kwargs["options"]["temperature"] == 0.2
            # Stream parameter is not explicitly set to False in non-streaming calls
            
            # Verify response
            assert isinstance(reply, Reply)
            assert reply.text == "Hello from Ollama!"
            assert reply.usage.input == 10
            assert reply.usage.output == 8

    def test_streaming(self):
        """Test streaming response."""
        mock_ollama_module = mock_ollama()
        mock_client = Mock()
        mock_ollama_module.Client.return_value = mock_client
        
        # Mock streaming chunks
        mock_chunks = [
            Mock(message=Mock(content="Hello"), usage=None),
            Mock(message=Mock(content=" from"), usage=None),
            Mock(message=Mock(content=" Ollama"), usage=None),
            Mock(message=Mock(content=""), usage=Mock(prompt_tokens=5, completion_tokens=3)),  # Final chunk with usage
        ]
        
        mock_client.chat.return_value = iter(mock_chunks)
        
        with patch.dict("sys.modules", {"ollama": mock_ollama_module}):
            if "pyhub.llm.ollama" in sys.modules:
                del sys.modules["pyhub.llm.ollama"]
            
            from pyhub.llm.ollama import OllamaLLM
            
            llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")
            chunks = list(llm.ask("Hello", stream=True))
            
            # Verify streaming was enabled
            call_kwargs = mock_client.chat.call_args[1]
            assert call_kwargs["stream"] is True
            
            # Verify we get Reply objects
            assert len(chunks) == 4
            assert all(isinstance(chunk, Reply) for chunk in chunks)
            
            # Check content
            assert chunks[0].text == "Hello"
            assert chunks[1].text == " from"
            assert chunks[2].text == " Ollama"
            
            # Last chunk should have usage
            assert chunks[3].text == ""
            # Usage values come from the Ollama response
            assert chunks[3].usage is not None
            assert chunks[3].usage.input == 5
            assert chunks[3].usage.output == 3

    def test_error_handling(self):
        """Test error handling."""
        mock_ollama_module = mock_ollama()
        mock_client = Mock()
        mock_ollama_module.Client.return_value = mock_client
        
        # Mock API error
        mock_client.chat.side_effect = Exception("Connection error")
        
        with patch.dict("sys.modules", {"ollama": mock_ollama_module}):
            if "pyhub.llm.ollama" in sys.modules:
                del sys.modules["pyhub.llm.ollama"]
            
            from pyhub.llm.ollama import OllamaLLM
            
            llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")
            
            # Ollama provider returns error in Reply instead of raising
            reply = llm.ask("Hello")
            assert "Error:" in reply.text
            assert "Connection error" in reply.text

    @pytest.mark.asyncio
    async def test_ask_async(self):
        """Test async ask method."""
        mock_ollama_module = mock_ollama()
        
        # Create a proper async mock client
        mock_async_client = Mock()
        # Important: chat needs to be an awaitable that returns a mock response
        async def mock_chat(**kwargs):
            # Mock response
            mock_response = Mock()
            mock_response.message = Mock(content="Async response")
            # Ollama response has usage with prompt_tokens and completion_tokens
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            return mock_response
            
        mock_async_client.chat = mock_chat
        
        # Return the mock client when AsyncClient is called
        mock_ollama_module.AsyncClient.return_value = mock_async_client
        
        with patch.dict("sys.modules", {"ollama": mock_ollama_module}):
            if "pyhub.llm.ollama" in sys.modules:
                del sys.modules["pyhub.llm.ollama"]
            
            from pyhub.llm.ollama import OllamaLLM
            
            llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")
            reply = await llm.ask_async("Hello async")
            
            # Verify async client creation - Ollama client doesn't take timeout in constructor
            mock_ollama_module.AsyncClient.assert_called_with(
                host="http://localhost:11434"
            )
            
            # Verify response
            assert isinstance(reply, Reply)
            assert reply.text == "Async response"
            assert reply.usage.total == 15