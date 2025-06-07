"""Integration tests using mock providers."""

import pytest
from unittest.mock import patch
from pyhub.llm import LLM
from pyhub.llm.factory import LLMFactory
from pyhub.llm.providers.mock import MockLLM
from pyhub.llm.cache import MemoryCache, FileCache
from pyhub.llm.types import Message, Reply
from pyhub.llm.templates import TemplateEngine
from pyhub.llm.settings import Settings


class TestFactoryIntegration:
    """Test factory integration with mock provider."""
    
    def setup_method(self):
        """Register mock provider before each test."""
        LLMFactory.register_provider("mock", MockLLM)
    
    def test_create_mock_llm(self):
        """Test creating a mock LLM through factory."""
        # Need to mock the detect provider to return "mock"
        with patch.object(LLMFactory, '_detect_provider', return_value="mock"):
            llm = LLM.create("mock-model")
            assert isinstance(llm, MockLLM)
            assert llm.model == "mock-model"
    
    def test_create_with_cache(self, memory_cache):
        """Test creating LLM with cache."""
        with patch.object(LLMFactory, '_detect_provider', return_value="mock"):
            llm = LLM.create("mock-model", cache=memory_cache)
            assert llm.cache == memory_cache
    
    def test_create_with_settings(self):
        """Test creating LLM with settings."""
        settings = Settings()
        settings.set("mock", {"api_key": "test-key"})
        
        with patch.object(LLMFactory, '_detect_provider', return_value="mock"):
            with patch.object(LLMFactory, 'get_settings', return_value=settings):
                llm = LLM.create("mock-model")
                assert isinstance(llm, MockLLM)


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
        assert "".join(chunks).strip() == "Mock response: Tell me a story"
    
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
        assert "".join(chunks).strip() == "Mock response: Stream async"


class TestCacheIntegration:
    """Test cache integration with LLM."""
    
    def test_memory_cache_integration(self, memory_cache):
        """Test LLM with memory cache."""
        llm = MockLLM(model="mock-model", cache=memory_cache)
        
        # Make same request twice
        response1 = llm.ask("Cached question", save_history=False)
        response2 = llm.ask("Cached question", save_history=False)
        
        # Both should return same response (mock doesn't use cache, but structure is there)
        assert response1.text == response2.text
        
        # Test cache directly
        memory_cache.set("test_key", "test_value")
        assert memory_cache.get("test_key") == "test_value"
    
    def test_file_cache_integration(self, temp_cache_dir):
        """Test LLM with file cache."""
        file_cache = FileCache(str(temp_cache_dir))
        llm = MockLLM(model="mock-model", cache=file_cache)
        
        # Make request
        response = llm.ask("File cached question", save_history=False)
        assert response.text == "Mock response: File cached question"
        
        # Test file cache directly
        file_cache.set("test_key", {"data": "test"})
        assert file_cache.get("test_key") == {"data": "test"}


class TestTemplateIntegration:
    """Test template engine integration."""
    
    def test_llm_with_templates(self, tmp_path):
        """Test LLM with template engine."""
        # Create template directory and file
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        template_file = template_dir / "question.j2"
        template_file.write_text("Question about {{ topic }}: {{ question }}")
        
        # Create LLM with template engine
        engine = TemplateEngine(str(template_dir))
        llm = MockLLM(model="mock-model", template_engine=engine)
        
        # Render template
        rendered = engine.render("question.j2", {
            "topic": "AI",
            "question": "What is machine learning?"
        })
        
        assert rendered == "Question about AI: What is machine learning?"
        
        # Use LLM
        response = llm.ask(rendered)
        assert "Question about AI" in response.text


class TestChainIntegration:
    """Test LLM chaining integration."""
    
    def test_chain_multiple_llms(self):
        """Test chaining multiple LLMs."""
        llm1 = MockLLM(model="mock-1", output_key="step1")
        llm1.set_mock_response("Step 1 complete")
        
        llm2 = MockLLM(model="mock-2", output_key="step2")
        llm2.set_mock_response("Step 2 complete")
        
        llm3 = MockLLM(model="mock-3", output_key="step3")
        llm3.set_mock_response("Final step")
        
        # Create chain
        chain = llm1 | llm2 | llm3
        
        # Execute chain
        result = chain.ask("Start")
        
        # Check results
        assert len(result.reply_list) == 3
        assert result.values["step1"] == "Step 1 complete: Start"
        assert result.values["step2"] == "Step 2 complete: Step 1 complete: Start"
        assert result.values["step3"] == "Final step: Step 2 complete: Step 1 complete: Start"
    
    @pytest.mark.asyncio
    async def test_async_chain(self):
        """Test async chaining."""
        llm1 = MockLLM(model="async-1")
        llm2 = MockLLM(model="async-2")
        
        chain = llm1 | llm2
        
        result = await chain.ask_async("Async start")
        assert len(result.reply_list) == 2
        assert "Mock response: Mock response: Async start" in result.text


class TestSettingsIntegration:
    """Test settings integration."""
    
    def test_settings_from_env(self, monkeypatch):
        """Test loading settings from environment."""
        monkeypatch.setenv("PYHUB_LLM_CACHE_DIR", "/tmp/test-cache")
        monkeypatch.setenv("PYHUB_LLM_MOCK_API_KEY", "env-api-key")
        
        settings = Settings()
        assert settings.get("cache_dir") == "/tmp/test-cache"
        assert settings.get_api_key("mock") == "env-api-key"
    
    def test_settings_priority(self, monkeypatch):
        """Test settings priority (env > config)."""
        monkeypatch.setenv("PYHUB_LLM_TEST_VALUE", "from-env")
        
        settings = Settings()
        settings.set("test_value", "from-config")
        
        # Environment should take priority
        assert settings.get("test_value") == "from-env"


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_provider_not_found(self):
        """Test error when provider not found."""
        from pyhub.llm.exceptions import ProviderNotFoundError
        
        with pytest.raises(ProviderNotFoundError):
            LLM.create("unknown-model-xyz")
    
    def test_missing_api_key(self):
        """Test handling missing API key."""
        with patch.object(LLMFactory, '_detect_provider', return_value="mock"):
            # Should create without error (mock doesn't require API key)
            llm = LLM.create("mock-model")
            assert isinstance(llm, MockLLM)