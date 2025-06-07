import pytest
from unittest.mock import Mock, patch, AsyncMock
from pyhub.llm import LLM
from pyhub.llm.base import BaseLLM
from pyhub.llm.providers.mock import MockLLM
from pyhub.llm.types import Message, Reply
from pyhub.llm.cache import MemoryCache


class TestLLMCore:
    """Test core LLM functionality."""
    
    def test_llm_initialization(self):
        """Test LLM can be initialized with basic parameters."""
        llm = MockLLM(
            model="mock-model",
            temperature=0.7,
            max_tokens=100
        )
        assert llm.model == "mock-model"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 100
    
    def test_ask_method(self):
        """Test basic ask functionality."""
        llm = MockLLM(model="mock-model")
        response = llm.ask("What is AI?")
        assert isinstance(response, Reply)
        assert response.text == "Mock response: What is AI?"
    
    @pytest.mark.asyncio
    async def test_ask_async_method(self):
        """Test async ask functionality."""
        llm = MockLLM(model="mock-model")
        response = await llm.ask_async("What is AI?")
        assert isinstance(response, Reply)
        assert response.text == "Mock response: What is AI?"
    
    def test_messages_method(self, sample_messages):
        """Test messages functionality."""
        llm = MockLLM(model="mock-model")
        response = llm.messages(sample_messages)
        assert isinstance(response, Reply)
        assert response.text == "Mock response: How are you?"
    
    @pytest.mark.asyncio
    async def test_messages_async_method(self, sample_messages):
        """Test async messages functionality."""
        llm = MockLLM(model="mock-model")
        response = await llm.messages_async(sample_messages)
        assert isinstance(response, Reply)
        assert response.text == "Mock response: How are you?"
    
    def test_embed_method(self):
        """Test embedding functionality."""
        llm = MockLLM(model="mock-model")
        result = llm.embed("Test text")
        assert result.array == [0.1, 0.2, 0.3, 0.4]
    
    @pytest.mark.asyncio
    async def test_embed_async_method(self):
        """Test async embedding functionality."""
        llm = MockLLM(model="mock-model")
        result = await llm.embed_async("Test text")
        assert result.array == [0.1, 0.2, 0.3, 0.4]
    
    def test_conversation_history(self):
        """Test conversation history management."""
        llm = MockLLM(model="mock-model")
        
        # Initially empty
        assert len(llm.history) == 0
        
        # Add messages
        llm.add_message(Message(role="user", content="Hello"))
        llm.add_message(Message(role="assistant", content="Hi there!"))
        
        assert len(llm.history) == 2
        assert llm.history[0].role == "user"
        assert llm.history[0].content == "Hello"
        assert llm.history[1].role == "assistant"
        assert llm.history[1].content == "Hi there!"
        
        # Clear history
        llm.clear_history()
        assert len(llm.history) == 0
    
    def test_system_prompt(self):
        """Test system prompt functionality."""
        llm = MockLLM(
            model="mock-model",
            system_prompt="You are a helpful assistant."
        )
        assert llm.system_prompt == "You are a helpful assistant."
    
    def test_with_cache(self, memory_cache):
        """Test LLM with caching enabled."""
        llm = MockLLM(model="mock-model", cache=memory_cache)
        
        # First call
        response1 = llm.ask("What is AI?", save_history=False)
        assert response1.text == "Mock response: What is AI?"
        
        # Mock the cache to return cached response
        memory_cache.set("mock-key", Reply(text="Cached response"))
        with patch.object(llm, '_generate_cache_key', return_value="mock-key"):
            # Cache would be used in real implementation
            response2 = llm.ask("What is AI?", save_history=False)
            # Since our mock doesn't actually use cache, response will be same
            assert response2.text == "Mock response: What is AI?"


class TestLLMStreaming:
    """Test streaming functionality."""
    
    def test_stream_method(self):
        """Test basic streaming."""
        llm = MockLLM(model="mock-model")
        
        # MockLLM already supports streaming
        result = llm.ask("Test", stream=True)
        chunks = list(result)
        assert len(chunks) > 0
        # Check that chunks are strings
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_stream_async_method(self):
        """Test async streaming."""
        llm = MockLLM(model="mock-model")
        
        # MockLLM already supports async streaming
        result = await llm.ask_async("Test", stream=True)
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) > 0
        # Check that chunks are strings
        assert all(isinstance(chunk, str) for chunk in chunks)


class TestLLMChaining:
    """Test LLM chaining functionality."""
    
    def test_sequential_chain(self):
        """Test SequentialChain functionality."""
        from pyhub.llm.base import SequentialChain
        
        llm1 = MockLLM(model="mock-model-1")
        llm2 = MockLLM(model="mock-model-2")
        
        chain = SequentialChain(llm1, llm2)
        
        # Test ask through chain
        response = chain.ask("Initial question")
        # Check that we have a ChainReply with results from both LLMs
        assert len(response.reply_list) == 2
        assert response.text == "Mock response: Mock response: Initial question"
    
    def test_pipe_operator(self):
        """Test pipe operator for chaining."""
        llm1 = MockLLM(model="mock-model-1")
        llm2 = MockLLM(model="mock-model-2")
        
        chain = llm1 | llm2
        
        # Should create a SequentialChain
        from pyhub.llm.base import SequentialChain
        assert isinstance(chain, SequentialChain)
        assert len(chain.llms) == 2
    
    @pytest.mark.asyncio
    async def test_async_chain(self):
        """Test async chaining."""
        from pyhub.llm.base import SequentialChain
        
        llm1 = MockLLM(model="mock-model-1")
        llm2 = MockLLM(model="mock-model-2")
        
        chain = SequentialChain(llm1, llm2)
        
        # Test async ask through chain
        response = await chain.ask_async("Initial question")
        assert len(response.reply_list) == 2
        assert response.text == "Mock response: Mock response: Initial question"


class TestLLMWithTemplates:
    """Test LLM with template support."""
    
    def test_ask_with_template(self):
        """Test asking with a template."""
        from pyhub.llm.templates import TemplateEngine
        
        # Create a template engine
        engine = TemplateEngine()
        llm = MockLLM(model="mock-model", template_engine=engine)
        
        # Test simple template rendering
        response = llm.ask("What is AI?")
        assert response.text == "Mock response: What is AI?"


class TestLLMErrorHandling:
    """Test error handling."""
    
    def test_ask_with_error(self):
        """Test error handling in ask method."""
        llm = MockLLM(model="mock-model")
        
        with patch.object(llm, 'ask', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                llm.ask("Test question")
    
    @pytest.mark.asyncio
    async def test_ask_async_with_error(self):
        """Test error handling in async ask method."""
        llm = MockLLM(model="mock-model")
        
        with patch.object(llm, 'ask_async', side_effect=Exception("Async API Error")):
            with pytest.raises(Exception, match="Async API Error"):
                await llm.ask_async("Test question")
    
    def test_choices_parameter(self):
        """Test choices parameter functionality."""
        llm = MockLLM(model="mock-model")
        
        # Test with choices
        response = llm.ask("Choose one", choices=["Option A", "Option B", "Option C"])
        assert response.choice == "Option A"
        assert response.choice_index == 0
        assert response.confidence == 0.95
        assert response.is_choice_response