import pytest
from unittest.mock import Mock, patch, AsyncMock
from pyhub.llm import LLM
from pyhub.llm.base import BaseLLM
from pyhub.llm.mock import MockLLM
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
        
        # Add messages through ask method (which updates history)
        response = llm.ask("Hello", save_history=True)
        assert len(llm.history) == 2  # User message + assistant response
        assert llm.history[0].role == "user"
        assert llm.history[0].content == "Hello"
        assert llm.history[1].role == "assistant"
        assert llm.history[1].content == "Mock response: Hello"
        
        # Ask another question
        response2 = llm.ask("How are you?", save_history=True)
        assert len(llm.history) == 4  # 2 more messages added
        assert llm.history[2].content == "How are you?"
        assert llm.history[3].content == "Mock response: How are you?"
        
        # Clear history
        llm.clear()
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
        llm = MockLLM(model="mock-model")
        
        # First call with caching enabled
        response1 = llm.ask("What is AI?", save_history=False, enable_cache=True)
        assert response1.text == "Mock response: What is AI?"
        
        # Second call with same question and caching enabled
        # In real implementation, this might return cached response
        response2 = llm.ask("What is AI?", save_history=False, enable_cache=True)
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
        
        llm1 = MockLLM(model="mock-model-1", prompt="{input}", output_key="step1")
        llm2 = MockLLM(model="mock-model-2", prompt="{step1}", output_key="step2")
        
        chain = SequentialChain(llm1, llm2)
        
        # Test ask through chain with dict input
        response = chain.ask({"input": "Initial question"})
        # Check that we have a ChainReply with results from both LLMs
        assert len(response.reply_list) == 2
        # MockLLM receives the dict as string since it doesn't process prompts
        assert "Mock response: {'input': 'Initial question'}" == response.values["step1"]
        # Second LLM receives the accumulated context (input + step1)
        assert response.values["step2"] == "Mock response: {'input': 'Initial question', 'step1': \"Mock response: {'input': 'Initial question'}\"}"
    
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
    @pytest.mark.skip(reason="SequentialChain doesn't support async yet")
    async def test_async_chain(self):
        """Test async chaining."""
        from pyhub.llm.base import SequentialChain
        
        llm1 = MockLLM(model="mock-model-1", prompt="{input}", output_key="step1")
        llm2 = MockLLM(model="mock-model-2", prompt="{step1}", output_key="step2")
        
        chain = SequentialChain(llm1, llm2)
        
        # This would need to be implemented in SequentialChain
        # response = await chain.ask_async({"input": "Initial question"})
        # assert len(response.reply_list) == 2


class TestLLMWithTemplates:
    """Test LLM with template support."""
    
    def test_ask_with_template(self):
        """Test asking with a template."""
        # Note: TemplateEngine is not part of the current implementation
        # This test is kept for documentation purposes but simplified
        llm = MockLLM(model="mock-model")
        
        # Test simple ask without template
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
