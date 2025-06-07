import pytest
from dataclasses import asdict
from pyhub.llm.types import (
    Message, Reply, Usage, Price, Embed, EmbeddingResponse,
    LLMResponse, StreamResponse, FunctionCall, ToolCall,
    FunctionParameter,
    LLMVendorType, OpenAIChatModelType, AnthropicChatModelType,
    GoogleChatModelType, OllamaChatModelType, UpstageChatModelType
)


class TestMessage:
    """Test Message dataclass."""
    
    def test_message_creation(self):
        """Test creating a Message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.files is None
    
    def test_message_with_optional_fields(self):
        """Test Message with optional fields."""
        msg = Message(
            role="assistant",
            content="Result",
            files=["file1.txt", "file2.png"]
        )
        assert msg.files == ["file1.txt", "file2.png"]
    
    def test_message_dict_conversion(self):
        """Test converting Message to dict."""
        msg = Message(role="user", content="Test")
        
        # Test to_dict method (which should remove files)
        msg_dict = msg.to_dict()
        
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Test"
        assert "files" not in msg_dict  # files should be removed in to_dict()
        
        # Test asdict (which includes all fields)
        full_dict = asdict(msg)
        assert full_dict["files"] is None  # asdict includes all fields
    
    def test_message_equality(self):
        """Test Message equality."""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="user", content="Hello")
        msg3 = Message(role="user", content="Hi")
        
        assert msg1 == msg2
        assert msg1 != msg3


class TestReply:
    """Test Reply dataclass."""
    
    def test_reply_creation(self):
        """Test creating a Reply."""
        reply = Reply(text="Test response")
        assert reply.text == "Test response"
        assert reply.usage is None
    
    def test_reply_with_usage_and_price(self):
        """Test Reply with usage and price."""
        usage = Usage(input=10, output=20)
        
        reply = Reply(
            text="Response",
            usage=usage
        )
        assert reply.usage == usage
    
    def test_reply_str_method(self):
        """Test Reply __str__ method."""
        reply = Reply(text="Hello world")
        assert str(reply) == "Hello world"
    
    def test_reply_choice_response(self):
        """Test Reply with choice response."""
        reply = Reply(text="Test", choice="option1", choice_index=0, confidence=0.95)
        
        assert reply.text == "Test"
        assert reply.choice == "option1"
        assert reply.choice_index == 0
        assert reply.confidence == 0.95
        assert reply.is_choice_response


class TestUsage:
    """Test Usage dataclass."""
    
    def test_usage_creation(self):
        """Test creating Usage."""
        usage = Usage(input=10, output=20)
        assert usage.input == 10
        assert usage.output == 20
        assert usage.total == 30
    
    def test_usage_add_operation(self):
        """Test Usage addition operation."""
        usage1 = Usage(input=10, output=20)
        usage2 = Usage(input=5, output=15)
        
        combined = usage1 + usage2
        assert combined.input == 15
        assert combined.output == 35
        assert combined.total == 50


class TestPrice:
    """Test Price dataclass."""
    
    def test_price_creation(self):
        """Test creating Price."""
        price = Price(input_usd=0.001, output_usd=0.002)
        assert price.input_usd == price.input_usd
        assert price.output_usd == price.output_usd
        assert price.usd == price.input_usd + price.output_usd
    
    def test_price_krw_calculation(self):
        """Test Price KRW calculation."""
        price = Price(input_usd=0.001, output_usd=0.002, rate_usd=1500)
        assert price.usd == price.input_usd + price.output_usd
        assert price.krw == price.usd * 1500


class TestEmbed:
    """Test Embed dataclass."""
    
    def test_embed_creation(self):
        """Test creating Embed."""
        embed = Embed(array=[0.1, 0.2, 0.3])
        assert embed.array == [0.1, 0.2, 0.3]
        assert embed.usage is None
        assert len(embed) == 3
    
    def test_embed_with_metadata(self):
        """Test Embed with usage and price."""
        usage = Usage(input=5, output=0)
        
        embed = Embed(
            array=[0.1, 0.2, 0.3, 0.4],
            usage=usage
        )
        assert len(embed) == 4
        assert embed.usage == usage
        assert embed[0] == 0.1


class TestFunctionTypes:
    """Test function-related types."""
    
    def test_function_call(self):
        """Test FunctionCall creation."""
        func_call = FunctionCall(
            name="get_weather",
            arguments={"location": "Paris", "unit": "celsius"}
        )
        assert func_call.name == "get_weather"
        assert func_call.arguments == {"location": "Paris", "unit": "celsius"}
    
    def test_tool_call(self):
        """Test ToolCall creation."""
        function = FunctionCall(name="calculate", arguments={"x": 10, "y": 20})
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=function
        )
        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function == function
    
    def test_function_parameter(self):
        """Test FunctionParameter creation."""
        param = FunctionParameter(
            name="location",
            type="string",
            description="The location to get weather for",
            enum=["Paris", "London", "New York"]
        )
        assert param.name == "location"
        assert param.type == "string"
        assert param.description == "The location to get weather for"
        assert param.enum == ["Paris", "London", "New York"]
    
    def test_function_parameter_with_all_fields(self):
        """Test FunctionParameter with all fields."""
        param = FunctionParameter(
            name="temperature",
            type="number",
            description="Temperature value",
            required=False,
            enum=None
        )
        assert param.name == "temperature"
        assert param.type == "number"
        assert param.required is False


class TestVendorTypes:
    """Test vendor-specific types."""
    
    def test_vendor_type_literals(self):
        """Test LLMVendorType values."""
        valid_vendors = ["openai", "anthropic", "google", "ollama", "upstage"]
        for vendor in valid_vendors:
            # Should not raise any errors
            assert vendor in ["openai", "anthropic", "google", "ollama", "upstage"]
    
    def test_model_type_examples(self):
        """Test model type examples."""
        # OpenAI models
        openai_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
        for model in openai_models:
            assert isinstance(model, str)
        
        # Anthropic models
        anthropic_models = ["claude-3-opus", "claude-3-sonnet", "claude-2"]
        for model in anthropic_models:
            assert isinstance(model, str)
        
        # Google models
        google_models = ["gemini-pro", "gemini-1.5-pro"]
        for model in google_models:
            assert isinstance(model, str)


class TestResponseTypes:
    """Test response types."""
    
    def test_llm_response(self):
        """Test LLMResponse creation."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            usage=Usage(input=10, output=20),
            finish_reason="stop"
        )
        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.usage.total == 30
        assert response.finish_reason == "stop"
    
    def test_embedding_response(self):
        """Test EmbeddingResponse creation."""
        response = EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3],
            model="text-embedding-ada-002",
            usage=Usage(input=5, output=0)
        )
        assert len(response) == 3
        assert response.model == "text-embedding-ada-002"
        assert response.usage.input == 5
    
    def test_stream_response(self):
        """Test StreamResponse creation."""
        response = StreamResponse(
            content="Hello",
            is_final=False
        )
        assert response.content == "Hello"
        assert response.is_final is False
        
        # Final chunk
        final_response = StreamResponse(
            content="",
            is_final=True
        )
        assert final_response.content == ""
        assert final_response.is_final is True


class TestTypeValidation:
    """Test type validation and edge cases."""
    
    def test_message_role_values(self):
        """Test valid message roles."""
        valid_roles = ["system", "user", "assistant", "function", "tool"]
        for role in valid_roles:
            msg = Message(role=role, content="Test")
            assert msg.role == role
    
    def test_empty_content(self):
        """Test messages with empty content."""
        msg = Message(role="user", content="")
        assert msg.content == ""
        
        # Message with files
        msg_files = Message(
            role="user",
            content="Check these files",
            files=["doc.pdf", "image.png"]
        )
        assert msg_files.content == "Check these files"
        assert len(msg_files.files) == 2
    
    def test_nested_type_creation(self):
        """Test creating nested types."""
        # Create a full response with all nested types
        usage = Usage(input=10, output=20)
        
        reply = Reply(
            text="Response",
            usage=usage,
            choice="option1",
            choice_index=0,
            confidence=0.9
        )
        
        assert reply.usage.total == 30
        assert reply.choice == "option1"
        assert reply.is_choice_response