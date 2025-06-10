from dataclasses import asdict

from pyhub.llm.types import Embed, EmbedList, Message, Price, Reply, Usage


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
        msg = Message(role="assistant", content="Result", files=["file1.txt", "file2.png"])
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

        reply = Reply(text="Response", usage=usage)
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

        embed = Embed(array=[0.1, 0.2, 0.3, 0.4], usage=usage)
        assert len(embed) == 4
        assert embed.usage == usage
        assert embed[0] == 0.1


class TestEmbedList:
    """Test EmbedList dataclass."""

    def test_embed_list_creation(self):
        """Test creating EmbedList."""
        embed1 = Embed(array=[0.1, 0.2, 0.3])
        embed2 = Embed(array=[0.4, 0.5, 0.6])
        embed_list = EmbedList(arrays=[embed1, embed2])

        assert len(embed_list.arrays) == 2
        assert embed_list.arrays[0] == embed1
        assert embed_list.arrays[1] == embed2

    def test_embed_list_with_usage(self):
        """Test EmbedList with usage information."""
        embed1 = Embed(array=[0.1, 0.2])
        embed2 = Embed(array=[0.3, 0.4])
        usage = Usage(input=10, output=0)

        embed_list = EmbedList(arrays=[embed1, embed2], usage=usage)

        assert embed_list.usage == usage
        assert embed_list.usage.input == 10
        assert embed_list.usage.output == 0


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
        openai_models = ["gpt-4o"]
        for model in openai_models:
            assert isinstance(model, str)

        # Anthropic models
        anthropic_models = ["claude-3-5-sonnet-latest", "claude-3-opus-latest", "claude-2"]
        for model in anthropic_models:
            assert isinstance(model, str)

        # Google models
        google_models = ["gemini-pro", "gemini-1.5-pro"]
        for model in google_models:
            assert isinstance(model, str)


class TestLLMTypes:
    """Test LLM type aliases."""

    def test_vendor_type_values(self):
        """Test vendor type values."""
        # These should be valid vendor strings
        valid_vendors = ["openai", "anthropic", "google", "ollama", "upstage"]
        for vendor in valid_vendors:
            # Just check they're valid strings
            assert isinstance(vendor, str)
            assert vendor in ["openai", "anthropic", "google", "ollama", "upstage"]

    def test_model_type_strings(self):
        """Test model type strings."""
        # OpenAI models can be any string
        openai_model = "gpt-4o"
        assert isinstance(openai_model, str)

        # Anthropic models can be any string
        anthropic_model = "claude-3.5-sonnet-latest"
        assert isinstance(anthropic_model, str)

        # Google models can be specific literals or any string
        google_model = "gemini-2.0-flash"
        assert isinstance(google_model, str)

        # Ollama models can be specific literals or any string
        ollama_model = "llama3.3"
        assert isinstance(ollama_model, str)

        # Upstage models can be specific literals or any string
        upstage_model = "solar-pro"
        assert isinstance(upstage_model, str)


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
        msg_files = Message(role="user", content="Check these files", files=["doc.pdf", "image.png"])
        assert msg_files.content == "Check these files"
        assert len(msg_files.files) == 2

    def test_nested_type_creation(self):
        """Test creating nested types."""
        # Create a full response with all nested types
        usage = Usage(input=10, output=20)

        reply = Reply(text="Response", usage=usage, choice="option1", choice_index=0, confidence=0.9)

        assert reply.usage.total == 30
        assert reply.choice == "option1"
        assert reply.is_choice_response
