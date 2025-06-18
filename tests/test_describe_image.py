"""Tests for describe_image methods."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from pyhub.llm.mock import MockLLM
from pyhub.llm.types import Reply


class TestDescribeImage:
    """Test describe_image functionality."""

    def test_describe_image_basic(self):
        """Test basic describe_image functionality."""
        llm = MockLLM(model="mock-model", response="This image shows a beautiful landscape with mountains and a lake.")

        # Test with string path
        response = llm.describe_image("test.jpg")
        assert isinstance(response, Reply)
        assert "landscape" in response.text

    def test_describe_image_with_path(self):
        """Test describe_image with Path object."""
        llm = MockLLM(model="mock-model", response="The image contains various objects.")

        # Test with Path object
        response = llm.describe_image(Path("test.jpg"))
        assert isinstance(response, Reply)
        assert "objects" in response.text

    def test_describe_image_with_io(self):
        """Test describe_image with IO object."""
        llm = MockLLM(model="mock-model", response="This is an uploaded image.")

        # Test with IO object
        io_obj = BytesIO(b"fake image data")
        io_obj.name = "test.jpg"
        response = llm.describe_image(io_obj)
        assert isinstance(response, Reply)
        assert "uploaded" in response.text

    def test_describe_image_custom_prompt(self):
        """Test describe_image with custom prompt."""
        llm = MockLLM(model="mock-model", response="I see blue, green, and red colors.")

        # Test with custom prompt
        custom_prompt = "What colors do you see?"
        response = llm.describe_image("test.jpg", prompt=custom_prompt)
        assert isinstance(response, Reply)
        assert "colors" in response.text

    def test_describe_image_with_options(self):
        """Test describe_image with additional options."""
        llm = MockLLM(model="mock-model", response="This artwork shows exceptional use of light and shadow.")

        # Test with additional options
        response = llm.describe_image(
            "test.jpg", system_prompt="You are an art critic.", temperature=0.8, max_tokens=500
        )
        assert isinstance(response, Reply)
        assert "artwork" in response.text

    def test_extract_text_from_image(self):
        """Test extract_text_from_image method."""
        llm = MockLLM(model="mock-model", response="The text in the image says: Hello World")

        response = llm.extract_text_from_image("test.jpg")
        assert isinstance(response, Reply)
        assert "text" in response.text.lower()

    def test_analyze_image_content(self):
        """Test analyze_image_content method."""
        llm = MockLLM(model="mock-model", response="The image contains 3 people, 2 cars, and a building.")

        response = llm.analyze_image_content("test.jpg")
        assert isinstance(response, Reply)
        assert "contains" in response.text

    @pytest.mark.asyncio
    async def test_describe_image_async(self):
        """Test async describe_image functionality."""
        llm = MockLLM(model="mock-model", response="This is an async image description.")

        # Test async method
        response = await llm.describe_image_async("test.jpg")
        assert isinstance(response, Reply)
        assert "async" in response.text

    @pytest.mark.asyncio
    async def test_describe_image_async_custom_prompt(self):
        """Test async describe_image with custom prompt."""
        llm = MockLLM(model="mock-model", response="I can see a table, chairs, and a lamp.")

        # Test with custom prompt
        custom_prompt = "What objects are visible?"
        response = await llm.describe_image_async("test.jpg", prompt=custom_prompt)
        assert isinstance(response, Reply)
        assert "table" in response.text or "objects" in response.text

    def test_describe_image_calls_ask_correctly(self):
        """Test that describe_image calls ask with correct parameters."""
        llm = MockLLM(model="mock-model")

        # Patch ask to capture the call
        with patch.object(llm, "ask") as mock_ask:
            mock_ask.return_value = Reply(text="Test response")

            # Store original values to verify they're restored
            original_system_prompt = llm.system_prompt
            original_temperature = llm.temperature
            original_max_tokens = llm.max_tokens

            llm.describe_image(
                "test.jpg",
                prompt="Custom prompt",
                system_prompt="Custom system",
                temperature=0.5,
                max_tokens=200,
                use_history=True,
            )

            # Check that ask was called with correct parameters
            mock_ask.assert_called_once_with(input="Custom prompt", files=["test.jpg"], use_history=True)

            # Check that original values were restored
            assert llm.system_prompt == original_system_prompt
            assert llm.temperature == original_temperature
            assert llm.max_tokens == original_max_tokens

    def test_describe_image_with_history(self):
        """Test describe_image with history management."""
        llm = MockLLM(model="mock-model")

        # Clear history first
        llm.clear()

        # Call with use_history=False (default)
        _response1 = llm.describe_image("test1.jpg")
        assert len(llm.history) == 0  # No history saved

        # Call with use_history=True
        _response2 = llm.describe_image("test2.jpg", use_history=True)
        assert len(llm.history) == 2  # User and assistant messages
