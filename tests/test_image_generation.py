"""Test image generation functionality."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from pyhub.llm import AnthropicLLM, GoogleLLM, OllamaLLM, OpenAILLM, UpstageLLM
from pyhub.llm.types import ImageReply, Usage


class TestImageGeneration:
    """Test image generation capabilities."""

    def test_image_reply_type(self):
        """Test ImageReply dataclass."""
        usage = Usage(input=10, output=0)
        reply = ImageReply(
            url="https://example.com/image.png",
            size="1024x1024",
            model="dall-e-3",
            revised_prompt="A beautiful sunset over mountains with vibrant colors",
            usage=usage,
        )

        assert reply.url == "https://example.com/image.png"
        assert reply.size == "1024x1024"
        assert reply.model == "dall-e-3"
        assert reply.revised_prompt is not None
        assert reply.width == 1024
        assert reply.height == 1024
        assert reply.usage == usage
        assert str(reply) == "Image URL: https://example.com/image.png"

    def test_image_reply_base64(self):
        """Test ImageReply with base64 data."""
        reply = ImageReply(
            base64_data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
            size="256x256",
            model="dall-e-2",
        )

        assert reply.base64_data is not None
        assert reply.url is None
        assert reply.width == 256
        assert reply.height == 256


class TestOpenAIImageGeneration:
    """Test OpenAI DALL-E image generation."""

    @patch("openai.OpenAI")
    def test_generate_image_basic(self, mock_openai):
        """Test basic image generation with DALL-E 3."""
        # Mock setup
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock(url="https://dalle.com/image123.png", revised_prompt="Enhanced: A beautiful sunset")]
        mock_client.images.generate.return_value = mock_response

        # Test
        llm = OpenAILLM(model="dall-e-3", api_key="test-key")
        reply = llm.generate_image("A beautiful sunset")

        # Verify
        assert isinstance(reply, ImageReply)
        assert reply.url == "https://dalle.com/image123.png"
        assert reply.revised_prompt == "Enhanced: A beautiful sunset"
        assert reply.model == "dall-e-3"
        assert reply.size == "1024x1024"  # Default size

        # Check API call
        mock_client.images.generate.assert_called_once_with(
            model="dall-e-3",
            prompt="A beautiful sunset",
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url",
            style="vivid",
        )

    @patch("openai.OpenAI")
    def test_generate_image_with_options(self, mock_openai):
        """Test image generation with custom options."""
        # Mock setup
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock(url="https://dalle.com/image456.png", revised_prompt="Enhanced prompt")]
        mock_client.images.generate.return_value = mock_response

        # Test
        llm = OpenAILLM(model="dall-e-3", api_key="test-key")
        reply = llm.generate_image("A professional headshot", size="1024x1792", quality="hd", style="natural")

        # Verify
        assert reply.size == "1024x1792"
        assert reply.width == 1024
        assert reply.height == 1792

        # Check API call
        mock_client.images.generate.assert_called_once_with(
            model="dall-e-3",
            prompt="A professional headshot",
            size="1024x1792",
            quality="hd",
            n=1,
            response_format="url",
            style="natural",
        )

    def test_invalid_size_for_dalle3(self):
        """Test error when using invalid size for DALL-E 3."""
        llm = OpenAILLM(model="dall-e-3", api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            llm.generate_image("test", size="800x600")

        assert "Invalid size '800x600' for model 'dall-e-3'" in str(exc_info.value)
        assert "Valid sizes are: 1024x1024, 1024x1792, 1792x1024" in str(exc_info.value)

    def test_invalid_size_for_dalle2(self):
        """Test error when using invalid size for DALL-E 2."""
        llm = OpenAILLM(model="dall-e-2", api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            llm.generate_image("test", size="1024x1792")

        assert "Invalid size '1024x1792' for model 'dall-e-2'" in str(exc_info.value)
        assert "Valid sizes are: 256x256, 512x512, 1024x1024" in str(exc_info.value)

    def test_gpt_model_cannot_generate_images(self):
        """Test error when trying to generate images with GPT models."""
        llm = OpenAILLM(model="gpt-4o", api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            llm.generate_image("test")

        assert "Model 'gpt-4o' does not support image generation" in str(exc_info.value)
        assert "Use 'dall-e-3' or 'dall-e-2' instead" in str(exc_info.value)

    @patch("openai.AsyncOpenAI")
    async def test_generate_image_async(self, mock_async_openai):
        """Test async image generation with DALL-E 3."""
        # Mock setup
        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [
            Mock(url="https://dalle.com/async_image.png", revised_prompt="Enhanced: Async generated image")
        ]
        # Create async mock that returns the response when awaited
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.images.generate.return_value = future

        # Test
        llm = OpenAILLM(model="dall-e-3", api_key="test-key")
        reply = await llm.generate_image_async("An async generated image")

        # Verify
        assert isinstance(reply, ImageReply)
        assert reply.url == "https://dalle.com/async_image.png"
        assert reply.revised_prompt == "Enhanced: Async generated image"
        assert reply.model == "dall-e-3"

        # Check API call
        mock_client.images.generate.assert_called_once_with(
            model="dall-e-3",
            prompt="An async generated image",
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url",
            style="vivid",
        )

    def test_get_supported_image_sizes(self):
        """Test getting supported image sizes for a model."""
        llm_dalle3 = OpenAILLM(model="dall-e-3", api_key="test-key")
        sizes_dalle3 = llm_dalle3.get_supported_image_sizes()
        assert sizes_dalle3 == ["1024x1024", "1024x1792", "1792x1024"]

        llm_dalle2 = OpenAILLM(model="dall-e-2", api_key="test-key")
        sizes_dalle2 = llm_dalle2.get_supported_image_sizes()
        assert sizes_dalle2 == ["256x256", "512x512", "1024x1024"]

        llm_gpt = OpenAILLM(model="gpt-4", api_key="test-key")
        sizes_gpt = llm_gpt.get_supported_image_sizes()
        assert sizes_gpt == []


class TestUnsupportedProviders:
    """Test that other providers properly indicate no support."""

    def test_anthropic_not_supported(self):
        """Test Anthropic doesn't support image generation."""
        llm = AnthropicLLM(model="claude-3-opus", api_key="test-key")

        with pytest.raises(NotImplementedError) as exc_info:
            llm.generate_image("test")

        assert "Anthropic does not support image generation" in str(exc_info.value)

    async def test_anthropic_not_supported_async(self):
        """Test Anthropic doesn't support async image generation."""
        llm = AnthropicLLM(model="claude-3-opus", api_key="test-key")

        with pytest.raises(NotImplementedError) as exc_info:
            await llm.generate_image_async("test")

        assert "Anthropic does not support image generation" in str(exc_info.value)

    def test_google_not_supported(self):
        """Test Google doesn't support image generation (via Gemini API)."""
        llm = GoogleLLM(model="gemini-pro", api_key="test-key")

        with pytest.raises(NotImplementedError) as exc_info:
            llm.generate_image("test")

        assert "Google does not support image generation" in str(exc_info.value)

    async def test_google_not_supported_async(self):
        """Test Google doesn't support async image generation."""
        llm = GoogleLLM(model="gemini-pro", api_key="test-key")

        with pytest.raises(NotImplementedError) as exc_info:
            await llm.generate_image_async("test")

        assert "Google does not support image generation" in str(exc_info.value)

    def test_ollama_not_supported(self):
        """Test Ollama doesn't support image generation."""
        llm = OllamaLLM(model="llama2")

        with pytest.raises(NotImplementedError) as exc_info:
            llm.generate_image("test")

        assert "Ollama does not support image generation" in str(exc_info.value)

    async def test_ollama_not_supported_async(self):
        """Test Ollama doesn't support async image generation."""
        llm = OllamaLLM(model="llama2")

        with pytest.raises(NotImplementedError) as exc_info:
            await llm.generate_image_async("test")

        assert "Ollama does not support image generation" in str(exc_info.value)

    def test_upstage_not_supported(self):
        """Test Upstage doesn't support image generation."""
        llm = UpstageLLM(model="solar-mini", api_key="test-key")

        with pytest.raises(NotImplementedError) as exc_info:
            llm.generate_image("test")

        assert "Upstage does not support image generation" in str(exc_info.value)

    async def test_upstage_not_supported_async(self):
        """Test Upstage doesn't support async image generation."""
        llm = UpstageLLM(model="solar-mini", api_key="test-key")

        with pytest.raises(NotImplementedError) as exc_info:
            await llm.generate_image_async("test")

        assert "Upstage does not support image generation" in str(exc_info.value)


class TestImageGenerationCapabilities:
    """Test capability detection for image generation."""

    def test_supports_image_generation(self):
        """Test checking if a model supports image generation."""
        # Models that support image generation
        llm_dalle3 = OpenAILLM(model="dall-e-3", api_key="test-key")
        assert llm_dalle3.supports("image_generation") is True

        llm_dalle2 = OpenAILLM(model="dall-e-2", api_key="test-key")
        assert llm_dalle2.supports("image_generation") is True

        # Models that don't support image generation
        llm_gpt = OpenAILLM(model="gpt-4", api_key="test-key")
        assert llm_gpt.supports("image_generation") is False

        llm_claude = AnthropicLLM(model="claude-3", api_key="test-key")
        assert llm_claude.supports("image_generation") is False

    def test_capabilities_property(self):
        """Test capabilities property includes image generation info."""
        llm = OpenAILLM(model="dall-e-3", api_key="test-key")
        caps = llm.capabilities

        assert "image_generation" in caps
        assert caps["image_generation"]["supported"] is True
        assert caps["image_generation"]["sizes"] == ["1024x1024", "1024x1792", "1792x1024"]
        assert caps["image_generation"]["qualities"] == ["standard", "hd"]
        assert caps["image_generation"]["styles"] == ["vivid", "natural"]


class TestImageReplyMethods:
    """Test ImageReply methods."""

    def test_image_reply_print(self):
        """Test ImageReply print method."""
        reply = ImageReply(
            url="https://example.com/image.png", revised_prompt="Enhanced prompt", usage=Usage(input=10, output=0)
        )

        # Capture print output
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        reply.print()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        assert "Image URL: https://example.com/image.png" in output
        assert "Revised prompt: Enhanced prompt" in output
        assert "Usage:" in output

    @patch("pathlib.Path.write_bytes")
    @patch("httpx.get")
    def test_image_reply_save(self, mock_get, mock_write):
        """Test saving image from URL."""
        # Create a simple valid PNG image data
        # PNG header + minimal IHDR chunk
        png_data = (
            b"\x89PNG\r\n\x1a\n"  # PNG signature
            b"\x00\x00\x00\rIHDR"  # IHDR chunk length and type
            b"\x00\x00\x00\x01\x00\x00\x00\x01"  # 1x1 image
            b"\x08\x02\x00\x00\x00"  # 8-bit RGB
            b"\x90\x77\x53\xde"  # CRC
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x01"  # IDAT chunk
            b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND chunk
        )

        # Mock HTTP response
        mock_response = Mock()
        mock_response.content = png_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Mock PIL if available to avoid actual image processing
        with patch.dict("sys.modules", {"PIL": None}):
            reply = ImageReply(url="https://example.com/image.png")

            # Test save with auto filename
            path = reply.save()
            assert isinstance(path, Path)
            mock_get.assert_called_once_with("https://example.com/image.png")
            mock_write.assert_called_once_with(png_data)

    @patch("pathlib.Path.write_bytes")
    def test_image_reply_save_base64(self, mock_write):
        """Test saving image from base64."""
        import base64

        # Create valid PNG data
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00"
            b"\x90\x77\x53\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x01"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        base64_data = base64.b64encode(png_data).decode()

        # Mock PIL to avoid actual image processing
        with patch.dict("sys.modules", {"PIL": None}):
            reply = ImageReply(base64_data=base64_data)

            path = reply.save("test.png")
            assert str(path) == "test.png"
            mock_write.assert_called_once_with(png_data)

    @patch("httpx.get")
    def test_image_reply_to_pil_no_pillow(self, mock_get):
        """Test to_pil when Pillow is not installed."""
        reply = ImageReply(url="https://example.com/image.png")

        # Mock Pillow import error
        with patch.dict("sys.modules", {"PIL": None}):
            with pytest.raises(ImportError) as exc_info:
                reply.to_pil()

            assert "Pillow library is required" in str(exc_info.value)
            assert "pip install 'pyhub-llm[image]'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_image_reply_save_async(self):
        """Test async save."""
        # Create valid PNG data
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00"
            b"\x90\x77\x53\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x01"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        # Mock async HTTP client
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.content = png_data
            mock_response.raise_for_status = Mock()
            mock_client.get.return_value = mock_response

            # Mock PIL to avoid actual image processing
            with patch.dict("sys.modules", {"PIL": None}):
                reply = ImageReply(url="https://example.com/image.png")

                with patch("pathlib.Path.write_bytes"):
                    # Use a unique filename to avoid conflicts
                    import uuid

                    unique_filename = f"async_test_{uuid.uuid4().hex[:8]}.png"
                    path = await reply.save_async(unique_filename)
                    assert str(path) == unique_filename

    def test_image_reply_save_to_bytesio(self):
        """Test saving to BytesIO."""
        from io import BytesIO

        # Create valid PNG data
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82"

        # Test with URL
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.content = png_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            reply = ImageReply(url="https://example.com/image.png")
            buffer = BytesIO()

            result = reply.save(buffer)

            # Should return the same buffer
            assert result is buffer

            # Check content
            buffer.seek(0)
            assert buffer.read() == png_data

    def test_image_reply_save_to_file_like_object(self):
        """Test saving to file-like object."""

        # Create mock file-like object
        class MockFile:
            def __init__(self):
                self.data = b""

            def write(self, data):
                self.data += data
                return len(data)

        # Test with base64 data
        import base64

        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8/x8AAQEBAAr7Hg0AAAAASUVORK5CYII="
        png_data = base64.b64decode(base64_data)

        reply = ImageReply(base64_data=base64_data)
        mock_file = MockFile()

        result = reply.save(mock_file)

        # Should return the file-like object
        assert result is mock_file
        assert mock_file.data == png_data

    @pytest.mark.asyncio
    async def test_image_reply_save_async_to_bytesio(self):
        """Test async saving to BytesIO."""
        from io import BytesIO

        # Create valid PNG data
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82"

        # Mock async HTTP client
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.content = png_data
            mock_response.raise_for_status = Mock()
            mock_client.get.return_value = mock_response

            reply = ImageReply(url="https://example.com/image.png")
            buffer = BytesIO()

            result = await reply.save_async(buffer)

            # Should return the same buffer
            assert result is buffer

            # Check content
            buffer.seek(0)
            assert buffer.read() == png_data

    @pytest.mark.asyncio
    async def test_image_reply_save_async_to_async_file_like(self):
        """Test async saving to async file-like object."""

        # Create mock async file-like object
        class AsyncMockFile:
            def __init__(self):
                self.data = b""

            async def awrite(self, data):
                self.data += data
                return len(data)

        import base64

        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8/x8AAQEBAAr7Hg0AAAAASUVORK5CYII="
        png_data = base64.b64decode(base64_data)

        reply = ImageReply(base64_data=base64_data)
        mock_file = AsyncMockFile()

        result = await reply.save_async(mock_file)

        # Should return the file-like object
        assert result is mock_file
        assert mock_file.data == png_data


class TestAsyncImageGeneration:
    """Test async image generation."""

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_generate_image_async_basic(self, mock_openai):
        """Test basic async image generation."""
        # Mock setup
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock(url="https://dalle.com/async-image.png", revised_prompt="Enhanced async prompt")]
        mock_client.images.generate.return_value = mock_response

        # Test
        llm = OpenAILLM(model="dall-e-3", api_key="test-key")
        reply = await llm.generate_image_async("Async test image")

        # Verify
        assert isinstance(reply, ImageReply)
        assert reply.url == "https://dalle.com/async-image.png"
        assert reply.revised_prompt == "Enhanced async prompt"

        # Check API call
        mock_client.images.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_not_supported_providers(self):
        """Test async image generation with unsupported providers."""
        providers = [
            (AnthropicLLM(model="claude-3", api_key="test"), "Anthropic"),
            (GoogleLLM(model="gemini-pro", api_key="test"), "Google"),
            (OllamaLLM(model="llama2"), "Ollama"),
        ]

        for llm, name in providers:
            with pytest.raises(NotImplementedError) as exc_info:
                await llm.generate_image_async("test")
            assert f"{name} does not support image generation" in str(exc_info.value)
