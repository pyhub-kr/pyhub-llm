"""Test Django integration for ImageReply."""

from unittest.mock import Mock, patch

import pytest

from pyhub.llm.types import ImageReply


class TestDjangoIntegration:
    """Test Django-specific features."""

    def test_to_django_file_with_url(self):
        """Test converting ImageReply with URL to Django file."""
        # Mock Django imports
        mock_content_file = Mock()
        mock_content_file_class = Mock(return_value=mock_content_file)

        with patch.dict("sys.modules", {"django.core.files.base": Mock(ContentFile=mock_content_file_class)}):
            # Create ImageReply with URL
            reply = ImageReply(url="https://example.com/image.png")

            # Mock httpx response
            with patch("httpx.get") as mock_get:
                mock_response = Mock()
                mock_response.content = b"fake image data"
                mock_response.raise_for_status = Mock()
                mock_get.return_value = mock_response

                # Convert to Django file
                result = reply.to_django_file("test.png")

            # Verify ContentFile was called with correct data
            mock_content_file_class.assert_called_once()
            args, kwargs = mock_content_file_class.call_args
            assert args[0] == b"fake image data"
            assert kwargs["name"] == "test.png"
            assert result == mock_content_file

    def test_to_django_file_with_base64(self):
        """Test converting ImageReply with base64 to Django file."""
        # Mock Django imports
        mock_content_file = Mock()
        mock_content_file_class = Mock(return_value=mock_content_file)

        with patch.dict("sys.modules", {"django.core.files.base": Mock(ContentFile=mock_content_file_class)}):
            # Create ImageReply with base64 data
            import base64

            test_data = b"test image data"
            base64_data = base64.b64encode(test_data).decode("utf-8")
            reply = ImageReply(base64_data=base64_data)

            # Convert to Django file
            result = reply.to_django_file("test.jpg")

            # Verify ContentFile was called with correct data
            mock_content_file_class.assert_called_once()
            args, kwargs = mock_content_file_class.call_args
            assert args[0] == test_data
            assert kwargs["name"] == "test.jpg"
            assert result == mock_content_file

    def test_to_django_file_auto_filename(self):
        """Test auto-generating filename when not provided."""
        # Mock Django imports and uuid
        mock_content_file = Mock()
        mock_content_file_class = Mock(return_value=mock_content_file)

        with patch.dict("sys.modules", {"django.core.files.base": Mock(ContentFile=mock_content_file_class)}):
            with patch("uuid.uuid4") as mock_uuid:
                mock_uuid.return_value = Mock(hex="abcd1234efgh5678")

                # Mock PIL.Image.open to raise ImportError (Pillow not available)
                with patch("PIL.Image.open", side_effect=ImportError):
                    # Create ImageReply with base64 data
                    import base64

                    test_data = b"test image data"
                    base64_data = base64.b64encode(test_data).decode("utf-8")
                    reply = ImageReply(base64_data=base64_data)

                    # Convert without filename
                    reply.to_django_file()

                    # Verify filename was auto-generated
                    args, kwargs = mock_content_file_class.call_args
                    assert kwargs["name"].startswith("image_abcd1234")
                    assert kwargs["name"].endswith(".png")  # default extension

    def test_to_django_file_auto_extension_with_pillow(self):
        """Test auto-detecting file extension with Pillow."""
        # Mock Django imports
        mock_content_file = Mock()
        mock_content_file_class = Mock(return_value=mock_content_file)

        # Mock PIL Image
        mock_image = Mock()
        mock_image.format = "JPEG"
        mock_image_open = Mock(return_value=mock_image)

        with patch.dict("sys.modules", {"django.core.files.base": Mock(ContentFile=mock_content_file_class)}):
            with patch("PIL.Image.open", mock_image_open):
                with patch("uuid.uuid4") as mock_uuid:
                    mock_uuid.return_value = Mock(hex="abcd1234efgh5678")

                    # Create ImageReply
                    import base64

                    test_data = b"test image data"
                    base64_data = base64.b64encode(test_data).decode("utf-8")
                    reply = ImageReply(base64_data=base64_data)

                    # Convert without filename
                    reply.to_django_file()

                    # Verify extension was detected
                    args, kwargs = mock_content_file_class.call_args
                    assert kwargs["name"].endswith(".jpeg")

    def test_to_django_file_no_django_installed(self):
        """Test error when Django is not installed."""
        # Ensure Django is "not installed"
        with patch.dict("sys.modules", {"django.core.files.base": None}):
            reply = ImageReply(url="https://example.com/image.png")

            with pytest.raises(ImportError) as exc_info:
                reply.to_django_file()

            assert "Django is required" in str(exc_info.value)
            assert "pip install django" in str(exc_info.value)

    def test_to_django_file_no_image_data(self):
        """Test error when no image data is available."""
        # Mock Django imports
        with patch.dict("sys.modules", {"django.core.files.base": Mock()}):
            reply = ImageReply()  # No URL or base64 data

            with pytest.raises(ValueError) as exc_info:
                reply.to_django_file()

            assert "No image data available" in str(exc_info.value)

    @pytest.mark.parametrize("filename", ["custom.png", "photo.jpg", "image.webp"])
    def test_to_django_file_custom_filenames(self, filename):
        """Test with various custom filenames."""
        # Mock Django imports
        mock_content_file = Mock()
        mock_content_file_class = Mock(return_value=mock_content_file)

        with patch.dict("sys.modules", {"django.core.files.base": Mock(ContentFile=mock_content_file_class)}):
            import base64

            test_data = b"test image data"
            base64_data = base64.b64encode(test_data).decode("utf-8")
            reply = ImageReply(base64_data=base64_data)

            # Convert with custom filename
            reply.to_django_file(filename)

            # Verify filename was used
            args, kwargs = mock_content_file_class.call_args
            assert kwargs["name"] == filename
