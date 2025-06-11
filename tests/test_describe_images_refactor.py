"""Test refactored describe_images method."""

from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from pyhub.llm.base import DescribeImageRequest
from pyhub.llm.mock import MockLLM
from pyhub.llm.types import Reply


def create_test_image(color="red", text="Test"):
    """테스트용 이미지 생성"""
    img = Image.new("RGB", (100, 50), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), text, fill="white")

    # BytesIO로 반환
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    img_bytes.name = f"{text.lower()}.png"
    return img_bytes


class TestDescribeImagesRefactor:
    """Test refactored describe_images functionality."""

    def test_single_image_string_path(self, tmp_path):
        """Test with single image path as string."""
        llm = MockLLM(model="mock-model", response="Image analysis complete.")

        # 임시 이미지 파일 생성
        img_path = tmp_path / "test.png"
        img = create_test_image()
        img_path.write_bytes(img.getvalue())

        response = llm.describe_images(str(img_path))

        assert isinstance(response, Reply)
        assert "analysis" in response.text or "Mock" in response.text

    def test_single_image_path_object(self, tmp_path):
        """Test with single image as Path object."""
        llm = MockLLM(model="mock-model")

        # 임시 이미지 파일 생성
        img_path = tmp_path / "test.png"
        img = create_test_image()
        img_path.write_bytes(img.getvalue())

        response = llm.describe_images(img_path)

        assert isinstance(response, Reply)
        assert "Mock response:" in response.text

    def test_single_image_io_object(self):
        """Test with single image as IO object."""
        llm = MockLLM(model="mock-model")

        img_io = create_test_image()
        response = llm.describe_images(img_io)

        assert isinstance(response, Reply)
        assert "Mock response:" in response.text

    def test_multiple_images_list(self, tmp_path):
        """Test with multiple images as list."""
        llm = MockLLM(model="mock-model")

        # 여러 이미지 생성
        images = []
        for i in range(3):
            img_path = tmp_path / f"test_{i}.png"
            img = create_test_image(color=["red", "green", "blue"][i])
            img_path.write_bytes(img.getvalue())
            images.append(str(img_path))

        responses = llm.describe_images(images)

        assert isinstance(responses, list)
        assert len(responses) == 3
        for resp in responses:
            assert isinstance(resp, Reply)
            assert "Mock response:" in resp.text

    def test_custom_prompt(self, tmp_path):
        """Test with custom prompt."""
        llm = MockLLM(model="mock-model")

        img_path = tmp_path / "test.png"
        img = create_test_image()
        img_path.write_bytes(img.getvalue())

        custom_prompt = "What color is this?"
        response = llm.describe_images(str(img_path), prompt=custom_prompt)

        assert isinstance(response, Reply)
        assert custom_prompt in response.text

    def test_custom_settings(self, tmp_path):
        """Test with custom temperature and max_tokens."""
        llm = MockLLM(model="mock-model", temperature=0.7, max_tokens=500)

        img_path = tmp_path / "test.png"
        img = create_test_image()
        img_path.write_bytes(img.getvalue())

        # 커스텀 설정으로 호출
        response = llm.describe_images(
            str(img_path), system_prompt="You are a color expert.", temperature=0.3, max_tokens=100
        )

        assert isinstance(response, Reply)
        # MockLLM은 설정 변경을 실제로 반영하지 않지만, 에러 없이 동작해야 함

    def test_legacy_describe_image_request(self, tmp_path):
        """Test with legacy DescribeImageRequest."""
        llm = MockLLM(model="mock-model")

        img_path = tmp_path / "test.png"
        img = create_test_image()
        img_path.write_bytes(img.getvalue())

        request = DescribeImageRequest(
            image=str(img_path),
            image_path=str(img_path),
            system_prompt="You are an art expert.",
            user_prompt="Analyze this artwork.",
            temperature=0.5,
            max_tokens=200,
        )

        response = llm.describe_images(request)

        assert isinstance(response, Reply)
        assert "Analyze this artwork." in response.text

    def test_mixed_input_types(self, tmp_path):
        """Test with mixed input types (string, Path, DescribeImageRequest)."""
        llm = MockLLM(model="mock-model")

        # 여러 타입의 이미지 준비
        img_path1 = tmp_path / "test1.png"
        img_path2 = tmp_path / "test2.png"

        img1 = create_test_image(color="red")
        img2 = create_test_image(color="blue")

        img_path1.write_bytes(img1.getvalue())
        img_path2.write_bytes(img2.getvalue())

        request = DescribeImageRequest(
            image=str(img_path2),
            image_path=str(img_path2),
            system_prompt="Focus on colors.",
            user_prompt="What color do you see?",
        )

        # 혼합된 타입으로 호출
        responses = llm.describe_images([str(img_path1), request])

        assert isinstance(responses, list)
        assert len(responses) == 2
        assert "Describe this image in detail." in responses[0].text
        assert "What color do you see?" in responses[1].text

    def test_with_history(self, tmp_path):
        """Test with conversation history."""
        llm = MockLLM(model="mock-model")

        img_path = tmp_path / "test.png"
        img = create_test_image()
        img_path.write_bytes(img.getvalue())

        # 히스토리 초기화
        llm.clear()
        assert len(llm.history) == 0

        # use_history=True로 호출
        response = llm.describe_images(str(img_path), use_history=True)

        assert isinstance(response, Reply)
        assert len(llm.history) == 2  # user + assistant messages

    def test_with_cache_injection(self, tmp_path):
        """Test with cache injection."""
        from pyhub.llm.cache import MemoryCache
        cache = MemoryCache()
        # Create LLM with cache injected
        llm = MockLLM(model="mock-model", cache=cache)

        img_path = tmp_path / "test.png"
        img = create_test_image()
        img_path.write_bytes(img.getvalue())

        # 캐싱은 자동으로 활성화됨 (cache가 주입되었으므로)
        response = llm.describe_images(str(img_path))

        assert isinstance(response, Reply)
        # MockLLM은 캐싱을 실제로 구현하지 않지만 에러 없이 동작해야 함

    def test_empty_list(self):
        """Test with empty list."""
        llm = MockLLM(model="mock-model")

        responses = llm.describe_images([])

        assert isinstance(responses, list)
        assert len(responses) == 0

    @pytest.mark.asyncio
    async def test_describe_images_async(self, tmp_path):
        """Test async version of describe_images."""
        llm = MockLLM(model="mock-model")

        img_path = tmp_path / "test.png"
        img = create_test_image()
        img_path.write_bytes(img.getvalue())

        response = await llm.describe_images_async(str(img_path))

        assert isinstance(response, Reply)
        assert "Mock response:" in response.text

    @pytest.mark.asyncio
    async def test_describe_images_async_multiple(self, tmp_path):
        """Test async version with multiple images."""
        llm = MockLLM(model="mock-model")

        # 여러 이미지 생성
        images = []
        for i in range(3):
            img_path = tmp_path / f"test_{i}.png"
            img = create_test_image(text=f"Image{i}")
            img_path.write_bytes(img.getvalue())
            images.append(str(img_path))

        responses = await llm.describe_images_async(images)

        assert isinstance(responses, list)
        assert len(responses) == 3
        for resp in responses:
            assert isinstance(resp, Reply)
