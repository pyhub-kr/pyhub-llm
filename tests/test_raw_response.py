"""
API raw response 기능 테스트
"""

import os
from unittest.mock import Mock, patch

import pytest

from pyhub.llm import LLM, OpenAILLM
from pyhub.llm.types import ImageReply, Reply


class TestRawResponse:
    """Raw response 기능 테스트"""

    def test_reply_has_raw_response_field(self):
        """Reply 객체에 raw_response 필드가 있는지 확인"""
        reply = Reply(text="테스트")
        assert hasattr(reply, "raw_response")
        assert reply.raw_response is None

    def test_image_reply_has_raw_response_field(self):
        """ImageReply 객체에 raw_response 필드가 있는지 확인"""
        image_reply = ImageReply(url="https://example.com/image.png")
        assert hasattr(image_reply, "raw_response")
        assert image_reply.raw_response is None

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API 키가 없습니다")
    def test_openai_raw_response(self):
        """OpenAI provider의 raw_response 지원 테스트"""
        llm = OpenAILLM(model="gpt-4o-mini", include_raw_response=True, max_tokens=10)
        reply = llm.ask("1+1은?")

        assert reply.raw_response is not None
        assert isinstance(reply.raw_response, dict)
        assert "id" in reply.raw_response
        assert "model" in reply.raw_response
        assert "choices" in reply.raw_response
        assert "usage" in reply.raw_response

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API 키가 없습니다")
    def test_openai_raw_response_disabled_by_default(self):
        """기본적으로 raw_response가 비활성화되어 있는지 확인"""
        llm = OpenAILLM(model="gpt-4o-mini", max_tokens=10)
        reply = llm.ask("1+1은?")

        assert reply.raw_response is None

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API 키가 없습니다")
    def test_openai_raw_response_via_ask_parameter(self):
        """ask 메서드의 파라미터로 raw_response 활성화 테스트"""
        llm = OpenAILLM(model="gpt-4o-mini", max_tokens=10)
        reply = llm.ask("1+1은?", include_raw_response=True)

        assert reply.raw_response is not None
        assert isinstance(reply.raw_response, dict)

    @patch("openai.OpenAI")
    @patch("openai.AsyncOpenAI")
    def test_mock_openai_raw_response(self, mock_async_openai, mock_sync_openai):
        """Mock을 사용한 OpenAI raw_response 테스트"""
        # Mock 응답 객체 생성
        mock_response = Mock()
        mock_response.id = "test-id"
        mock_response.model = "gpt-4o-mini"
        mock_response.choices = [Mock(message=Mock(content="2"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_response.model_dump.return_value = {
            "id": "test-id",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "2"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        # Mock client 설정
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_sync_openai.return_value = mock_client

        llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key", include_raw_response=True)
        reply = llm.ask("1+1은?")

        assert reply.text == "2"
        assert reply.raw_response is not None
        assert reply.raw_response["id"] == "test-id"
        assert reply.raw_response["model"] == "gpt-4o-mini"

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API 키가 없습니다")
    async def test_openai_raw_response_async(self):
        """비동기 OpenAI raw_response 테스트"""
        llm = OpenAILLM(model="gpt-4o-mini", include_raw_response=True, max_tokens=10)
        reply = await llm.ask_async("1+1은?")

        assert reply.raw_response is not None
        assert isinstance(reply.raw_response, dict)
        assert "id" in reply.raw_response
        assert "model" in reply.raw_response

    @patch("openai.OpenAI")
    def test_openai_image_raw_response(self, mock_openai):
        """OpenAI 이미지 생성의 raw_response 테스트"""
        # Mock 이미지 응답 객체 생성
        mock_image_data = Mock(url="https://example.com/image.png", b64_json=None)
        mock_response = Mock()
        mock_response.data = [mock_image_data]
        mock_response.created = 1234567890
        mock_response.model_dump.return_value = {
            "created": 1234567890,
            "data": [{"url": "https://example.com/image.png"}],
        }

        # Mock client 설정
        mock_client = Mock()
        mock_client.images.generate.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = OpenAILLM(model="dall-e-2", api_key="test-key", include_raw_response=True)
        image_reply = llm.generate_image("작은 빨간 사과", size="256x256")

        assert image_reply.raw_response is not None
        assert isinstance(image_reply.raw_response, dict)
        assert "created" in image_reply.raw_response
        assert "data" in image_reply.raw_response

    @patch("openai.OpenAI")
    @patch("openai.AsyncOpenAI")
    def test_llm_create_with_raw_response(self, mock_async_openai, mock_sync_openai):
        """LLM.create()로 생성 시 raw_response 옵션 테스트"""
        # Mock 응답 객체 생성
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="테스트"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_response.model_dump.return_value = {"test": "data"}

        # Mock client 설정
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_sync_openai.return_value = mock_client

        llm = LLM.create("gpt-4o-mini", api_key="test-key", include_raw_response=True)
        reply = llm.ask("테스트")

        assert reply.raw_response is not None
        assert reply.raw_response == {"test": "data"}
