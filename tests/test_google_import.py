"""Google LLM import 테스트"""

from unittest.mock import MagicMock, patch

import pytest


class TestGoogleImport:
    """Google LLM import 및 초기화 테스트"""

    def test_google_llm_import(self):
        """GoogleLLM이 정상적으로 import 되는지 테스트"""
        try:
            from pyhub.llm.google import GoogleLLM

            assert GoogleLLM is not None
        except ImportError as e:
            pytest.skip(f"Google provider not available: {e}")

    def test_google_llm_initialization(self):
        """GoogleLLM 초기화가 정상적으로 되는지 테스트"""
        # Clear cached module
        import sys

        if "pyhub.llm.google" in sys.modules:
            del sys.modules["pyhub.llm.google"]

        # google.genai 모듈을 모킹
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response_class = MagicMock()
        mock_embed_response_class = MagicMock()
        mock_config_class = MagicMock()
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.genai": mock_genai,
                "google.genai.types": MagicMock(
                    GenerateContentResponse=mock_response_class,
                    EmbedContentResponse=mock_embed_response_class,
                    GenerateContentConfig=mock_config_class,
                ),
            },
        ):
            # Mock llm_settings to have no API key
            with patch("pyhub.llm.settings.llm_settings.google_api_key", None):
                from pyhub.llm.google import GoogleLLM

                # API 키 없이 초기화 시도
                llm = GoogleLLM(model="gemini-1.5-flash")
                assert llm is not None
                assert llm._client is None  # No API key, so client should be None

    def test_google_llm_with_api_key(self):
        """API 키와 함께 GoogleLLM 초기화 테스트"""
        # Clear cached module
        import sys

        if "pyhub.llm.google" in sys.modules:
            del sys.modules["pyhub.llm.google"]

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.genai": mock_genai,
                "google.genai.types": MagicMock(
                    GenerateContentResponse=MagicMock(),
                    EmbedContentResponse=MagicMock(),
                    GenerateContentConfig=MagicMock(),
                ),
            },
        ):
            from pyhub.llm.google import GoogleLLM

            # API 키와 함께 초기화
            llm = GoogleLLM(model="gemini-1.5-flash", api_key="test-api-key")

            # Client가 API 키와 함께 생성되었는지 확인
            mock_genai.Client.assert_called_once_with(api_key="test-api-key")
            assert llm._client == mock_client

    def test_google_genai_not_installed(self):
        """google-genai가 설치되지 않은 경우 에러 테스트"""
        # google import 실패를 시뮬레이션
        with patch.dict("sys.modules", {"google": None}):
            # 캐시된 모듈 제거
            import sys

            if "pyhub.llm.google" in sys.modules:
                del sys.modules["pyhub.llm.google"]

            from pyhub.llm.google import GoogleLLM

            # 초기화 시 ImportError가 발생해야 함
            with pytest.raises(ImportError, match="google-genai package not installed"):
                GoogleLLM(model="gemini-1.5-flash")
