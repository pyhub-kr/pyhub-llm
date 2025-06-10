"""PDF 파일 지원 테스트"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pyhub.llm import LLM
from pyhub.llm.utils.files import IOType, encode_files


class TestPDFSupport:
    """PDF 파일 지원 테스트"""

    def test_provider_file_types(self):
        """각 Provider의 지원 파일 타입 확인"""
        # OpenAI
        from pyhub.llm.openai import OpenAILLM

        assert IOType.IMAGE in OpenAILLM.SUPPORTED_FILE_TYPES
        assert IOType.PDF in OpenAILLM.SUPPORTED_FILE_TYPES

        # Anthropic
        from pyhub.llm.anthropic import AnthropicLLM

        assert IOType.IMAGE in AnthropicLLM.SUPPORTED_FILE_TYPES
        assert IOType.PDF in AnthropicLLM.SUPPORTED_FILE_TYPES

        # Google
        from pyhub.llm.google import GoogleLLM

        assert IOType.IMAGE in GoogleLLM.SUPPORTED_FILE_TYPES
        assert IOType.PDF in GoogleLLM.SUPPORTED_FILE_TYPES

        # Ollama - 이미지만 지원
        from pyhub.llm.ollama import OllamaLLM

        assert IOType.IMAGE in OllamaLLM.SUPPORTED_FILE_TYPES
        assert IOType.PDF not in OllamaLLM.SUPPORTED_FILE_TYPES

    def test_encode_files_with_pdf(self):
        """PDF 파일 인코딩 테스트"""
        # 테스트용 PDF 파일 생성
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            # 간단한 PDF 헤더
            f.write(b"%PDF-1.4\n")
            test_pdf_path = Path(f.name)

        try:
            # PDF 직접 인코딩 (PDF 지원 Provider)
            encoded = encode_files([test_pdf_path], allowed_types=[IOType.IMAGE, IOType.PDF], convert_mode="base64")

            assert len(encoded) == 1
            assert encoded[0].startswith("data:application/pdf;base64,")

        finally:
            test_pdf_path.unlink()

    @patch("pyhub.llm.utils.files.pdf_to_images")
    def test_encode_files_pdf_to_image_conversion(self, mock_pdf_to_images):
        """PDF를 이미지로 변환하는 테스트"""
        # Mock 설정
        mock_pdf_to_images.return_value = [
            (b"fake_image_data1", "image/png"),
            (b"fake_image_data2", "image/png"),
        ]

        # 테스트용 PDF 파일 생성
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n")
            test_pdf_path = Path(f.name)

        try:
            # PDF를 이미지로 변환 (Ollama처럼 PDF 미지원 Provider)
            encoded = encode_files(
                [test_pdf_path],
                allowed_types=[IOType.IMAGE],  # PDF는 허용 안 함
                convert_mode="base64",
                pdf_to_image_for_unsupported=True,
            )

            # 2개의 페이지가 2개의 이미지로 변환됨
            assert len(encoded) == 2
            assert all(url.startswith("data:image/png;base64,") for url in encoded)

            # pdf_to_images 함수가 호출되었는지 확인
            mock_pdf_to_images.assert_called_once()

        finally:
            test_pdf_path.unlink()

    @patch("pyhub.llm.utils.files.logger")
    def test_pdf_conversion_warning_log(self, mock_logger):
        """PDF 변환 시 경고 로그 테스트"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n")
            test_pdf_path = Path(f.name)

        try:
            # PDF 미지원으로 처리
            encoded = encode_files(
                [test_pdf_path],
                allowed_types=[IOType.IMAGE],  # PDF 미허용
                convert_mode="base64",
                pdf_to_image_for_unsupported=True,
            )

            # 경고 로그가 기록되었는지 확인
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "PDF" in warning_msg
            assert "이미지로 변환" in warning_msg

        finally:
            test_pdf_path.unlink()

    @pytest.mark.skipif(True, reason="PyMuPDF 의존성 필요")
    def test_pdf_to_images_real(self):
        """실제 PDF 변환 테스트 (PyMuPDF 필요)"""
        # 실제 PDF 파일이 있을 때만 테스트
        # PDF 생성 및 변환 로직
        pass


class TestProviderPDFHandling:
    """Provider별 PDF 처리 테스트"""

    @patch("pyhub.llm.openai.OpenAILLM._make_ask")
    def test_openai_pdf_support(self, mock_make_ask):
        """OpenAI의 PDF 지원 테스트"""
        mock_make_ask.return_value = Mock(text="PDF processed")

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n")
            test_pdf_path = Path(f.name)

        try:
            llm = LLM.create("gpt-4o")
            # PDF 파일이 그대로 전달되어야 함
            response = llm.ask("What's in this PDF?", files=[test_pdf_path])

            # _make_ask가 호출되었는지 확인
            mock_make_ask.assert_called_once()

        finally:
            test_pdf_path.unlink()

    @patch("pyhub.llm.utils.files.pdf_to_images")
    def test_ollama_pdf_to_image_conversion(self, mock_pdf_to_images):
        """Ollama의 PDF->이미지 변환 테스트"""
        # Mock 설정
        mock_pdf_to_images.return_value = [(b"fake_image", "image/png")]

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n")
            test_pdf_path = Path(f.name)

        try:
            # encode_files를 직접 테스트
            from pyhub.llm.utils.files import encode_files

            # Ollama처럼 PDF를 지원하지 않는 경우
            encoded = encode_files(
                [test_pdf_path],
                allowed_types=[IOType.IMAGE],  # PDF는 허용하지 않음
                convert_mode="base64",
                pdf_to_image_for_unsupported=True,
            )

            # pdf_to_images가 호출되었는지 확인
            mock_pdf_to_images.assert_called_once()

        finally:
            test_pdf_path.unlink()
