"""
CLI describe 명령어 테스트

describe 명령어의 주요 기능을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from typer.testing import CliRunner
from pathlib import Path
import json
from PIL import Image
import io

from pyhub.llm.__main__ import app
from pyhub.llm.types import Reply, Usage

runner = CliRunner()


@pytest.fixture
def mock_llm():
    """Mock LLM with image description capability"""
    mock = Mock()
    mock.describe_image.return_value = Reply(
        text="This is an image of a cat sitting on a mat.",
        usage=Usage(input=50, output=30, cost=0.002)
    )
    mock.describe_images = Mock(return_value=[
        Reply(text="Image 1: A cat", usage=Usage(input=40, output=20)),
        Reply(text="Image 2: A dog", usage=Usage(input=40, output=20))
    ])
    return mock


@pytest.fixture
def sample_image(tmp_path):
    """테스트용 이미지 파일 생성"""
    # 간단한 RGB 이미지 생성
    img = Image.new('RGB', (100, 100), color='red')
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_images(tmp_path):
    """여러 테스트용 이미지 파일 생성"""
    images = []
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        img = Image.new('RGB', (100, 100), color=color)
        img_path = tmp_path / f"test_image_{i}.jpg"
        img.save(img_path)
        images.append(img_path)
    return images


class TestDescribeCommand:
    """describe 명령어 테스트"""
    
    def test_describe_single_image(self, mock_llm, sample_image):
        """단일 이미지 설명 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "describe",
                str(sample_image)
            ])
            
            assert result.exit_code == 0
            
            # describe_image가 호출되었는지 확인
            mock_llm.describe_image.assert_called_once()
            
            # 설명이 출력되었는지 확인
            assert "cat sitting on a mat" in result.output
    
    def test_describe_with_custom_prompt(self, mock_llm, sample_image):
        """커스텀 프롬프트 사용 테스트"""
        custom_prompt = "What colors do you see in this image?"
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "describe",
                str(sample_image),
                "--prompt", custom_prompt
            ])
            
            assert result.exit_code == 0
            
            # 커스텀 프롬프트가 전달되었는지 확인
            call_args = mock_llm.describe_image.call_args
            assert custom_prompt in str(call_args)
    
    def test_describe_multiple_images(self, mock_llm, sample_images):
        """여러 이미지 설명 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "describe",
                *[str(img) for img in sample_images]
            ])
            
            assert result.exit_code == 0
            
            # describe_images가 호출되었는지 확인
            assert mock_llm.describe_images.called or mock_llm.describe_image.call_count == len(sample_images)
    
    def test_describe_with_model_option(self, mock_llm, sample_image):
        """모델 옵션 테스트"""
        with patch('pyhub.llm.LLM.create') as mock_create:
            mock_create.return_value = mock_llm
            
            result = runner.invoke(app, [
                "describe",
                str(sample_image),
                "--model", "gpt-4o"
            ])
            
            assert result.exit_code == 0
            
            # 지정한 모델로 LLM이 생성되었는지 확인
            mock_create.assert_called_with("gpt-4o")
    
    def test_describe_save_to_file(self, mock_llm, sample_image, tmp_path):
        """결과를 파일로 저장하는 테스트"""
        output_file = tmp_path / "descriptions.json"
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "describe",
                str(sample_image),
                "--output", str(output_file)
            ])
            
            assert result.exit_code == 0
            assert output_file.exists()
            
            # 저장된 파일 내용 확인
            with open(output_file) as f:
                data = json.load(f)
                assert "descriptions" in data or "results" in data
    
    def test_describe_with_detail_level(self, mock_llm, sample_image):
        """상세도 레벨 옵션 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            # 간단한 설명
            result = runner.invoke(app, [
                "describe",
                str(sample_image),
                "--detail", "low"
            ])
            assert result.exit_code == 0
            
            # 상세한 설명
            result = runner.invoke(app, [
                "describe",
                str(sample_image),
                "--detail", "high"
            ])
            assert result.exit_code == 0
    
    def test_describe_batch_from_directory(self, mock_llm, sample_images):
        """디렉토리의 모든 이미지 처리 테스트"""
        directory = sample_images[0].parent
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "describe",
                str(directory),
                "--batch"
            ])
            
            assert result.exit_code == 0
            
            # 모든 이미지가 처리되었는지 확인
            assert mock_llm.describe_image.call_count >= len(sample_images)
    
    def test_describe_with_format_option(self, mock_llm, sample_image):
        """출력 형식 옵션 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            # JSON 형식
            result = runner.invoke(app, [
                "describe",
                str(sample_image),
                "--format", "json"
            ])
            assert result.exit_code == 0
            # JSON 형식으로 출력되었는지 확인
            assert "{" in result.output or "[" in result.output
            
            # 텍스트 형식 (기본)
            result = runner.invoke(app, [
                "describe",
                str(sample_image),
                "--format", "text"
            ])
            assert result.exit_code == 0
    
    def test_describe_invalid_image_path(self, mock_llm):
        """존재하지 않는 이미지 경로 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "describe",
                "/path/to/nonexistent/image.jpg"
            ])
            
            assert result.exit_code != 0
            assert "not found" in result.output.lower() or "없" in result.output
    
    def test_describe_with_cost_display(self, mock_llm, sample_image):
        """비용 표시 옵션 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "describe",
                str(sample_image),
                "--show-cost"
            ])
            
            assert result.exit_code == 0
            
            # 비용 정보가 출력되었는지 확인
            assert "cost" in result.output.lower() or "비용" in result.output
            assert "0.002" in result.output  # mock에서 설정한 비용
    
    def test_describe_from_url(self, mock_llm):
        """URL에서 이미지 가져오기 테스트"""
        image_url = "https://example.com/image.jpg"
        
        # URL에서 이미지 다운로드를 모킹
        mock_image_data = io.BytesIO()
        Image.new('RGB', (100, 100), color='blue').save(mock_image_data, format='JPEG')
        mock_image_data.seek(0)
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value.read.return_value = mock_image_data.getvalue()
                
                result = runner.invoke(app, [
                    "describe",
                    image_url
                ])
                
                # URL 지원 여부에 따라 결과가 달라질 수 있음
                # 실제 구현에서 URL을 지원한다면 성공해야 함
    
    def test_describe_with_template(self, mock_llm, sample_image, tmp_path):
        """템플릿 파일 사용 테스트"""
        template_file = tmp_path / "template.toml"
        template_content = """
[describe_template]
prompt = "Analyze this image and provide: 1) Main subject 2) Colors 3) Mood"
detail = "high"
"""
        template_file.write_text(template_content)
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "describe",
                str(sample_image),
                "--template", str(template_file)
            ])
            
            assert result.exit_code == 0
    
    def test_describe_parallel_processing(self, mock_llm, sample_images):
        """병렬 처리 옵션 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "describe",
                *[str(img) for img in sample_images],
                "--parallel", "2"
            ])
            
            assert result.exit_code == 0
            
            # 모든 이미지가 처리되었는지 확인
            total_calls = mock_llm.describe_image.call_count + len(mock_llm.describe_images.call_args_list)
            assert total_calls >= len(sample_images)