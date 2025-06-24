"""
CLI embed 명령어 테스트

embed 서브커맨드들의 주요 기능을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from typer.testing import CliRunner
from pathlib import Path
import json
import numpy as np

from pyhub.llm.__main__ import app
from pyhub.llm.types import Reply, Usage

runner = CliRunner()


@pytest.fixture
def mock_llm():
    """Mock LLM with embedding capability"""
    mock = Mock()
    # 768차원 벡터 반환 (일반적인 임베딩 크기)
    mock.embed.return_value = np.random.rand(768).tolist()
    mock.embed_batch = Mock(return_value=[
        np.random.rand(768).tolist() for _ in range(3)
    ])
    return mock


@pytest.fixture
def sample_jsonl_content():
    """테스트용 JSONL 콘텐츠"""
    return '\n'.join([
        json.dumps({"id": 1, "text": "Hello world"}),
        json.dumps({"id": 2, "text": "Python programming"}),
        json.dumps({"id": 3, "text": "Machine learning"})
    ])


class TestEmbedCommands:
    """embed 명령어 테스트"""
    
    def test_embed_text_basic(self, mock_llm):
        """기본 텍스트 임베딩 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "embed", "text",
                "Hello, world!",
                "--model", "text-embedding-3-small"
            ])
            
            assert result.exit_code == 0
            
            # embed 메서드가 호출되었는지 확인
            mock_llm.embed.assert_called_once_with("Hello, world!")
            
            # 벡터가 출력되었는지 확인
            assert "[" in result.output  # 벡터 배열 시작
            assert "]" in result.output  # 벡터 배열 끝
    
    def test_embed_text_save_to_file(self, mock_llm, tmp_path):
        """임베딩 결과를 파일로 저장하는 테스트"""
        output_file = tmp_path / "embedding.json"
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "embed", "text",
                "Test text",
                "--output", str(output_file)
            ])
            
            assert result.exit_code == 0
            assert output_file.exists()
            
            # 저장된 파일 내용 확인
            with open(output_file) as f:
                data = json.load(f)
                assert "text" in data
                assert "embedding" in data
                assert data["text"] == "Test text"
                assert isinstance(data["embedding"], list)
    
    def test_embed_text_dimension_display(self, mock_llm):
        """임베딩 차원 표시 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "embed", "text",
                "Test",
                "--show-dimension"
            ])
            
            assert result.exit_code == 0
            assert "768" in result.output  # 차원 수가 표시되어야 함
    
    def test_embed_similarity_two_texts(self, mock_llm):
        """두 텍스트 간 유사도 계산 테스트"""
        # 고정된 임베딩 값 설정
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.8, 0.6, 0.0]
        mock_llm.embed.side_effect = [embedding1, embedding2]
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "embed", "similarity",
                "cat",
                "dog"
            ])
            
            assert result.exit_code == 0
            
            # 두 번 embed가 호출되었는지 확인
            assert mock_llm.embed.call_count == 2
            
            # 코사인 유사도가 출력되었는지 확인
            assert "similarity" in result.output.lower() or "유사도" in result.output
            assert "0." in result.output  # 유사도 값
    
    def test_embed_similarity_with_files(self, mock_llm, tmp_path):
        """파일에서 임베딩을 읽어 유사도 계산하는 테스트"""
        # 임베딩 파일 생성
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.8, 0.6, 0.0]
        
        file1 = tmp_path / "embed1.json"
        file2 = tmp_path / "embed2.json"
        
        file1.write_text(json.dumps({"embedding": embedding1}))
        file2.write_text(json.dumps({"embedding": embedding2}))
        
        result = runner.invoke(app, [
            "embed", "similarity",
            str(file1),
            str(file2),
            "--from-files"
        ])
        
        assert result.exit_code == 0
        assert "similarity" in result.output.lower() or "유사도" in result.output
    
    def test_embed_batch(self, mock_llm, tmp_path):
        """배치 임베딩 테스트"""
        input_file = tmp_path / "texts.txt"
        output_file = tmp_path / "embeddings.jsonl"
        
        # 입력 파일 생성
        input_file.write_text("Text 1\nText 2\nText 3")
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "embed", "batch",
                str(input_file),
                "--output", str(output_file)
            ])
            
            assert result.exit_code == 0
            assert output_file.exists()
            
            # 출력 파일 확인
            lines = output_file.read_text().strip().split('\n')
            assert len(lines) == 3
            
            # 각 줄이 올바른 JSON인지 확인
            for line in lines:
                data = json.loads(line)
                assert "text" in data
                assert "embedding" in data
    
    def test_embed_fill_jsonl(self, mock_llm, tmp_path, sample_jsonl_content):
        """JSONL 파일에 임베딩 추가 테스트"""
        input_file = tmp_path / "data.jsonl"
        output_file = tmp_path / "data_with_embeddings.jsonl"
        
        # 입력 파일 생성
        input_file.write_text(sample_jsonl_content)
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "embed", "fill-jsonl",
                str(input_file),
                "--text-field", "text",
                "--output", str(output_file),
                "--embedding-field", "vector"
            ])
            
            assert result.exit_code == 0
            assert output_file.exists()
            
            # 출력 파일 검증
            lines = output_file.read_text().strip().split('\n')
            assert len(lines) == 3
            
            for line in lines:
                data = json.loads(line)
                assert "text" in data
                assert "vector" in data  # 임베딩 필드가 추가되었는지
                assert isinstance(data["vector"], list)
    
    def test_embed_fill_jsonl_skip_existing(self, mock_llm, tmp_path):
        """이미 임베딩이 있는 항목은 건너뛰는 테스트"""
        # 일부 항목에 이미 임베딩이 있는 JSONL
        content = '\n'.join([
            json.dumps({"id": 1, "text": "Hello", "embedding": [0.1, 0.2]}),
            json.dumps({"id": 2, "text": "World"})
        ])
        
        input_file = tmp_path / "data.jsonl"
        input_file.write_text(content)
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "embed", "fill-jsonl",
                str(input_file),
                "--text-field", "text",
                "--skip-existing"
            ])
            
            assert result.exit_code == 0
            
            # embed가 한 번만 호출되었는지 확인 (두 번째 항목만)
            assert mock_llm.embed.call_count == 1
            mock_llm.embed.assert_called_with("World")
    
    def test_embed_batch_with_batch_size(self, mock_llm, tmp_path):
        """배치 크기를 지정한 배치 임베딩 테스트"""
        input_file = tmp_path / "texts.txt"
        texts = [f"Text {i}" for i in range(10)]
        input_file.write_text('\n'.join(texts))
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "embed", "batch",
                str(input_file),
                "--batch-size", "3"
            ])
            
            assert result.exit_code == 0
            
            # embed_batch가 여러 번 호출되었는지 확인
            # 10개 텍스트를 3개씩 배치로 처리하면 4번 호출
            assert mock_llm.embed_batch.call_count >= 3
    
    def test_embed_similarity_different_metrics(self, mock_llm):
        """다양한 유사도 메트릭 테스트"""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.8, 0.6, 0.0]
        mock_llm.embed.side_effect = [embedding1, embedding2]
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            # 코사인 유사도 (기본값)
            result = runner.invoke(app, [
                "embed", "similarity",
                "text1",
                "text2"
            ])
            assert result.exit_code == 0
            
            # 유클리드 거리
            result = runner.invoke(app, [
                "embed", "similarity",
                "text1",
                "text2",
                "--metric", "euclidean"
            ])
            assert result.exit_code == 0
            
            # 내적
            result = runner.invoke(app, [
                "embed", "similarity",
                "text1",
                "text2",
                "--metric", "dot"
            ])
            assert result.exit_code == 0
    
    def test_embed_error_handling(self):
        """에러 처리 테스트"""
        with patch('pyhub.llm.LLM.create') as mock_create:
            mock_create.side_effect = Exception("API key not found")
            
            result = runner.invoke(app, [
                "embed", "text",
                "Test text"
            ])
            
            assert result.exit_code != 0
            assert "API key not found" in result.output