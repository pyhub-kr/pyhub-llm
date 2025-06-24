"""
CLI compare 명령어 테스트

compare 명령어의 주요 기능을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typer.testing import CliRunner
from pathlib import Path
import json
import asyncio

from pyhub.llm.__main__ import app
from pyhub.llm.types import Reply, Usage

runner = CliRunner()


@pytest.fixture
def mock_llm_factory():
    """Mock LLM factory for multiple models"""
    models = {}
    
    def create_mock_llm(model_name):
        if model_name not in models:
            mock = Mock()
            mock.model_name = model_name
            
            # 각 모델마다 다른 응답 설정
            if "gpt" in model_name:
                response = f"Response from {model_name}: GPT style"
                usage = Usage(input=10, output=20, cost=0.001)
            elif "claude" in model_name:
                response = f"Response from {model_name}: Claude style"
                usage = Usage(input=15, output=25, cost=0.0015)
            else:
                response = f"Response from {model_name}: Generic style"
                usage = Usage(input=5, output=10, cost=0.0005)
            
            mock.ask_sync.return_value = Reply(text=response, usage=usage)
            mock.ask_async = AsyncMock(return_value=Reply(text=response, usage=usage))
            models[model_name] = mock
        
        return models[model_name]
    
    with patch('pyhub.llm.LLM.create', side_effect=create_mock_llm):
        yield create_mock_llm


@pytest.fixture
def mock_console():
    """Mock Rich Console"""
    with patch('pyhub.llm.commands.compare.console') as mock:
        mock.print = Mock()
        yield mock


class TestCompareCommand:
    """compare 명령어 테스트"""
    
    def test_compare_basic(self, mock_llm_factory, mock_console):
        """기본 비교 기능 테스트"""
        result = runner.invoke(app, [
            "compare",
            "What is Python?",
            "--models", "gpt-4o-mini,claude-3-haiku"
        ])
        
        assert result.exit_code == 0
        
        # 두 모델이 모두 호출되었는지 확인
        assert mock_console.print.called
        
        # 출력에 두 모델의 응답이 포함되었는지 확인
        output_str = str(mock_console.print.call_args_list)
        assert "gpt-4o-mini" in output_str
        assert "claude-3-haiku" in output_str
    
    def test_compare_with_single_model(self, mock_llm_factory, mock_console):
        """단일 모델로 비교 (기본 모델과 비교)"""
        result = runner.invoke(app, [
            "compare",
            "Hello",
            "--models", "gpt-4o"
        ])
        
        assert result.exit_code == 0
        
        # 지정한 모델과 기본 모델이 사용되었는지 확인
        output_str = str(mock_console.print.call_args_list)
        assert "gpt-4o" in output_str
    
    def test_compare_format_side_by_side(self, mock_llm_factory, mock_console):
        """side-by-side 형식 테스트"""
        result = runner.invoke(app, [
            "compare",
            "Test prompt",
            "--models", "gpt-4o-mini,claude-3-haiku",
            "--format", "side-by-side"
        ])
        
        assert result.exit_code == 0
        
        # Table이 생성되었는지 확인 (side-by-side는 테이블 형식)
        printed_args = [call[0][0] for call in mock_console.print.call_args_list]
        assert any(hasattr(arg, '__class__') and 
                  arg.__class__.__name__ == 'Table' 
                  for arg in printed_args)
    
    def test_compare_format_sequential(self, mock_llm_factory, mock_console):
        """sequential 형식 테스트"""
        result = runner.invoke(app, [
            "compare",
            "Test prompt",
            "--models", "gpt-4o-mini,claude-3-haiku",
            "--format", "sequential"
        ])
        
        assert result.exit_code == 0
        
        # 순차적으로 출력되었는지 확인
        output_str = result.output
        assert "gpt-4o-mini" in output_str
        assert "claude-3-haiku" in output_str
    
    def test_compare_with_system_prompt(self, mock_llm_factory, mock_console):
        """시스템 프롬프트 사용 테스트"""
        result = runner.invoke(app, [
            "compare",
            "Explain quantum computing",
            "--models", "gpt-4o,claude-3-opus",
            "--system-prompt", "You are a physics professor"
        ])
        
        assert result.exit_code == 0
        
        # 시스템 프롬프트가 전달되었는지 확인은 
        # 실제 구현에서 LLM.create 호출 시 확인해야 함
    
    def test_compare_save_results(self, mock_llm_factory, mock_console, tmp_path):
        """결과 저장 기능 테스트"""
        output_file = tmp_path / "comparison_results.json"
        
        result = runner.invoke(app, [
            "compare",
            "Test query",
            "--models", "gpt-4o-mini,claude-3-haiku",
            "--save", str(output_file)
        ])
        
        assert result.exit_code == 0
        
        # 파일이 생성되었는지 확인
        assert output_file.exists()
        
        # JSON 파일 내용 확인
        with open(output_file) as f:
            data = json.load(f)
            assert "results" in data
            assert len(data["results"]) >= 2
            assert "prompt" in data
            assert data["prompt"] == "Test query"
    
    def test_compare_show_stats(self, mock_llm_factory, mock_console):
        """통계 표시 옵션 테스트"""
        result = runner.invoke(app, [
            "compare",
            "Calculate 2+2",
            "--models", "gpt-4o-mini,claude-3-haiku,gpt-3.5-turbo",
            "--show-stats"
        ])
        
        assert result.exit_code == 0
        
        # 통계 테이블이 출력되었는지 확인
        output_str = result.output
        assert "Total Cost" in output_str or "비용" in output_str
        assert "Response Time" in output_str or "응답 시간" in output_str
    
    def test_compare_with_temperature(self, mock_llm_factory, mock_console):
        """temperature 옵션 테스트"""
        result = runner.invoke(app, [
            "compare",
            "Write a poem",
            "--models", "gpt-4o,claude-3-opus",
            "--temperature", "1.5"
        ])
        
        assert result.exit_code == 0
        # temperature가 각 모델에 전달되었는지는 실제 구현에서 확인
    
    def test_compare_with_max_tokens(self, mock_llm_factory, mock_console):
        """max_tokens 옵션 테스트"""
        result = runner.invoke(app, [
            "compare",
            "Explain AI",
            "--models", "gpt-4o-mini,claude-3-haiku",
            "--max-tokens", "100"
        ])
        
        assert result.exit_code == 0
        # max_tokens가 각 모델에 전달되었는지는 실제 구현에서 확인
    
    def test_compare_error_handling(self, mock_console):
        """에러 처리 테스트"""
        with patch('pyhub.llm.LLM.create') as mock_create:
            # 첫 번째 모델은 성공, 두 번째는 실패
            mock_llm1 = Mock()
            mock_llm1.ask_sync.return_value = Reply(
                text="Success", 
                usage=Usage(input=5, output=10)
            )
            
            mock_create.side_effect = [
                mock_llm1,
                Exception("API Error for second model")
            ]
            
            result = runner.invoke(app, [
                "compare",
                "Test",
                "--models", "gpt-4o,invalid-model"
            ])
            
            # 부분적으로 성공해도 결과를 보여줘야 함
            assert "Success" in result.output
            assert "Error" in result.output or "error" in result.output
    
    def test_compare_parallel_execution(self, mock_console):
        """병렬 실행 테스트"""
        call_times = []
        
        async def mock_ask_async(prompt):
            call_time = asyncio.get_event_loop().time()
            call_times.append(call_time)
            # 실제 지연 시뮬레이션
            await asyncio.sleep(0.1)
            return Reply(text=f"Response at {call_time}", usage=Usage(input=5, output=10))
        
        with patch('pyhub.llm.LLM.create') as mock_create:
            mock_llm = Mock()
            mock_llm.ask_async = mock_ask_async
            mock_create.return_value = mock_llm
            
            result = runner.invoke(app, [
                "compare",
                "Test parallel",
                "--models", "model1,model2,model3"
            ])
            
            assert result.exit_code == 0
            
            # 병렬로 실행되었다면 호출 시간이 거의 동시여야 함
            if len(call_times) >= 2:
                time_diff = max(call_times) - min(call_times)
                assert time_diff < 0.05  # 50ms 이내
    
    def test_compare_format_table(self, mock_llm_factory, mock_console):
        """table 형식 테스트"""
        result = runner.invoke(app, [
            "compare",
            "What is 2+2?",
            "--models", "gpt-4o-mini,claude-3-haiku",
            "--format", "table"
        ])
        
        assert result.exit_code == 0
        
        # 테이블이 출력되었는지 확인
        printed_args = [call[0][0] for call in mock_console.print.call_args_list]
        assert any(hasattr(arg, '__class__') and 
                  arg.__class__.__name__ == 'Table' 
                  for arg in printed_args)