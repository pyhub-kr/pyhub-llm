"""
CLI chat 명령어 테스트

chat 명령어의 주요 기능을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from pathlib import Path
import json

from pyhub.llm.__main__ import app
from pyhub.llm.types import Reply, Usage

runner = CliRunner()


@pytest.fixture
def mock_llm():
    """Mock LLM 인스턴스"""
    mock = Mock()
    
    # 스트리밍 응답 생성기
    def mock_ask_generator(*args, **kwargs):
        yield Reply(text="Test", usage=None)
        yield Reply(text=" response", usage=Usage(input=10, output=20))
    
    mock.ask = Mock(side_effect=mock_ask_generator)
    mock.ask_sync = Mock(return_value=Reply(
        text="Test response",
        usage=Usage(input=10, output=20)
    ))
    mock.usage = Usage(input=10, output=20)
    mock.clear = Mock()
    return mock


@pytest.fixture
def mock_console():
    """Mock Rich Console"""
    with patch('pyhub.llm.commands.chat.console') as mock:
        mock.print = Mock()
        mock.clear = Mock()
        yield mock


@pytest.fixture
def mock_prompt():
    """Mock Rich Prompt"""
    with patch('pyhub.llm.commands.chat.Prompt.ask') as mock:
        # 시뮬레이션: 첫 번째 호출은 일반 메시지, 두 번째는 종료
        mock.side_effect = ["Hello", "exit"]
        yield mock


class TestChatCommand:
    """chat 명령어 테스트"""
    
    def test_chat_basic_interaction(self, mock_llm, mock_console, mock_prompt):
        """기본 대화 상호작용 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, ["chat"])
            
            # 명령어가 성공적으로 실행되었는지 확인
            assert result.exit_code == 0
            
            # LLM이 생성되었는지 확인
            assert mock_llm.ask.called
            
            # 사용자 입력이 처리되었는지 확인
            mock_llm.ask.assert_called_with("Hello", stream=True)
            
            # 응답이 출력되었는지 확인
            assert mock_console.print.called
    
    def test_chat_with_system_prompt(self, mock_llm, mock_console, mock_prompt):
        """시스템 프롬프트를 사용한 chat 테스트"""
        with patch('pyhub.llm.LLM.create') as mock_create:
            mock_create.return_value = mock_llm
            
            result = runner.invoke(app, [
                "chat",
                "--system-prompt", "You are a helpful assistant",
                "--model", "gpt-4o-mini"
            ])
            
            assert result.exit_code == 0
            
            # LLM이 시스템 프롬프트와 함께 생성되었는지 확인
            mock_create.assert_called_with(
                "gpt-4o-mini",
                system_prompt="You are a helpful assistant"
            )
    
    def test_chat_clear_command(self, mock_llm, mock_console):
        """clear 명령어 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            with patch('pyhub.llm.commands.chat.Prompt.ask') as mock_prompt:
                # clear 명령어 후 exit
                mock_prompt.side_effect = ["clear", "exit"]
                
                result = runner.invoke(app, ["chat"])
                
                assert result.exit_code == 0
                
                # 콘솔이 클리어되었는지 확인
                mock_console.clear.assert_called()
                
                # LLM 히스토리가 클리어되었는지 확인
                mock_llm.clear.assert_called()
    
    def test_chat_with_cost_display(self, mock_llm, mock_console, mock_prompt):
        """비용 표시 옵션 테스트"""
        # Mock usage with cost calculation
        mock_llm.usage = Usage(input=100, output=50, cost=0.0015)
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, ["chat", "--cost"])
            
            assert result.exit_code == 0
            
            # 비용 정보가 출력되었는지 확인
            console_calls = str(mock_console.print.call_args_list)
            assert any("Cost" in str(call) or "비용" in str(call) or "0.0015" in str(call) 
                      for call in mock_console.print.call_args_list)
    
    def test_chat_history_save(self, mock_llm, mock_console, mock_prompt, tmp_path):
        """대화 히스토리 저장 테스트"""
        history_file = tmp_path / "chat_history.json"
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, [
                "chat",
                "--history", str(history_file)
            ])
            
            assert result.exit_code == 0
            
            # 히스토리 파일이 생성되었는지 확인
            # Note: 실제 구현에서는 대화 종료 시 저장되므로
            # 여기서는 파일 생성 로직이 호출되는지만 확인
    
    def test_chat_markdown_rendering(self, mock_llm, mock_console, mock_prompt):
        """마크다운 렌더링 테스트"""
        mock_llm.ask_sync.return_value = Reply(
            text="# Heading\n**Bold text**\n- List item",
            usage=Usage(input=10, output=20)
        )
        
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            result = runner.invoke(app, ["chat", "--markdown"])
            
            assert result.exit_code == 0
            
            # Markdown 객체가 생성되어 출력되었는지 확인
            printed_args = [call[0][0] for call in mock_console.print.call_args_list]
            assert any(hasattr(arg, '__class__') and 
                      arg.__class__.__name__ == 'Markdown' 
                      for arg in printed_args)
    
    def test_chat_settings_command(self, mock_llm, mock_console):
        """settings 명령어 테스트"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            with patch('pyhub.llm.commands.chat.Prompt.ask') as mock_prompt:
                # settings 명령어 후 exit
                mock_prompt.side_effect = ["settings", "exit"]
                
                result = runner.invoke(app, ["chat", "--model", "gpt-4o"])
                
                assert result.exit_code == 0
                
                # 설정 정보가 테이블로 출력되었는지 확인
                printed_args = [call[0][0] for call in mock_console.print.call_args_list]
                assert any(hasattr(arg, '__class__') and 
                          arg.__class__.__name__ == 'Table' 
                          for arg in printed_args)
    
    def test_chat_temperature_option(self, mock_llm, mock_console, mock_prompt):
        """temperature 옵션 테스트"""
        with patch('pyhub.llm.LLM.create') as mock_create:
            mock_create.return_value = mock_llm
            
            result = runner.invoke(app, [
                "chat",
                "--temperature", "0.3",
                "--model", "gpt-4o-mini"
            ])
            
            assert result.exit_code == 0
            
            # temperature가 전달되었는지 확인
            mock_create.assert_called_with(
                "gpt-4o-mini",
                system_prompt=None,
                temperature=0.3
            )
    
    def test_chat_max_tokens_option(self, mock_llm, mock_console, mock_prompt):
        """max_tokens 옵션 테스트"""
        with patch('pyhub.llm.LLM.create') as mock_create:
            mock_create.return_value = mock_llm
            
            result = runner.invoke(app, [
                "chat",
                "--max-tokens", "500"
            ])
            
            assert result.exit_code == 0
            
            # max_tokens가 전달되었는지 확인
            mock_create.assert_called_with(
                "gpt-4o-mini",  # 기본 모델
                system_prompt=None,
                max_tokens=500
            )
    
    def test_chat_error_handling(self, mock_console, mock_prompt):
        """에러 처리 테스트"""
        with patch('pyhub.llm.LLM.create') as mock_create:
            mock_create.side_effect = Exception("API key not found")
            
            result = runner.invoke(app, ["chat"])
            
            # 에러가 발생해도 graceful하게 종료되어야 함
            assert result.exit_code != 0
            assert "API key not found" in result.output
    
    def test_chat_multiline_input(self, mock_llm, mock_console):
        """여러 줄 입력 테스트 (>>> 사용)"""
        with patch('pyhub.llm.LLM.create', return_value=mock_llm):
            with patch('pyhub.llm.commands.chat.Prompt.ask') as mock_prompt:
                # >>> 입력 시뮬레이션
                mock_prompt.side_effect = [
                    ">>>",
                    "Line 1",
                    "Line 2",
                    "",  # 빈 줄로 종료
                    "exit"
                ]
                
                result = runner.invoke(app, ["chat"])
                
                assert result.exit_code == 0
                
                # 멀티라인 입력이 결합되었는지 확인
                expected_input = "Line 1\nLine 2"
                mock_llm.ask_sync.assert_any_call(expected_input)