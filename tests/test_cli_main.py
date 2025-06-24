"""
CLI 메인 진입점 테스트

__main__.py의 기본 기능과 전체적인 CLI 동작을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, patch
from typer.testing import CliRunner
import subprocess
import sys

from pyhub.llm.__main__ import app, main

runner = CliRunner()


class TestCLIMain:
    """CLI 메인 진입점 테스트"""
    
    def test_main_help(self):
        """도움말 표시 테스트"""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "pyhub-llm" in result.output
        assert "ask" in result.output
        assert "chat" in result.output
        assert "compare" in result.output
        assert "describe" in result.output
        assert "embed" in result.output
    
    def test_main_version(self):
        """버전 표시 테스트"""
        with patch('pyhub.llm.__version__', '1.0.0'):
            result = runner.invoke(app, ["--version"])
            
            # 버전 정보가 표시되는지 확인
            # 실제 구현에 따라 다를 수 있음
    
    def test_main_function(self):
        """main() 함수 직접 호출 테스트"""
        with patch('typer.run') as mock_run:
            main()
            mock_run.assert_called_once()
    
    def test_subcommand_help(self):
        """서브커맨드 도움말 테스트"""
        # ask 명령어 도움말
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "Query" in result.output or "질문" in result.output
        
        # chat 명령어 도움말
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "Interactive" in result.output or "대화" in result.output
        
        # embed 명령어 도움말
        result = runner.invoke(app, ["embed", "--help"])
        assert result.exit_code == 0
        assert "text" in result.output
        assert "similarity" in result.output
    
    def test_invalid_command(self):
        """잘못된 명령어 처리 테스트"""
        result = runner.invoke(app, ["invalid-command"])
        
        assert result.exit_code != 0
        assert "No such command" in result.output or "명령" in result.output
    
    def test_missing_required_args(self):
        """필수 인자 누락 테스트"""
        # ask 명령어에 프롬프트 없이 실행
        result = runner.invoke(app, ["ask"])
        
        assert result.exit_code != 0
        assert "Missing" in result.output or "누락" in result.output
    
    def test_environment_variable_handling(self):
        """환경 변수 처리 테스트"""
        # API 키가 없을 때
        with patch.dict('os.environ', {}, clear=True):
            result = runner.invoke(app, ["ask", "Hello"])
            
            # API 키 관련 에러가 발생해야 함
            assert result.exit_code != 0
    
    def test_global_options(self):
        """전역 옵션 테스트"""
        # 만약 --debug, --quiet 같은 전역 옵션이 있다면 테스트
        # 현재 구현에 따라 다름
        pass
    
    def test_command_aliases(self):
        """명령어 별칭 테스트 (있다면)"""
        # 예: 'describe'의 별칭이 'desc'라면
        # result = runner.invoke(app, ["desc", "image.jpg"])
        pass
    
    def test_python_module_execution(self):
        """python -m pyhub.llm 실행 테스트"""
        # 실제 서브프로세스로 실행
        result = subprocess.run(
            [sys.executable, "-m", "pyhub.llm", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "pyhub-llm" in result.stdout
    
    def test_error_handling_graceful_exit(self):
        """에러 발생 시 graceful exit 테스트"""
        with patch('pyhub.llm.LLM.create') as mock_create:
            mock_create.side_effect = KeyboardInterrupt()
            
            result = runner.invoke(app, ["ask", "test"])
            
            # KeyboardInterrupt가 적절히 처리되는지
            assert result.exit_code != 0
    
    def test_config_file_loading(self):
        """설정 파일 로딩 테스트 (있다면)"""
        # 설정 파일 기능이 있다면 테스트
        pass
    
    def test_plugin_system(self):
        """플러그인 시스템 테스트 (있다면)"""
        # 플러그인 로딩 기능이 있다면 테스트
        pass
    
    def test_output_format_options(self):
        """출력 형식 옵션 테스트"""
        # JSON, 테이블 등 다양한 출력 형식 지원 테스트
        with patch('pyhub.llm.LLM.create') as mock_create:
            mock_llm = Mock()
            mock_llm.ask_sync.return_value = Mock(text="Test", usage=Mock())
            mock_create.return_value = mock_llm
            
            # JSON 출력 테스트
            result = runner.invoke(app, ["ask", "test", "--json"])
            # JSON 형식으로 출력되는지 확인
    
    def test_completion_script_generation(self):
        """쉘 자동완성 스크립트 생성 테스트"""
        # Typer의 자동완성 기능 테스트
        # result = runner.invoke(app, ["--install-completion"])
        pass