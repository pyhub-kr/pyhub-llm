import json
import tempfile
from pathlib import Path

import pytest

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from pyhub.llm.mcp import McpConfig
from pyhub.llm.mcp.config_loader import load_mcp_config


class TestMCPConfigLoader:
    """MCP 설정 파일 로더 테스트"""

    def test_load_from_json_file(self):
        """JSON 파일에서 MCP 설정 로드"""
        config_data = {
            "mcpServers": [
                {"cmd": "python calculator.py", "name": "calculator"},
                {"url": "http://localhost:8888/mcp", "name": "web"},
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            configs = load_mcp_config(temp_path)
            assert len(configs) == 2
            assert isinstance(configs[0], McpConfig)
            assert configs[0].transport == "stdio"
            assert configs[0].name == "calculator"
            assert configs[0].cmd == "python calculator.py"
            assert isinstance(configs[1], McpConfig)
            assert configs[1].transport == "streamable_http"
            assert configs[1].name == "web"
            assert configs[1].url == "http://localhost:8888/mcp"
        finally:
            Path(temp_path).unlink()

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_load_from_yaml_file(self):
        """YAML 파일에서 MCP 설정 로드"""
        yaml_content = """
mcpServers:
  - cmd: pyhub-llm mcp-server run calculator
    name: calculator
    timeout: 60
  - url: ws://localhost:8080/ws
    name: ws_server
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            configs = load_mcp_config(temp_path)
            assert len(configs) == 2
            assert isinstance(configs[0], McpConfig)
            assert configs[0].transport == "stdio"
            assert configs[0].name == "calculator"
            assert configs[0].cmd == "pyhub-llm mcp-server run calculator"
            assert configs[0].timeout == 60
            assert isinstance(configs[1], McpConfig)
            assert configs[1].transport == "websocket"
            assert configs[1].name == "ws_server"
            assert configs[1].url == "ws://localhost:8080/ws"
        finally:
            Path(temp_path).unlink()

    def test_load_from_dict(self):
        """딕셔너리에서 MCP 설정 로드"""
        config_data = {
            "mcpServers": [
                {"cmd": "python server.py", "name": "test1"},
                {"url": "http://localhost:8080", "name": "test2"},
                {"url": "ws://localhost:8080", "name": "test3"},
            ]
        }

        configs = load_mcp_config(config_data)
        assert len(configs) == 3
        assert configs[0].transport == "stdio"
        assert configs[1].transport == "streamable_http"
        assert configs[2].transport == "websocket"

    def test_load_from_list(self):
        """리스트에서 MCP 설정 로드"""
        config_list = [
            {"cmd": "python server.py", "name": "test1"},
            {"url": "http://localhost:8080", "name": "test2"},
        ]

        configs = load_mcp_config(config_list)
        assert len(configs) == 2
        assert configs[0].transport == "stdio"
        assert configs[1].transport == "streamable_http"

    def test_type_conversion(self):
        """type 필드를 transport로 변환"""
        config_data = {
            "mcpServers": [
                {"type": "stdio", "cmd": "python server.py", "name": "test"},
                {"type": "streamable_http", "url": "http://localhost:8080", "name": "test2"},
            ]
        }

        configs = load_mcp_config(config_data)
        assert len(configs) == 2
        assert configs[0].transport == "stdio"
        assert configs[1].transport == "streamable_http"

    def test_validation_missing_required_field(self):
        """type 필드 누락 시 에러 (name은 이제 선택적)"""
        config_dict = {
            "mcpServers": [
                {
                    # cmd와 url 모두 누락
                    "name": "test"
                }
            ]
        }

        with pytest.raises(ValueError, match="'cmd', 'command', 또는 'url' 중 하나는 반드시 지정되어야 합니다"):
            load_mcp_config(config_dict)

    def test_validation_invalid_type(self):
        """잘못된 설정 타입"""
        with pytest.raises(TypeError):
            load_mcp_config(123)

    def test_validation_missing_cmd_for_stdio(self):
        """stdio transport에 cmd 누락"""
        config_dict = {
            "mcpServers": [
                {
                    "transport": "stdio",
                    "name": "test"
                    # cmd 누락
                }
            ]
        }

        with pytest.raises(ValueError, match="stdio transport에는 'cmd' 또는 'command' 필드가 필요합니다"):
            load_mcp_config(config_dict)

    def test_validation_missing_url_for_http(self):
        """http transport에 url 누락"""
        config_dict = {
            "mcpServers": [
                {
                    "transport": "streamable_http",
                    "name": "test"
                    # url 누락
                }
            ]
        }

        with pytest.raises(ValueError, match="streamable_http transport에는 'url' 필드가 필요합니다"):
            load_mcp_config(config_dict)

    def test_validation_invalid_url_format(self):
        """잘못된 URL 형식"""
        config_dict = {
            "mcpServers": [
                {
                    "url": "invalid-url",
                    "name": "test"
                }
            ]
        }

        with pytest.raises(ValueError, match="지원하지 않는 URL 스키마"):
            load_mcp_config(config_dict)

    def test_env_normalization(self):
        """환경 변수 정규화"""
        config_dict = {
            "mcpServers": [
                {
                    "cmd": "python server.py",
                    "name": "test",
                    "env": {"DEBUG": "1", "PORT": "8080"},
                }
            ]
        }

        configs = load_mcp_config(config_dict)
        assert len(configs) == 1
        assert configs[0].env == {"DEBUG": "1", "PORT": "8080"}

    def test_file_not_found(self):
        """존재하지 않는 파일"""
        with pytest.raises(FileNotFoundError):
            load_mcp_config("/nonexistent/file.json")

    def test_invalid_json(self):
        """잘못된 JSON 파일"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_mcp_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_empty_config(self):
        """빈 설정"""
        configs = load_mcp_config([])
        assert len(configs) == 0

    def test_mixed_llm_and_mcp_config(self):
        """LLM과 MCP 설정이 혼재된 경우"""
        config_data = {
            "llm": {"model": "gpt-4"},
            "mcpServers": [
                {"cmd": "python server.py", "name": "test"},
            ],
        }

        configs = load_mcp_config(config_data)
        assert len(configs) == 1
        assert configs[0].name == "test"

    def test_command_args_conversion(self):
        """command + args를 cmd로 변환"""
        config_dict = {
            "mcpServers": [
                {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["server.py", "--port", "8080"],
                    "name": "test"
                }
            ]
        }

        configs = load_mcp_config(config_dict)
        assert len(configs) == 1
        assert configs[0].cmd == ["python", "server.py", "--port", "8080"]
        assert configs[0].command == "python"
        assert configs[0].args == ["server.py", "--port", "8080"]