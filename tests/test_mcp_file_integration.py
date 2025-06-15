import json
import tempfile
from pathlib import Path

import pytest

try:
    import yaml  # noqa: F401

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from pyhub.llm import LLM
from pyhub.llm.mcp import McpConfig


class TestMCPFileIntegration:
    """MCP 설정 파일 통합 테스트"""

    def test_llm_create_with_json_file(self):
        """JSON 파일로 MCP 설정을 전달하는 경우"""
        config_data = {
            "mcpServers": [{"type": "stdio", "name": "calculator", "cmd": "python calculator.py", "timeout": 60}]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            # 파일 경로를 문자열로 전달
            llm = LLM.create("gpt-4o-mini", mcp_servers=temp_path)
            assert len(llm.mcp_servers) == 1
            assert isinstance(llm.mcp_servers[0], McpConfig)
            assert llm.mcp_servers[0].name == "calculator"
            assert llm.mcp_servers[0].timeout == 60
        finally:
            Path(temp_path).unlink()

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_llm_create_with_yaml_file(self):
        """YAML 파일로 MCP 설정을 전달하는 경우"""
        yaml_content = """
mcpServers:
  - type: stdio
    name: calc
    cmd: pyhub-llm mcp-server run calculator
  - type: streamable_http
    name: web
    url: http://localhost:8888/mcp
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            llm = LLM.create("gpt-4o-mini", mcp_servers=temp_path)
            assert len(llm.mcp_servers) == 2
            assert llm.mcp_servers[0].name == "calc"
            assert llm.mcp_servers[1].name == "web"
        finally:
            Path(temp_path).unlink()

    def test_llm_create_with_dict_config(self):
        """dict로 MCP 설정을 전달하는 경우"""
        config_dict = {"mcpServers": [{"type": "stdio", "name": "test", "cmd": "test command"}]}

        llm = LLM.create("gpt-4o-mini", mcp_servers=config_dict)
        assert len(llm.mcp_servers) == 1
        assert llm.mcp_servers[0].name == "test"

    def test_llm_create_with_list_of_dicts(self):
        """dict 리스트로 MCP 설정을 전달하는 경우"""
        config_list = [
            {"type": "stdio", "name": "server1", "cmd": "cmd1"},
            {"type": "stdio", "name": "server2", "cmd": "cmd2"},
        ]

        llm = LLM.create("gpt-4o-mini", mcp_servers=config_list)
        assert len(llm.mcp_servers) == 2
        assert llm.mcp_servers[0].name == "server1"
        assert llm.mcp_servers[1].name == "server2"

    def test_llm_create_with_single_config_object(self):
        """단일 McpConfig 객체로 전달하는 경우"""
        config = McpConfig(name="single", cmd="single command")

        llm = LLM.create("gpt-4o-mini", mcp_servers=config)
        assert len(llm.mcp_servers) == 1
        assert llm.mcp_servers[0] == config

    def test_llm_create_with_list_of_config_objects(self):
        """McpConfig 객체 리스트로 전달하는 경우"""
        configs = [
            McpConfig(name="config1", cmd="cmd1"),
            McpConfig(name="config2", url="http://localhost:8080"),
        ]

        llm = LLM.create("gpt-4o-mini", mcp_servers=configs)
        assert len(llm.mcp_servers) == 2
        assert llm.mcp_servers == configs

    def test_llm_create_with_invalid_file(self):
        """존재하지 않는 파일 경로를 전달하는 경우"""
        with pytest.raises(FileNotFoundError):
            LLM.create("gpt-4o-mini", mcp_servers="/nonexistent/file.json")

    def test_llm_create_with_invalid_config(self):
        """잘못된 설정을 전달하는 경우"""
        invalid_config = {
            "mcpServers": [
                {
                    # type 필드 누락 (name은 이제 선택적)
                    "cmd": "test"
                }
            ]
        }

        # cmd만 있는 경우 정상 작동해야 함
        llm = LLM.create("gpt-4o-mini", mcp_servers=invalid_config)
        assert len(llm.mcp_servers) == 1
        assert llm.mcp_servers[0].cmd == "test"

    @pytest.mark.asyncio
    async def test_llm_create_async_with_file(self):
        """create_async로 파일 경로를 전달하는 경우"""
        config_data = {
            "mcpServers": [
                {
                    "type": "stdio",
                    "name": "async_test",
                    "cmd": "test command",
                    "timeout": "30",  # 문자열로 전달해도 정상 변환
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with patch("pyhub.llm.base.BaseLLM.initialize_mcp", new=AsyncMock()) as mock_init:
                llm = await LLM.create_async("gpt-4o-mini", mcp_servers=temp_path)
                assert len(llm.mcp_servers) == 1
                assert llm.mcp_servers[0].timeout == "30"  # 문자열로 유지됨
                mock_init.assert_called_once()
        finally:
            Path(temp_path).unlink()

    def test_mixed_config_file(self):
        """LLM 설정과 MCP 설정이 함께 있는 파일"""
        config_data = {
            "model": "gpt-4o",
            "temperature": 0.8,
            "mcpServers": [
                {
                    "type": "stdio",
                    "name": "calc",
                    "cmd": "python calc.py",
                    "filter_tools": "add,subtract,multiply",  # 문자열로 전달
                }
            ],
            "cache": {"type": "memory"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            llm = LLM.create("gpt-4o-mini", mcp_servers=temp_path)
            assert len(llm.mcp_servers) == 1
            assert llm.mcp_servers[0].filter_tools == "add,subtract,multiply"  # 문자열로 유지됨
        finally:
            Path(temp_path).unlink()


# Import mocking을 위한 추가
from unittest.mock import AsyncMock, patch
