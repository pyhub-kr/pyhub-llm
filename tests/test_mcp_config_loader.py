import json
import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from pyhub.llm.mcp import (
    McpServerConfig,
    McpStdioConfig,
    McpStreamableHttpConfig,
    McpWebSocketConfig,
    McpSseConfig
)
from pyhub.llm.mcp.config_loader import load_mcp_config, validate_mcp_config, normalize_mcp_config


class TestMCPConfigLoader:
    """MCP 설정 파일 로더 테스트"""
    
    def test_load_from_json_file(self):
        """JSON 파일에서 MCP 설정 로드"""
        config_data = {
            "mcpServers": [
                {
                    "type": "stdio",
                    "name": "calculator",
                    "cmd": "python calculator.py",
                    "description": "계산기"
                },
                {
                    "type": "streamable_http",
                    "name": "web",
                    "url": "http://localhost:8888/mcp"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            configs = load_mcp_config(temp_path)
            assert len(configs) == 2
            assert isinstance(configs[0], McpStdioConfig)
            assert configs[0].name == "calculator"
            assert configs[0].cmd == "python calculator.py"
            assert isinstance(configs[1], McpStreamableHttpConfig)
            assert configs[1].name == "web"
            assert configs[1].url == "http://localhost:8888/mcp"
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_load_from_yaml_file(self):
        """YAML 파일에서 MCP 설정 로드"""
        yaml_content = """
mcpServers:
  - type: stdio
    name: calculator
    cmd: pyhub-llm mcp-server run calculator
    timeout: 60
  - type: websocket
    name: ws_server
    url: ws://localhost:8080/ws
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            configs = load_mcp_config(temp_path)
            assert len(configs) == 2
            assert isinstance(configs[0], McpStdioConfig)
            assert configs[0].timeout == 60
            assert isinstance(configs[1], McpWebSocketConfig)
        finally:
            Path(temp_path).unlink()
    
    def test_load_from_dict(self):
        """dict에서 직접 MCP 설정 로드"""
        config_dict = {
            "mcpServers": [
                {
                    "type": "sse",
                    "name": "sse_server",
                    "url": "http://localhost:8080/sse"
                }
            ]
        }
        
        configs = load_mcp_config(config_dict)
        assert len(configs) == 1
        assert isinstance(configs[0], McpSseConfig)
    
    def test_load_from_list(self):
        """리스트에서 직접 MCP 설정 로드"""
        config_list = [
            {
                "type": "stdio",
                "name": "test",
                "cmd": "test command"
            }
        ]
        
        configs = load_mcp_config(config_list)
        assert len(configs) == 1
        assert isinstance(configs[0], McpStdioConfig)
    
    def test_type_conversion(self):
        """타입 자동 변환 테스트"""
        config_dict = {
            "mcpServers": [
                {
                    "type": "stdio",
                    "name": "test",
                    "cmd": "test",
                    "timeout": "30",  # 문자열 -> 숫자
                    "filter_tools": "tool1,tool2,tool3"  # 문자열 -> 리스트
                }
            ]
        }
        
        configs = load_mcp_config(config_dict)
        assert configs[0].timeout == 30
        assert configs[0].filter_tools == ["tool1", "tool2", "tool3"]
    
    def test_validation_missing_required_field(self):
        """필수 필드 누락 시 에러"""
        config_dict = {
            "mcpServers": [
                {
                    "type": "stdio",
                    # name 필드 누락
                    "cmd": "test"
                }
            ]
        }
        
        with pytest.raises(ValueError, match="Missing required field 'name'"):
            load_mcp_config(config_dict)
    
    def test_validation_invalid_type(self):
        """잘못된 type 값"""
        config_dict = {
            "mcpServers": [
                {
                    "type": "invalid_type",
                    "name": "test",
                    "cmd": "test"
                }
            ]
        }
        
        with pytest.raises(ValueError, match="Invalid type"):
            load_mcp_config(config_dict)
    
    def test_validation_missing_cmd_for_stdio(self):
        """stdio 타입에서 cmd 필드 누락"""
        config_dict = {
            "mcpServers": [
                {
                    "type": "stdio",
                    "name": "test"
                    # cmd 필드 누락
                }
            ]
        }
        
        with pytest.raises(ValueError, match="'cmd' is required for stdio type"):
            load_mcp_config(config_dict)
    
    def test_validation_missing_url_for_http(self):
        """HTTP 타입에서 url 필드 누락"""
        config_dict = {
            "mcpServers": [
                {
                    "type": "streamable_http",
                    "name": "test"
                    # url 필드 누락
                }
            ]
        }
        
        with pytest.raises(ValueError, match="'url' is required for streamable_http type"):
            load_mcp_config(config_dict)
    
    def test_validation_invalid_url_format(self):
        """잘못된 URL 형식"""
        config_dict = {
            "mcpServers": [
                {
                    "type": "streamable_http",
                    "name": "test",
                    "url": "not a valid url"
                }
            ]
        }
        
        with pytest.raises(ValueError, match="Invalid URL format"):
            load_mcp_config(config_dict)
    
    def test_env_normalization(self):
        """환경 변수 정규화"""
        config_dict = {
            "mcpServers": [
                {
                    "type": "stdio",
                    "name": "test",
                    "cmd": "test",
                    "env": {
                        "PATH": "/usr/bin",
                        "NUMBER": 123  # 숫자 -> 문자열
                    }
                }
            ]
        }
        
        configs = load_mcp_config(config_dict)
        assert configs[0].env["PATH"] == "/usr/bin"
        assert configs[0].env["NUMBER"] == "123"
    
    def test_file_not_found(self):
        """존재하지 않는 파일"""
        with pytest.raises(FileNotFoundError):
            load_mcp_config("/nonexistent/file.json")
    
    def test_invalid_json(self):
        """잘못된 JSON 파일"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_mcp_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_empty_config(self):
        """빈 설정"""
        config_dict = {"mcpServers": []}
        configs = load_mcp_config(config_dict)
        assert configs == []
    
    def test_mixed_llm_and_mcp_config(self):
        """LLM 설정과 MCP 설정이 함께 있는 경우"""
        config_dict = {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "mcpServers": [
                {
                    "type": "stdio",
                    "name": "calc",
                    "cmd": "python calc.py"
                }
            ],
            "cache": {
                "type": "memory",
                "ttl": 3600
            }
        }
        
        configs = load_mcp_config(config_dict)
        assert len(configs) == 1
        assert configs[0].name == "calc"
    
    def test_normalize_filter_tools(self):
        """filter_tools 정규화 테스트"""
        test_cases = [
            ("tool1,tool2,tool3", ["tool1", "tool2", "tool3"]),
            ("tool1, tool2, tool3", ["tool1", "tool2", "tool3"]),  # 공백 처리
            (["tool1", "tool2"], ["tool1", "tool2"]),  # 이미 리스트인 경우
            (None, None),  # None 값
        ]
        
        for input_val, expected in test_cases:
            config_dict = {
                "mcpServers": [{
                    "type": "stdio",
                    "name": "test",
                    "cmd": "test",
                    "filter_tools": input_val
                }]
            }
            
            configs = load_mcp_config(config_dict)
            assert configs[0].filter_tools == expected