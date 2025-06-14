"""MCP 설정 팩토리 및 자동 감지 테스트"""

import pytest

from pyhub.llm.mcp.configs import McpConfig, create_mcp_config


class TestCreateMcpConfig:
    """create_mcp_config 팩토리 함수 테스트"""

    def test_create_from_string_stdio(self):
        """문자열로 stdio 설정 생성"""
        config = create_mcp_config("python calculator.py")
        
        assert isinstance(config, McpConfig)
        assert config.transport == "stdio"
        assert config.cmd == "python calculator.py"
        assert config.name is None  # 자동 생성되지 않음

    def test_create_from_string_http_url(self):
        """HTTP URL 문자열로 설정 생성"""
        config = create_mcp_config("http://localhost:8080/mcp")
        
        assert isinstance(config, McpConfig)
        assert config.transport == "streamable_http"
        assert config.url == "http://localhost:8080/mcp"
        assert config.name is None

    def test_create_from_string_https_url(self):
        """HTTPS URL 문자열로 설정 생성"""
        config = create_mcp_config("https://api.example.com/mcp")
        
        assert isinstance(config, McpConfig)
        assert config.transport == "streamable_http"
        assert config.url == "https://api.example.com/mcp"

    def test_create_from_string_websocket_url(self):
        """WebSocket URL 문자열로 설정 생성"""
        config = create_mcp_config("ws://localhost:8080/ws")
        
        assert isinstance(config, McpConfig)
        assert config.transport == "websocket"
        assert config.url == "ws://localhost:8080/ws"

    def test_create_from_string_wss_url(self):
        """WSS URL 문자열로 설정 생성"""
        config = create_mcp_config("wss://api.example.com/ws")
        
        assert isinstance(config, McpConfig)
        assert config.transport == "websocket"
        assert config.url == "wss://api.example.com/ws"

    def test_create_from_dict_stdio(self):
        """딕셔너리로 stdio 설정 생성"""
        config_dict = {
            "cmd": "python server.py",
            "name": "test_server",
            "timeout": 30
        }
        
        config = create_mcp_config(config_dict)
        
        assert isinstance(config, McpConfig)
        assert config.transport == "stdio"
        assert config.cmd == "python server.py"
        assert config.name == "test_server"
        assert config.timeout == 30

    def test_create_from_dict_with_explicit_transport(self):
        """transport가 명시된 딕셔너리"""
        config_dict = {
            "transport": "streamable_http",
            "url": "http://localhost:8080",
            "name": "http_server"
        }
        
        config = create_mcp_config(config_dict)
        
        assert isinstance(config, McpConfig)
        assert config.transport == "streamable_http"
        assert config.url == "http://localhost:8080"
        assert config.name == "http_server"

    def test_create_from_dict_command_args(self):
        """command + args 조합"""
        config_dict = {
            "cmd": ["python", "server.py", "--port", "8080"],
            "name": "cmd_args_server"
        }
        
        config = create_mcp_config(config_dict)
        
        assert isinstance(config, McpConfig)
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.args == ["server.py", "--port", "8080"]
        assert config.cmd == ["python", "server.py", "--port", "8080"]

    def test_create_from_existing_config(self):
        """기존 McpConfig 객체 전달"""
        original = McpConfig(name="test", cmd="python test.py")
        config = create_mcp_config(original)
        
        # 동일한 객체를 반환해야 함
        assert config is original

    def test_cmd_string_fallback(self):
        """인식되지 않는 문자열은 cmd로 처리"""
        config = create_mcp_config("arbitrary-command-string")
        assert config.transport == "stdio"
        assert config.cmd == "arbitrary-command-string"

    def test_invalid_dict_input(self):
        """필수 필드가 없는 딕셔너리"""
        with pytest.raises(ValueError, match="cmd 또는 url 중 하나는 반드시 지정되어야 합니다"):
            create_mcp_config({"name": "test"})

    def test_invalid_input_type(self):
        """지원하지 않는 입력 타입"""
        with pytest.raises(TypeError, match="지원하지 않는 config 타입"):
            create_mcp_config(123)


class TestTransportAutoDetection:
    """Transport 자동 감지 테스트"""

    def test_detect_stdio_from_cmd(self):
        """cmd 필드로 stdio 감지"""
        config = McpConfig(cmd="python server.py")
        assert config.transport == "stdio"

    def test_detect_stdio_from_command(self):
        """command 필드로 stdio 감지"""
        config = McpConfig(cmd=["python", "server.py"])
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.args == ["server.py"]

    def test_detect_http_from_url(self):
        """HTTP URL로 streamable_http 감지"""
        config = McpConfig(url="http://localhost:8080")
        assert config.transport == "streamable_http"

    def test_detect_https_from_url(self):
        """HTTPS URL로 streamable_http 감지"""
        config = McpConfig(url="https://api.example.com")
        assert config.transport == "streamable_http"

    def test_detect_websocket_from_ws_url(self):
        """WS URL로 websocket 감지"""
        config = McpConfig(url="ws://localhost:8080")
        assert config.transport == "websocket"

    def test_detect_websocket_from_wss_url(self):
        """WSS URL로 websocket 감지"""
        config = McpConfig(url="wss://api.example.com")
        assert config.transport == "websocket"

    def test_sse_endpoint_detection(self):
        """SSE 엔드포인트 패턴 감지"""
        config = McpConfig(url="http://localhost:8080/sse")
        assert config.transport == "sse"

    def test_sse_stream_detection(self):
        """SSE 스트림 패턴 감지"""
        config = McpConfig(url="http://localhost:8080/stream", headers={"accept": "text/event-stream"})
        assert config.transport == "sse"

    def test_explicit_transport_override(self):
        """명시적 transport가 자동 감지를 우선함"""
        config = McpConfig(
            transport="sse",
            url="http://localhost:8080"  # 일반적으로는 streamable_http로 감지될 URL
        )
        assert config.transport == "sse"

    def test_no_detection_possible(self):
        """감지할 수 없는 경우"""
        with pytest.raises(ValueError, match="cmd 또는 url 중 하나는 반드시 지정되어야 합니다"):
            McpConfig(name="test_only")


class TestMcpConfigValidation:
    """McpConfig 검증 테스트"""

    def test_valid_stdio_config(self):
        """유효한 stdio 설정"""
        config = McpConfig(name="test", cmd="python server.py", timeout=30)
        
        assert config.name == "test"
        assert config.cmd == "python server.py"
        assert config.timeout == 30
        assert config.transport == "stdio"

    def test_valid_http_config(self):
        """유효한 HTTP 설정"""
        config = McpConfig(
            name="http_server",
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer token"}
        )
        
        assert config.name == "http_server"
        assert config.url == "http://localhost:8080/mcp"
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.transport == "streamable_http"

    def test_cmd_string_parsing(self):
        """문자열 cmd 파싱"""
        config = McpConfig(cmd="python '/path/with spaces/server.py' --config 'config.json'")
        
        assert config.command == "python"
        assert config.args == ["/path/with spaces/server.py", "--config", "config.json"]

    def test_cmd_list_handling(self):
        """리스트 cmd 처리"""
        cmd_list = ["python", "/path/to/server.py", "--port", "8080"]
        config = McpConfig(cmd=cmd_list)
        
        assert config.command == "python"
        assert config.args == ["/path/to/server.py", "--port", "8080"]
        assert config.cmd == cmd_list

    def test_to_dict_conversion(self):
        """to_dict() 메서드 테스트"""
        config = McpConfig(
            name="test_server",
            cmd="python server.py",
            timeout=60,
            env={"DEBUG": "1"}
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["name"] == "test_server"
        assert config_dict["transport"] == "stdio"
        assert config_dict["command"] == "python"
        assert config_dict["args"] == ["server.py"]
        assert config_dict["timeout"] == 60
        assert config_dict["env"] == {"DEBUG": "1"}

    def test_to_dict_with_url(self):
        """URL이 있는 설정의 to_dict()"""
        config = McpConfig(
            name="web_server",
            url="https://api.example.com/mcp",
            headers={"Content-Type": "application/json"}
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["name"] == "web_server"
        assert config_dict["transport"] == "streamable_http"
        assert config_dict["url"] == "https://api.example.com/mcp"
        assert config_dict["headers"] == {"Content-Type": "application/json"}

    def test_optional_fields(self):
        """선택적 필드들"""
        config = McpConfig(cmd="python server.py")  # name 없이
        
        assert config.name is None
        assert config.cmd == "python server.py"
        assert config.transport == "stdio"

    def test_policy_field(self):
        """정책 필드 테스트"""
        from pyhub.llm.mcp.policies import MCPConnectionPolicy
        
        config = McpConfig(
            cmd="python server.py",
            policy=MCPConnectionPolicy.REQUIRED
        )
        
        assert config.policy == MCPConnectionPolicy.REQUIRED
        
        config_dict = config.to_dict()
        assert config_dict["policy"] == MCPConnectionPolicy.REQUIRED