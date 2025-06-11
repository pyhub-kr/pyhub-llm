"""MCPClient의 Config 지원 테스트"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pyhub.llm.mcp import MCPClient
from pyhub.llm.mcp.configs import McpStdioConfig, McpStreamableHttpConfig


class TestMCPClientConfig:
    """MCPClient의 다양한 설정 방식 테스트"""
    
    def test_mcpclient_with_dataclass_config(self):
        """Dataclass config로 MCPClient 생성"""
        # STDIO config
        stdio_config = McpStdioConfig(
            name="test_calculator",
            cmd="python calculator.py"
        )
        
        client = MCPClient(stdio_config)
        assert client.transport is not None
        assert client.server_params is None
        
        # HTTP config
        http_config = McpStreamableHttpConfig(
            name="test_http",
            url="http://localhost:8080/mcp"
        )
        
        client = MCPClient(http_config)
        assert client.transport is not None
        assert client.server_params is None
    
    def test_mcpclient_with_dict_config(self):
        """기존 dict 방식으로 MCPClient 생성"""
        config = {
            "transport": "stdio",
            "command": "python",
            "args": ["calculator.py"]
        }
        
        client = MCPClient(config)
        assert client.transport is not None
        assert client.server_params is None
    
    def test_mcpclient_with_legacy_params(self):
        """레거시 StdioServerParameters로 MCPClient 생성"""
        # Mock StdioServerParameters
        mock_params = MagicMock()
        mock_params.command = "python"
        mock_params.args = ["calculator.py"]
        
        client = MCPClient(mock_params)
        assert client.transport is None
        assert client.server_params == mock_params
    
    def test_config_to_dict_conversion(self):
        """Config의 to_dict() 메서드 테스트"""
        config = McpStdioConfig(
            name="test",
            cmd=["python", "server.py", "--port", "8080"],
            description="Test server"
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["transport"] == "stdio"
        assert config_dict["command"] == "python"
        assert config_dict["args"] == ["server.py", "--port", "8080"]
        assert config_dict["description"] == "Test server"
    
    def test_cmd_parsing_string(self):
        """문자열 cmd 파싱 테스트"""
        config = McpStdioConfig(
            name="test",
            cmd="python '/path/to/my server.py' --config 'config.json'"
        )
        
        assert config.command == "python"
        assert config.args == ["/path/to/my server.py", "--config", "config.json"]
    
    def test_cmd_parsing_list(self):
        """리스트 cmd 파싱 테스트"""
        config = McpStdioConfig(
            name="test",
            cmd=["python", "/path/to/server.py", "--config", "config.json"]
        )
        
        assert config.command == "python"
        assert config.args == ["/path/to/server.py", "--config", "config.json"]
    
    @pytest.mark.asyncio
    async def test_mcpclient_connect_with_config(self):
        """Config를 사용한 MCPClient 연결 테스트"""
        config = McpStdioConfig(
            name="test",
            cmd="echo test"
        )
        
        # Mock transport and session
        mock_transport = MagicMock()
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_session = AsyncMock()
        
        # Mock transport.connect() to return an async context manager
        mock_connect_cm = AsyncMock()
        mock_connect_cm.__aenter__.return_value = (mock_read, mock_write)
        mock_connect_cm.__aexit__.return_value = None
        mock_transport.connect.return_value = mock_connect_cm
        
        # Mock session as async context manager
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.initialize = AsyncMock()
        
        with patch('pyhub.llm.mcp.client.create_transport', return_value=mock_transport):
            with patch('mcp.ClientSession', return_value=mock_session):
                client = MCPClient(config)
                
                async with client.connect() as connected_client:
                    assert connected_client._session == mock_session
                    mock_session.initialize.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])