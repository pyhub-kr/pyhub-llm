"""MCP 연결 정책 테스트 (수정 버전)"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import logging

from pyhub.llm import LLM
from pyhub.llm.mcp.configs import McpStdioConfig
from pyhub.llm.mcp.policies import MCPConnectionPolicy, MCPConnectionError


@pytest.mark.asyncio
class TestMCPConnectionPolicy:
    """MCP 연결 정책 테스트"""
    
    async def test_optional_policy_default(self):
        """OPTIONAL 정책 (기본값) 테스트"""
        # 실제 연결 실패를 시뮬레이션 (존재하지 않는 명령)
        llm = LLM.create("gpt-4o-mini", mcp_servers=[{
            "type": "stdio",
            "name": "test_server",
            "cmd": ["nonexistent_command_xyz"]
        }])
        
        # MCP 초기화 - 실패해도 예외 없음
        await llm.initialize_mcp()
        
        # 연결은 시도되었지만 실패
        assert llm._mcp_connected is True  # MultiServerMCPClient는 생성됨
        assert len(llm._mcp_tools) == 0   # 하지만 도구는 없음
        assert llm.mcp_policy == MCPConnectionPolicy.OPTIONAL
        
        await llm.close_mcp()
    
    async def test_required_policy_failure(self):
        """REQUIRED 정책 - 연결 실패 시 예외 발생"""
        # REQUIRED 정책으로 생성
        llm = LLM.create(
            "gpt-4o-mini",
            mcp_servers=[{
                "type": "stdio",
                "name": "test_server",
                "cmd": ["nonexistent_command_xyz"]
            }],
            mcp_policy=MCPConnectionPolicy.REQUIRED
        )
        
        # MCP 초기화 시 예외 발생해야 함
        with pytest.raises(MCPConnectionError) as exc_info:
            await llm.initialize_mcp()
        
        assert "Failed to connect to MCP servers" in str(exc_info.value)
        assert "test_server" in exc_info.value.failed_servers
    
    async def test_required_policy_with_echo(self):
        """REQUIRED 정책 - echo 명령으로 테스트"""
        # echo는 대부분의 시스템에 존재하는 명령
        llm = LLM.create(
            "gpt-4o-mini",
            mcp_servers=[{
                "type": "stdio",
                "name": "echo_server",
                "cmd": ["echo", "hello"]  # 간단한 echo 명령
            }],
            mcp_policy=MCPConnectionPolicy.REQUIRED
        )
        
        # echo 명령은 MCP 프로토콜을 구현하지 않으므로 실패할 것
        with pytest.raises(MCPConnectionError):
            await llm.initialize_mcp()
    
    async def test_warn_policy(self):
        """WARN 정책 - 경고만 표시"""
        # 로거 설정
        logger = logging.getLogger('pyhub.llm.base')
        
        # 핸들러를 추가하여 로그 캡처
        captured_logs = []
        handler = logging.Handler()
        handler.emit = lambda record: captured_logs.append(record)
        logger.addHandler(handler)
        
        try:
            # WARN 정책으로 생성
            llm = LLM.create(
                "gpt-4o-mini",
                mcp_servers=[{
                    "type": "stdio",
                    "name": "test_server",
                    "cmd": ["nonexistent_command_xyz"]
                }],
                mcp_policy=MCPConnectionPolicy.WARN
            )
            
            # MCP 초기화
            await llm.initialize_mcp()
            
            # 경고 로그 확인
            warning_logs = [log for log in captured_logs if log.levelno == logging.WARNING]
            assert any("Failed to connect to some MCP servers" in log.getMessage() for log in warning_logs)
            
            assert llm._mcp_connected is True
            await llm.close_mcp()
            
        finally:
            logger.removeHandler(handler)
    
    async def test_policy_with_multiple_servers(self):
        """여러 서버 중 일부만 실패하는 경우"""
        servers = [
            {"type": "stdio", "name": "server1", "cmd": ["echo", "1"]},
            {"type": "stdio", "name": "server2", "cmd": ["nonexistent_xyz"]},
        ]
        
        # WARN 정책
        llm = LLM.create(
            "gpt-4o-mini",
            mcp_servers=servers,
            mcp_policy=MCPConnectionPolicy.WARN
        )
        
        # 로거 설정
        logger = logging.getLogger('pyhub.llm.base')
        captured_logs = []
        handler = logging.Handler()
        handler.emit = lambda record: captured_logs.append(record)
        logger.addHandler(handler)
        
        try:
            await llm.initialize_mcp()
            
            # 경고 로그 확인
            warning_logs = [log for log in captured_logs if log.levelno == logging.WARNING]
            # server1과 server2 모두 실패할 것 (echo는 MCP 프로토콜이 아님)
            assert any("Failed to connect to some MCP servers" in log.getMessage() for log in warning_logs)
            
            assert llm._mcp_connected is True
            await llm.close_mcp()
            
        finally:
            logger.removeHandler(handler)
    
    async def test_config_with_policy(self):
        """McpStdioConfig에 정책 설정"""
        config = McpStdioConfig(
            name="test_server",
            cmd="echo test",
            policy=MCPConnectionPolicy.REQUIRED
        )
        
        assert config.policy == MCPConnectionPolicy.REQUIRED
        
        # to_dict에 정책 포함 확인
        config_dict = config.to_dict()
        assert config_dict["policy"] == MCPConnectionPolicy.REQUIRED
    
    async def test_policy_propagation(self):
        """정책이 제대로 전파되는지 확인"""
        # 1. 정책 없이 생성
        llm1 = LLM.create("gpt-4o-mini")
        assert llm1.mcp_policy is None
        
        # 2. OPTIONAL 정책
        llm2 = LLM.create("gpt-4o-mini", mcp_policy=MCPConnectionPolicy.OPTIONAL)
        assert llm2.mcp_policy == MCPConnectionPolicy.OPTIONAL
        
        # 3. REQUIRED 정책
        llm3 = LLM.create("gpt-4o-mini", mcp_policy=MCPConnectionPolicy.REQUIRED)
        assert llm3.mcp_policy == MCPConnectionPolicy.REQUIRED
        
        # 4. WARN 정책
        llm4 = LLM.create("gpt-4o-mini", mcp_policy=MCPConnectionPolicy.WARN)
        assert llm4.mcp_policy == MCPConnectionPolicy.WARN
    
    async def test_mcp_connection_error_attributes(self):
        """MCPConnectionError 속성 테스트"""
        error = MCPConnectionError("Test error", ["server1", "server2"])
        assert str(error) == "Test error"
        assert error.failed_servers == ["server1", "server2"]
        
        # failed_servers 없이도 생성 가능
        error2 = MCPConnectionError("Another error")
        assert error2.failed_servers == []