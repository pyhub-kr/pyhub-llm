"""MCP 서버 설정을 위한 Dataclass 정의"""

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

__all__ = [
    "McpServerConfig",
    "McpStdioConfig",
    "McpStreamableHttpConfig",
    "McpWebSocketConfig",
    "McpSseConfig",
]


@dataclass(kw_only=True)
class McpServerConfig:
    """MCP 서버 설정 기본 클래스"""
    name: str  # 서버 식별자
    description: Optional[str] = None
    filter_tools: Optional[List[str]] = None
    timeout: int = 30
    
    def to_dict(self) -> Dict[str, any]:
        """설정을 딕셔너리로 변환"""
        return {
            "transport": getattr(self, "transport", None),
            "description": self.description,
            "filter_tools": self.filter_tools,
            "timeout": self.timeout,
        }


@dataclass
class McpStdioConfig(McpServerConfig):
    """STDIO transport용 MCP 서버 설정
    
    Examples:
        >>> # 문자열로 간단히 지정
        >>> config = McpStdioConfig(
        ...     name="calculator",
        ...     cmd="pyhub-llm mcp-server run calculator"
        ... )
        
        >>> # 공백이 포함된 경로
        >>> config = McpStdioConfig(
        ...     name="custom",
        ...     cmd="python '/path/to/my server.py' --config 'config.json'"
        ... )
        
        >>> # 리스트로 정확히 제어
        >>> config = McpStdioConfig(
        ...     name="custom",
        ...     cmd=["python", "/path/to/server.py", "--port", "8080"]
        ... )
    """
    cmd: Union[str, List[str]]  # 실행할 명령 (문자열 또는 리스트)
    transport: Literal["stdio"] = "stdio"
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    
    # 내부 사용 필드
    _command: Optional[str] = field(init=False, default=None)
    _args: List[str] = field(init=False, default_factory=list)
    
    def __post_init__(self):
        """cmd 필드를 command와 args로 자동 분리"""
        if isinstance(self.cmd, str):
            parts = shlex.split(self.cmd)
            if not parts:
                raise ValueError("cmd 문자열이 비어있습니다")
            self._command = parts[0]
            self._args = parts[1:] if len(parts) > 1 else []
        elif isinstance(self.cmd, list):
            if not self.cmd:
                raise ValueError("cmd 리스트가 비어있습니다")
            self._command = self.cmd[0]
            self._args = list(self.cmd[1:]) if len(self.cmd) > 1 else []
        else:
            raise TypeError(f"cmd는 str 또는 List[str]이어야 합니다: {type(self.cmd)}")
    
    @property
    def command(self) -> str:
        """실행할 명령어"""
        return self._command
    
    @property
    def args(self) -> List[str]:
        """명령어 인자들"""
        return self._args
    
    def to_dict(self) -> Dict[str, any]:
        """설정을 딕셔너리로 변환"""
        result = super().to_dict()
        result.update({
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "cwd": self.cwd,
        })
        return result


@dataclass
class McpStreamableHttpConfig(McpServerConfig):
    """HTTP/SSE transport용 MCP 서버 설정
    
    Examples:
        >>> config = McpStreamableHttpConfig(
        ...     name="greeting",
        ...     url="http://localhost:8888/mcp",
        ...     description="다국어 인사말 서버"
        ... )
    """
    url: str
    transport: Literal["streamable_http"] = "streamable_http"
    headers: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, any]:
        """설정을 딕셔너리로 변환"""
        result = super().to_dict()
        result.update({
            "url": self.url,
            "headers": self.headers,
        })
        return result


@dataclass
class McpWebSocketConfig(McpServerConfig):
    """WebSocket transport용 MCP 서버 설정
    
    Examples:
        >>> config = McpWebSocketConfig(
        ...     name="realtime",
        ...     url="ws://localhost:8080/mcp/ws"
        ... )
    """
    url: str
    transport: Literal["websocket"] = "websocket"
    headers: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, any]:
        """설정을 딕셔너리로 변환"""
        result = super().to_dict()
        result.update({
            "url": self.url,
            "headers": self.headers,
        })
        return result


@dataclass
class McpSseConfig(McpServerConfig):
    """SSE transport용 MCP 서버 설정
    
    Examples:
        >>> config = McpSseConfig(
        ...     name="events",
        ...     url="http://localhost:8080/mcp/sse"
        ... )
    """
    url: str
    transport: Literal["sse"] = "sse"
    headers: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, any]:
        """설정을 딕셔너리로 변환"""
        result = super().to_dict()
        result.update({
            "url": self.url,
            "headers": self.headers,
        })
        return result