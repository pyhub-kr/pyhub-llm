"""MCP 서버 설정을 위한 통합 Config 클래스"""

import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from .policies import MCPConnectionPolicy

__all__ = [
    "McpConfig",
    "create_mcp_config",
]


@dataclass(kw_only=True)
class McpConfig:
    """통합 MCP 서버 설정

    모든 transport 타입(stdio, http, websocket, sse)을 지원하는 통합 설정 클래스입니다.
    transport 타입은 제공된 필드들을 기반으로 자동으로 감지됩니다.

    Examples:
        >>> # STDIO 서버
        >>> config = McpConfig(cmd="python server.py")

        >>> # HTTP 서버
        >>> config = McpConfig(url="http://localhost:8080/mcp")

        >>> # WebSocket 서버
        >>> config = McpConfig(url="ws://localhost:8080/ws")

        >>> # 명시적 transport 지정
        >>> config = McpConfig(
        ...     cmd="python server.py",
        ...     transport="stdio",
        ...     env={"DEBUG": "1"}
        ... )
    """

    # 공통 필드
    name: Optional[str] = None  # 서버 식별자 (선택적 - 서버에서 자동으로 가져올 수 있음)
    transport: Optional[str] = None  # transport 타입 (자동 감지 또는 명시 지정)
    filter_tools: Optional[List[str]] = None  # 필터링할 도구 목록
    timeout: int = 30  # 연결 타임아웃
    policy: MCPConnectionPolicy = MCPConnectionPolicy.OPTIONAL  # 연결 정책

    # STDIO transport 필드들
    cmd: Optional[Union[str, List[str]]] = None  # 실행할 명령 (문자열 또는 리스트)
    env: Optional[Dict[str, str]] = None  # 환경 변수
    cwd: Optional[str] = None  # 작업 디렉토리

    # HTTP/WebSocket/SSE transport 필드들
    url: Optional[str] = None  # 서버 URL
    headers: Optional[Dict[str, str]] = None  # HTTP 헤더

    # 내부 사용 필드들 (STDIO용)
    _command: Optional[str] = field(init=False, default=None)
    _args: List[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        """초기화 후 처리: transport 감지 및 검증"""
        # transport 자동 감지
        if not self.transport:
            self.transport = self._detect_transport()

        # STDIO 명령어 파싱
        if self.transport == "stdio" and self.cmd:
            self._parse_cmd()

        # 설정 검증
        self._validate_config()

    def _detect_transport(self) -> str:
        """설정 내용을 기반으로 transport 타입 자동 감지"""
        if self.cmd:
            return "stdio"
        elif self.url:
            parsed = urlparse(self.url)
            if parsed.scheme in ("ws", "wss"):
                return "websocket"
            elif "/sse" in self.url or (self.headers and "text/event-stream" in str(self.headers).lower()):
                return "sse"
            elif parsed.scheme in ("http", "https"):
                return "streamable_http"
            else:
                raise ValueError(f"지원하지 않는 URL 스키마: {parsed.scheme}")
        else:
            raise ValueError("cmd 또는 url 중 하나는 반드시 지정되어야 합니다")

    def _parse_cmd(self):
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

    def _validate_config(self):
        """설정 유효성 검증"""
        if self.transport == "stdio":
            if not self.cmd:
                raise ValueError("stdio transport에는 cmd가 필요합니다")
        elif self.transport in ("streamable_http", "websocket", "sse"):
            if not self.url:
                raise ValueError(f"{self.transport} transport에는 url이 필요합니다")
        else:
            raise ValueError(f"지원하지 않는 transport: {self.transport}")

    @property
    def command(self) -> Optional[str]:
        """STDIO transport용 명령어 반환"""
        return self._command

    @property
    def args(self) -> List[str]:
        """STDIO transport용 인수 리스트 반환"""
        return self._args.copy() if self._args else []

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        result = {
            "name": self.name,
            "transport": self.transport,
            "filter_tools": self.filter_tools,
            "timeout": self.timeout,
            "policy": self.policy,
        }

        # transport별 필드 추가
        if self.transport == "stdio":
            result.update(
                {
                    "command": self.command,
                    "args": self.args,
                    "env": self.env,
                    "cwd": self.cwd,
                }
            )
        elif self.transport in ("streamable_http", "websocket", "sse"):
            result.update(
                {
                    "url": self.url,
                    "headers": self.headers,
                }
            )

        return result


def create_mcp_config(config_input: Union[str, dict, McpConfig]) -> McpConfig:
    """다양한 입력을 McpConfig로 변환하는 Factory 함수

    Args:
        config_input: 다음 중 하나
            - str: 명령어 또는 URL 문자열
            - dict: 설정 딕셔너리
            - McpConfig: 기존 설정 객체

    Returns:
        McpConfig 인스턴스

    Examples:
        >>> # 문자열에서 생성
        >>> config = create_mcp_config("python server.py")  # stdio
        >>> config = create_mcp_config("http://localhost:8080")  # http
        >>> config = create_mcp_config("ws://localhost:8080")  # websocket

        >>> # 딕셔너리에서 생성
        >>> config = create_mcp_config({
        ...     "cmd": "python server.py",
        ...     "env": {"DEBUG": "1"}
        ... })

        >>> # 기존 객체 반환
        >>> config = create_mcp_config(existing_config)
    """
    if isinstance(config_input, McpConfig):
        return config_input
    elif isinstance(config_input, dict):
        return McpConfig(**config_input)
    elif isinstance(config_input, str):
        return _parse_string_config(config_input)
    else:
        raise TypeError(f"지원하지 않는 config 타입: {type(config_input)}")


def _parse_string_config(config_str: str) -> McpConfig:
    """문자열을 파싱하여 McpConfig 생성

    Args:
        config_str: 파싱할 문자열

    Returns:
        McpConfig 인스턴스

    Examples:
        >>> config = _parse_string_config("python server.py")
        >>> config.transport  # "stdio"
        >>> config.cmd  # "python server.py"

        >>> config = _parse_string_config("http://localhost:8080")
        >>> config.transport  # "streamable_http"
        >>> config.url  # "http://localhost:8080"
    """
    config_str = config_str.strip()

    # URL 패턴 감지 (http, https, ws, wss로 시작)
    if config_str.startswith(("http://", "https://", "ws://", "wss://")):
        return McpConfig(url=config_str)

    # 명령어 패턴으로 간주 (stdio)
    else:
        return McpConfig(cmd=config_str)
