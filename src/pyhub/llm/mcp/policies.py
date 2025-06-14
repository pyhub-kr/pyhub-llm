"""MCP 연결 정책 정의"""

from enum import Enum


class MCPConnectionPolicy(Enum):
    """MCP 연결 실패 시 동작 정책

    Attributes:
        OPTIONAL: MCP 연결 실패를 무시하고 계속 진행 (기본값)
        REQUIRED: MCP 연결 실패 시 예외 발생
        WARN: MCP 연결 실패 시 경고만 하고 계속 진행
    """

    OPTIONAL = "optional"  # 실패해도 계속 (현재 동작)
    REQUIRED = "required"  # 실패 시 예외 발생
    WARN = "warn"  # 실패 시 경고 후 계속


class MCPConnectionError(Exception):
    """MCP 연결 실패 예외"""

    def __init__(self, message: str, failed_servers: list = None):
        super().__init__(message)
        self.failed_servers = failed_servers or []
