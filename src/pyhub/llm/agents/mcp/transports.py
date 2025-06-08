"""MCP transport implementations with extended support for various protocols."""

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Transport(ABC):
    """Base transport interface for MCP connections."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize transport with configuration.

        Args:
            config: Transport configuration dictionary
        """
        self.config = config
        self.description = config.get("description", "")
        self.timeout = config.get("timeout", 30)

    @abstractmethod
    @asynccontextmanager
    async def connect(self):
        """Establish connection and return read/write streams."""
        pass

    def __str__(self) -> str:
        """String representation of the transport."""
        transport_type = self.config.get("transport", "unknown")
        return f"{transport_type.title()}Transport({self.description})"


class StdioTransport(Transport):
    """Standard I/O transport for MCP servers."""

    @asynccontextmanager
    async def connect(self):
        """Connect to MCP server via stdio."""
        try:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError("MCP stdio support requires 'mcp' package. Install it with: pip install mcp")

        # UTF-8 환경변수 설정 (Windows 호환)
        env = self.config.get("env", {}).copy()
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})

        server_params = StdioServerParameters(command=self.config["command"], args=self.config.get("args", []), env=env)

        logger.debug(f"Connecting to STDIO server: {self.config['command']} {' '.join(self.config.get('args', []))}")

        try:
            async with stdio_client(server_params) as (read, write):
                logger.info(f"✅ Connected to STDIO server: {self.config['command']}")
                yield read, write
        except Exception as e:
            logger.error(f"❌ Failed to connect to STDIO server: {e}")
            raise


class StreamableHTTPTransport(Transport):
    """Streamable HTTP transport for MCP servers."""

    @asynccontextmanager
    async def connect(self):
        """Connect to MCP server via streamable HTTP."""
        try:
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            raise ImportError("MCP streamable HTTP support requires 'mcp' package. Install it with: pip install mcp")

        url = self.config["url"]
        headers = self.config.get("headers", {})

        logger.debug(f"Connecting to HTTP server: {url}")

        try:
            # streamablehttp_client returns (read, write, _)
            async with streamablehttp_client(url, headers) as (read, write, _):
                logger.info(f"✅ Connected to HTTP server: {url}")
                yield read, write
        except Exception as e:
            logger.error(f"❌ Failed to connect to HTTP server: {e}")
            raise


class SSETransport(Transport):
    """Server-Sent Events transport for MCP servers."""

    @asynccontextmanager
    async def connect(self):
        """Connect to MCP server via SSE."""
        try:
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError("MCP SSE support requires 'mcp[sse]' package. Install it with: pip install 'mcp[sse]'")

        url = self.config["url"]
        headers = self.config.get("headers", {})

        logger.debug(f"Connecting to SSE server: {url}")

        try:
            async with sse_client(url, headers=headers) as (read, write):
                logger.info(f"✅ Connected to SSE server: {url}")
                yield read, write
        except Exception as e:
            logger.error(f"❌ Failed to connect to SSE server: {e}")
            raise


class WebSocketTransport(Transport):
    """WebSocket transport for MCP servers."""

    @asynccontextmanager
    async def connect(self):
        """Connect to MCP server via WebSocket."""
        try:
            from mcp.client.websocket import websocket_client
        except ImportError:
            raise ImportError(
                "MCP WebSocket support requires 'mcp[websocket]' package. Install it with: pip install 'mcp[websocket]'"
            )

        url = self.config["url"]
        headers = self.config.get("headers", {})

        logger.debug(f"Connecting to WebSocket server: {url}")

        try:
            async with websocket_client(url, headers=headers) as (read, write):
                logger.info(f"✅ Connected to WebSocket server: {url}")
                yield read, write
        except Exception as e:
            logger.error(f"❌ Failed to connect to WebSocket server: {e}")
            raise


def create_transport(config: Dict[str, Any]) -> Transport:
    """Create appropriate transport based on configuration.

    Args:
        config: Server configuration dict

    Returns:
        Transport instance

    Raises:
        ValueError: If transport type is unsupported or required fields are missing
    """
    # transport 타입 추론
    transport_type = config.get("transport")
    if not transport_type:
        transport_type = infer_transport_type(config)

    # 필수 필드 검증
    _validate_transport_config(transport_type, config)

    # Transport 매핑
    transport_map = {
        "stdio": StdioTransport,
        "sse": SSETransport,
        "streamable_http": StreamableHTTPTransport,
        "websocket": WebSocketTransport,
    }

    if transport_type not in transport_map:
        supported_types = ", ".join(transport_map.keys())
        raise ValueError(f"Unsupported transport type: {transport_type}. Supported types: {supported_types}")

    # Transport 인스턴스 생성
    transport_class = transport_map[transport_type]
    return transport_class(config)


def _validate_transport_config(transport_type: str, config: Dict[str, Any]):
    """Validate transport configuration based on type.

    Args:
        transport_type: Type of transport
        config: Configuration dictionary

    Raises:
        ValueError: If required fields are missing
    """
    if transport_type == "stdio":
        if "command" not in config:
            raise ValueError("stdio transport requires 'command' field")
    elif transport_type in ["sse", "streamable_http", "websocket"]:
        if "url" not in config:
            raise ValueError(f"{transport_type} transport requires 'url' field")
    else:
        raise ValueError(f"Unknown transport type for validation: {transport_type}")


def infer_transport_type(config: Dict[str, Any]) -> str:
    """Infer transport type from configuration.

    Args:
        config: Server configuration dict

    Returns:
        Transport type string

    Raises:
        ValueError: If transport type cannot be inferred
    """
    # 명시적으로 지정된 경우
    if "transport" in config:
        return config["transport"]

    # URL 기반 추론
    if "url" in config:
        url = config["url"].lower()
        if url.startswith("ws://") or url.startswith("wss://"):
            return "websocket"
        elif "/sse" in url or "server-sent-events" in url:
            return "sse"
        else:
            return "streamable_http"

    # command가 있으면 stdio
    if "command" in config:
        return "stdio"

    raise ValueError(
        "Cannot infer transport type. " "Provide either 'transport', 'url' (for network), or 'command' (for stdio)"
    )
