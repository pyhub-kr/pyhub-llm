"""MCP (Model Context Protocol) integration for pyhub agents."""

from .client import MCPClient
from .configs import (
    McpServerConfig,
    McpSseConfig,
    McpStdioConfig,
    McpStreamableHttpConfig,
    McpWebSocketConfig,
)
from .config_loader import load_mcp_config, normalize_mcp_config, validate_mcp_config
from .loader import load_mcp_tools
from .multi_client import (
    MultiServerMCPClient,
    create_multi_server_client_from_config,
)
from .transports import (
    SSETransport,
    StdioTransport,
    StreamableHTTPTransport,
    WebSocketTransport,
)
from .wrapper import MCPTool

__all__ = [
    "MCPClient",
    "load_mcp_tools",
    "MCPTool",
    "MultiServerMCPClient",
    "create_multi_server_client_from_config",
    "StdioTransport",
    "StreamableHTTPTransport",
    "WebSocketTransport",
    "SSETransport",
    # Config classes
    "McpServerConfig",
    "McpStdioConfig",
    "McpStreamableHttpConfig",
    "McpWebSocketConfig",
    "McpSseConfig",
    # Config loader functions
    "load_mcp_config",
    "validate_mcp_config",
    "normalize_mcp_config",
]
