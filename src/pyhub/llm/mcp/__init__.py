"""MCP (Model Context Protocol) integration for pyhub agents."""

from .client import MCPClient
from .config_loader import load_mcp_config
from .config_loader_class import MCPConfigLoader
from .configs import McpConfig, create_mcp_config
from .loader import load_mcp_tools
from .multi_client import (
    MultiServerMCPClient,
    create_multi_server_client_from_config,
)
from .policies import MCPConnectionError, MCPConnectionPolicy
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
    "McpConfig",
    "create_mcp_config",
    # Config loader functions
    "load_mcp_config",
    # Config loader class
    "MCPConfigLoader",
    # Policies
    "MCPConnectionPolicy",
    "MCPConnectionError",
]
