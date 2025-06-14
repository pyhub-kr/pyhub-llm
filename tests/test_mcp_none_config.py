"""Test for handling None configuration in MCP multi-client"""

import pytest

from pyhub.llm.mcp.multi_client import MultiServerMCPClient
from pyhub.llm.mcp.transports import StdioTransport


def test_none_config_in_dict_raises_error():
    """Test that None config values in dictionary raise proper error"""
    servers = {
        "valid_server": {"transport": "stdio", "command": "echo", "args": ["hello"]},
        "invalid_server": None,  # This should cause an error
    }

    # Should raise ValueError during initialization
    with pytest.raises(ValueError) as exc_info:
        MultiServerMCPClient(servers)

    assert "Server configuration for 'invalid_server' cannot be None" in str(exc_info.value)


def test_valid_configs_work():
    """Test that valid configurations work properly"""
    servers = {
        "server1": {"transport": "stdio", "command": "echo", "args": ["hello"]},
        "server2": {"transport": "streamable_http", "url": "http://localhost:8080"},
    }

    # Should not raise any error
    client = MultiServerMCPClient(servers)
    assert len(client.servers) == 2
    assert "server1" in client.servers
    assert "server2" in client.servers


@pytest.mark.asyncio
async def test_none_config_in_connect():
    """Test that if somehow None config gets to _connect_server, it's handled properly"""
    client = MultiServerMCPClient({})

    # Manually try to connect with None config
    await client._connect_server("test_server", None)

    # The error should be caught and stored in connection_errors
    assert "test_server" in client._connection_errors
    assert "Server configuration for 'test_server' is None" in client._connection_errors["test_server"]


def test_stdio_transport_with_none_env():
    """Test that StdioTransport handles None env correctly"""
    config = {"transport": "stdio", "command": "echo", "args": ["hello"], "env": None}  # This should not cause an error

    # Should not raise any error
    transport = StdioTransport(config)
    assert transport.config["env"] is None

    # The connect method should handle None env gracefully
    # (We can't test the full connect without actually running a server)
