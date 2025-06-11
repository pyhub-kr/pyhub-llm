#!/usr/bin/env python3
"""
Test all MCP transport types with pyhub-llm

This script demonstrates how to connect to MCP servers using different
transport methods and test their functionality.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add pyhub-llm to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from pyhub.llm import LLM
from pyhub.llm.mcp import MCPClient, load_mcp_tools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_stdio_transport():
    """Test STDIO transport with calculator server."""
    logger.info("=== Testing STDIO Transport ===")
    
    try:
        # STDIO server configuration
        config = {
            "transport": "stdio",
            "command": "python3",
            "args": [str(Path(__file__).parent.parent / "stdio" / "calculator_server.py")],
            "description": "STDIO Calculator Server"
        }
        
        # Create MCP client and test connection
        async with MCPClient(config).connect() as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"STDIO - Available tools: {[tool['name'] for tool in tools]}")
            
            # Test a tool
            result = await client.execute_tool("add", {"a": 10, "b": 5})
            logger.info(f"STDIO - add(10, 5) = {result}")
            
            # Test with LLM
            llm = LLMFactory.create("mock")  # Use mock LLM for testing
            
            # Load tools for LLM
            mcp_tools = await load_mcp_tools(client)
            
            response = await llm.ask_async(
                "Calculate 25 + 15",
                tools=mcp_tools
            )
            logger.info(f"STDIO - LLM response: {response.text}")
            
        logger.info("✅ STDIO transport test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ STDIO transport test failed: {e}")


async def test_sse_transport():
    """Test SSE transport with calculator server."""
    logger.info("=== Testing SSE Transport ===")
    
    try:
        # SSE server configuration
        config = {
            "transport": "sse",
            "url": "http://localhost:8001/mcp/sse",
            "description": "SSE Calculator Server"
        }
        
        # Create MCP client and test connection
        async with MCPClient(config).connect() as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"SSE - Available tools: {[tool['name'] for tool in tools]}")
            
            # Test a tool
            result = await client.execute_tool("multiply", {"a": 7, "b": 8})
            logger.info(f"SSE - multiply(7, 8) = {result}")
            
            # Test with LLM
            llm = LLMFactory.create("mock")
            mcp_tools = await load_mcp_tools(client)
            
            response = await llm.ask_async(
                "What's 12 times 3?",
                tools=mcp_tools
            )
            logger.info(f"SSE - LLM response: {response.text}")
            
        logger.info("✅ SSE transport test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ SSE transport test failed: {e}")


async def test_websocket_transport():
    """Test WebSocket transport with calculator server."""
    logger.info("=== Testing WebSocket Transport ===")
    
    try:
        # WebSocket server configuration
        config = {
            "transport": "websocket",
            "url": "ws://localhost:8002/mcp/ws",
            "description": "WebSocket Calculator Server"
        }
        
        # Create MCP client and test connection
        async with MCPClient(config).connect() as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"WebSocket - Available tools: {[tool['name'] for tool in tools]}")
            
            # Test a tool
            result = await client.execute_tool("divide", {"a": 20, "b": 4})
            logger.info(f"WebSocket - divide(20, 4) = {result}")
            
            # Test with LLM
            llm = LLMFactory.create("mock")
            mcp_tools = await load_mcp_tools(client)
            
            response = await llm.ask_async(
                "Calculate 100 divided by 5",
                tools=mcp_tools
            )
            logger.info(f"WebSocket - LLM response: {response.text}")
            
        logger.info("✅ WebSocket transport test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ WebSocket transport test failed: {e}")


async def test_http_transport():
    """Test HTTP transport with calculator server."""
    logger.info("=== Testing HTTP Transport ===")
    
    try:
        # HTTP server configuration
        config = {
            "transport": "streamable_http",
            "url": "http://localhost:8003/mcp",
            "description": "HTTP Calculator Server"
        }
        
        # Create MCP client and test connection
        async with MCPClient(config).connect() as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"HTTP - Available tools: {[tool['name'] for tool in tools]}")
            
            # Test a tool
            result = await client.execute_tool("get_time", {})
            logger.info(f"HTTP - get_time() = {result}")
            
            # Test with LLM
            llm = LLMFactory.create("mock")
            mcp_tools = await load_mcp_tools(client)
            
            response = await llm.ask_async(
                "What time is it?",
                tools=mcp_tools
            )
            logger.info(f"HTTP - LLM response: {response.text}")
            
        logger.info("✅ HTTP transport test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ HTTP transport test failed: {e}")


async def test_all_transports():
    """Test all transport types."""
    logger.info("🚀 Starting MCP Transport Tests")
    
    # Test STDIO (always available)
    await test_stdio_transport()
    
    # Test network transports (may require servers to be running)
    transports_to_test = [
        ("SSE", test_sse_transport),
        ("WebSocket", test_websocket_transport), 
        ("HTTP", test_http_transport)
    ]
    
    for transport_name, test_func in transports_to_test:
        try:
            await test_func()
        except Exception as e:
            logger.warning(f"⚠️ {transport_name} transport test skipped (server may not be running): {e}")
    
    logger.info("🎉 All MCP transport tests completed!")


async def demo_usage():
    """Demonstrate practical usage examples."""
    logger.info("=== MCP Usage Demo ===")
    
    # Example 1: Auto-detect transport from URL
    configs = [
        {"url": "ws://localhost:8002/mcp/ws"},  # Auto-detects WebSocket
        {"url": "http://localhost:8001/mcp/sse"},  # Auto-detects SSE
        {"command": "python3", "args": ["server.py"]},  # Auto-detects STDIO
    ]
    
    for config in configs:
        try:
            client = MCPClient(config)
            logger.info(f"Created client for config: {config}")
        except Exception as e:
            logger.error(f"Failed to create client: {e}")
    
    # Example 2: Filter specific tools
    stdio_config = {
        "command": "python3",
        "args": [str(Path(__file__).parent.parent / "stdio" / "calculator_server.py")]
    }
    
    try:
        # Load only specific tools
        tools = await load_mcp_tools(stdio_config, filter_tools=["add", "multiply"])
        logger.info(f"Filtered tools: {[tool.name for tool in tools]}")
    except Exception as e:
        logger.error(f"Tool filtering demo failed: {e}")


if __name__ == "__main__":
    # Run all tests
    asyncio.run(test_all_transports())
    
    # Run usage demo
    asyncio.run(demo_usage())