#!/usr/bin/env python3
"""
STDIO MCP Server Example - Calculator Tools

This server communicates via standard input/output and provides
basic calculator functionality as MCP tools.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Create MCP server instance
server = Server("calculator-server")


# Tool definitions
@server.tool()
async def add(a: float, b: float) -> str:
    """Add two numbers together."""
    result = a + b
    logger.info(f"add({a}, {b}) = {result}")
    return f"The sum of {a} and {b} is {result}"


@server.tool()
async def multiply(a: float, b: float) -> str:
    """Multiply two numbers."""
    result = a * b
    logger.info(f"multiply({a}, {b}) = {result}")
    return f"The product of {a} and {b} is {result}"


@server.tool()
async def subtract(a: float, b: float) -> str:
    """Subtract b from a."""
    result = a - b
    logger.info(f"subtract({a}, {b}) = {result}")
    return f"{a} minus {b} equals {result}"


@server.tool()
async def divide(a: float, b: float) -> str:
    """Divide a by b."""
    if b == 0:
        return "Error: Division by zero is not allowed"
    result = a / b
    logger.info(f"divide({a}, {b}) = {result}")
    return f"{a} divided by {b} equals {result}"


@server.tool()
async def get_time() -> str:
    """Get the current time."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"get_time() = {current_time}")
    return f"Current time is: {current_time}"


@server.tool()
async def echo(message: str) -> str:
    """Echo back the provided message."""
    logger.info(f"echo('{message}')")
    return f"Echo: {message}"


@server.tool()
async def calculate_expression(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # Only allow safe mathematical operations
        allowed_chars = "0123456789+-*/()., "
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        # Evaluate the expression
        result = eval(expression)
        logger.info(f"calculate_expression('{expression}') = {result}")
        return f"Result: {expression} = {result}"
    except Exception as e:
        logger.error(f"Error evaluating expression '{expression}': {e}")
        return f"Error: Failed to evaluate expression - {str(e)}"


async def main():
    """Run the STDIO MCP server."""
    logger.info("Starting Calculator MCP Server (STDIO)")
    
    try:
        # Run the server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())