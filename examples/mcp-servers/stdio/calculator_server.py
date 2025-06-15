#!/usr/bin/env python3
"""
STDIO MCP Server Example - Calculator Tools

This server communicates via standard input/output and provides
basic calculator functionality as MCP tools.
"""

import asyncio
import logging
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server

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
        # simpleeval을 사용한 안전한 계산
        try:
            import simpleeval
            import math
            evaluator = simpleeval.SimpleEval()
            # 기본 수학 함수들 허용
            evaluator.functions.update({
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'sqrt': math.sqrt,
                'log': math.log, 'exp': math.exp, 'pow': pow
            })
            evaluator.names.update({'pi': math.pi, 'e': math.e})
            result = evaluator.eval(expression)
        except ImportError:
            # simpleeval이 없으면 기본 제한 방식 사용
            import re
            # 위험한 키워드 검사
            dangerous_patterns = [
                r'__\w+__', r'import', r'exec', r'eval', r'open', r'file',
                r'globals', r'locals', r'vars', r'dir'
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, expression, re.IGNORECASE):
                    return f"Error: 위험한 키워드 사용 금지: {pattern}"

            # 허용된 문자만 확인 (수학 함수명 포함)
            if not re.match(r'^[0-9+\-*/().,\s_a-zA-Z]+$', expression):
                return "Error: 허용되지 않은 문자 포함"

            import math
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max, 'pow': pow,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'sqrt': math.sqrt,
                'log': math.log, 'exp': math.exp, 'pi': math.pi, 'e': math.e
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)

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