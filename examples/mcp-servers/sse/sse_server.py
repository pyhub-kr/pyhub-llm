#!/usr/bin/env python3
"""
SSE MCP Server Example - Calculator Tools

This server communicates via Server-Sent Events (SSE) and provides
basic calculator functionality as MCP tools.
"""

import json
import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from sse_starlette.sse import EventSourceResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="SSE MCP Calculator Server", version="1.0.0")

# Create MCP server instance
mcp_server = Server("calculator-sse-server")


# Tool definitions (same as STDIO server)
@mcp_server.tool()
async def add(a: float, b: float) -> str:
    """Add two numbers together."""
    result = a + b
    logger.info(f"add({a}, {b}) = {result}")
    return f"The sum of {a} and {b} is {result}"


@mcp_server.tool()
async def multiply(a: float, b: float) -> str:
    """Multiply two numbers."""
    result = a * b
    logger.info(f"multiply({a}, {b}) = {result}")
    return f"The product of {a} and {b} is {result}"


@mcp_server.tool()
async def subtract(a: float, b: float) -> str:
    """Subtract b from a."""
    result = a - b
    logger.info(f"subtract({a}, {b}) = {result}")
    return f"{a} minus {b} equals {result}"


@mcp_server.tool()
async def divide(a: float, b: float) -> str:
    """Divide a by b."""
    if b == 0:
        return "Error: Division by zero is not allowed"
    result = a / b
    logger.info(f"divide({a}, {b}) = {result}")
    return f"{a} divided by {b} equals {result}"


@mcp_server.tool()
async def get_time() -> str:
    """Get the current time."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"get_time() = {current_time}")
    return f"Current time is: {current_time}"


@mcp_server.tool()
async def echo(message: str) -> str:
    """Echo back the provided message."""
    logger.info(f"echo('{message}')")
    return f"Echo: {message}"


@mcp_server.tool()
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


@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "name": "SSE MCP Calculator Server",
        "version": "1.0.0",
        "transport": "sse",
        "endpoints": {
            "mcp": "/mcp/sse"
        }
    }


@app.get("/mcp/sse")
async def mcp_sse_endpoint(request: Request):
    """SSE endpoint for MCP communication."""
    logger.info("New SSE MCP connection established")

    async def event_generator():
        try:
            # Create SSE transport
            transport = SseServerTransport("/mcp/sse")

            # Handle MCP session
            async with transport.connect() as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options()
                )
        except Exception as e:
            logger.error(f"SSE MCP session error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    logger.info("Starting SSE MCP Calculator Server on http://localhost:8001")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )