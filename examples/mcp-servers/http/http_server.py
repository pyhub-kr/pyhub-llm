#!/usr/bin/env python3
"""
Streamable HTTP MCP Server Example - Calculator Tools

This server communicates via streamable HTTP and provides
basic calculator functionality as MCP tools.
"""

import asyncio
import json
import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request, Response
from mcp.server import Server
from mcp.server.streamable_http import StreamableHTTPServerTransport

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="HTTP MCP Calculator Server", version="1.0.0")

# Create MCP server instance
mcp_server = Server("calculator-http-server")


# Tool definitions (same as other servers)
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
        "name": "HTTP MCP Calculator Server",
        "version": "1.0.0",
        "transport": "streamable_http",
        "endpoints": {
            "mcp": "/mcp"
        }
    }


@app.post("/mcp")
async def mcp_http_endpoint(request: Request):
    """HTTP endpoint for MCP communication."""
    logger.info("New HTTP MCP request received")

    try:
        # Get request body
        body = await request.body()

        # Create HTTP transport
        transport = StreamableHTTPServerTransport()

        # Process MCP request
        async def process_request():
            async with transport.connect() as (read_stream, write_stream):
                # Send request to MCP server
                await write_stream.send(body)

                # Get response from MCP server
                response = await read_stream.receive()
                return response

        response_data = await process_request()

        return Response(
            content=response_data,
            media_type="application/json",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

    except Exception as e:
        logger.error(f"HTTP MCP request error: {e}")
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json"
        )


@app.options("/mcp")
async def mcp_options():
    """CORS preflight handler for MCP endpoint."""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )


class SimpleHTTPTransport:
    """Simple HTTP transport implementation for MCP."""

    def __init__(self):
        self.request_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

    async def connect(self):
        """Context manager for connection."""
        return self, self

    async def __aenter__(self):
        return self, self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def send_request(self, data: bytes) -> bytes:
        """Send request and get response."""
        await self.request_queue.put(data)
        response = await self.response_queue.get()
        return response

    async def read(self) -> str:
        """Read message from queue."""
        data = await self.request_queue.get()
        return data.decode('utf-8')

    async def write(self, message: str):
        """Write message to queue."""
        await self.response_queue.put(message.encode('utf-8'))


# Global transport instance for simple endpoint
simple_transport = SimpleHTTPTransport()


@app.post("/mcp/simple")
async def simple_mcp_http_endpoint(request: Request):
    """Simplified HTTP endpoint for MCP communication."""
    logger.info("New simple HTTP MCP request received")

    try:
        # Get request body
        body = await request.body()

        # Use simple transport
        response_data = await simple_transport.send_request(body)

        return Response(
            content=response_data,
            media_type="application/json",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

    except Exception as e:
        logger.error(f"Simple HTTP MCP request error: {e}")
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json"
        )


# Background task to handle MCP server with simple transport
async def run_simple_mcp_server():
    """Run MCP server with simple transport in background."""
    try:
        async with simple_transport.connect() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Simple MCP server error: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background MCP server."""
    asyncio.create_task(run_simple_mcp_server())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    logger.info("Starting HTTP MCP Calculator Server on http://localhost:8003")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )