"""
간단한 계산기 MCP 서버

MCP 라이브러리를 활용한 표준 MCP 서버 구현입니다.
"""
import asyncio
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions


# Create a server instance
server = Server("calculator-server")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """사용 가능한 도구 목록을 반환합니다."""
    return [
        types.Tool(
            name="add",
            description="두 숫자를 더합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "첫 번째 숫자"},
                    "b": {"type": "number", "description": "두 번째 숫자"}
                },
                "required": ["a", "b"]
            }
        ),
        types.Tool(
            name="subtract",
            description="두 숫자를 뺍니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "첫 번째 숫자"},
                    "b": {"type": "number", "description": "두 번째 숫자"}
                },
                "required": ["a", "b"]
            }
        ),
        types.Tool(
            name="multiply",
            description="두 숫자를 곱합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "첫 번째 숫자"},
                    "b": {"type": "number", "description": "두 번째 숫자"}
                },
                "required": ["a", "b"]
            }
        ),
        types.Tool(
            name="divide",
            description="두 숫자를 나눕니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "나누어지는 수"},
                    "b": {"type": "number", "description": "나누는 수"}
                },
                "required": ["a", "b"]
            }
        ),
        types.Tool(
            name="power",
            description="거듭제곱을 계산합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "base": {"type": "number", "description": "밑"},
                    "exponent": {"type": "number", "description": "지수"}
                },
                "required": ["base", "exponent"]
            }
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, any] | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """도구를 실행하고 결과를 반환합니다."""
    if not arguments:
        raise ValueError("Arguments are required")
    
    try:
        if name == "add":
            result = arguments["a"] + arguments["b"]
        elif name == "subtract":
            result = arguments["a"] - arguments["b"]
        elif name == "multiply":
            result = arguments["a"] * arguments["b"]
        elif name == "divide":
            if arguments["b"] == 0:
                raise ValueError("0으로 나눌 수 없습니다")
            result = arguments["a"] / arguments["b"]
        elif name == "power":
            result = arguments["base"] ** arguments["exponent"]
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        return [
            types.TextContent(
                type="text",
                text=str(result)
            )
        ]
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"오류: {str(e)}"
            )
        ]


async def run():
    """서버를 실행합니다."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="calculator",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


async def main():
    """메인 함수"""
    await run()


if __name__ == "__main__":
    asyncio.run(main())