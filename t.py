import asyncio
from pyhub.llm import LLM
from pyhub.llm.mcp import McpStdioConfig


async def main():
    # MCP가 자동으로 초기화되는 LLM 생성
    llm = await LLM.create_async(
        "gpt-4o-mini",
        mcp_servers=McpStdioConfig(
            name="calculator",
            cmd="pyhub-llm mcp-server run calculator"
        )
    )

    # MCP 도구가 자동으로 사용됨
    response = await llm.ask_async("25와 17을 더하면?")
    print(response.text)

    # 사용 후 MCP 연결 종료
    await llm.close_mcp()


asyncio.run(main())

