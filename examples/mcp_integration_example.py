"""
MCP (Model Context Protocol) 통합 예제

이 예제는 LLM과 MCP 서버를 통합하여 사용하는 방법을 보여줍니다.

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import asyncio
import logging
from pyhub.llm import LLM
from pyhub.llm.mcp import McpStdioConfig

# 로깅 설정 (디버깅용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_1_basic_mcp():
    """예제 1: 기본적인 MCP 사용법"""
    print("\n=== 예제 1: 기본적인 MCP 사용법 ===")

    # MCP 서버 설정 (내장 계산기 서버)
    calc_config = McpStdioConfig(
        name="calculator",
        cmd="pyhub-llm mcp-server run calculator",
        description="수학 계산 도구"
    )

    # LLM 생성 시 MCP 설정 전달
    llm = LLM.create("gpt-4o-mini", mcp_servers=calc_config)

    # MCP 연결 초기화
    await llm.initialize_mcp()

    try:
        # MCP 도구를 사용하여 질문
        response = await llm.ask_async(
            "25와 17을 더한 다음, 그 결과를 3으로 나누면 얼마인가요? "
            "소수점 둘째 자리까지 알려주세요."
        )
        print(f"답변: {response.text}")
    finally:
        # MCP 연결 종료
        await llm.close_mcp()


async def example_2_create_async():
    """예제 2: create_async를 사용한 자동 초기화"""
    print("\n=== 예제 2: create_async를 사용한 자동 초기화 ===")

    # create_async를 사용하면 MCP가 자동으로 초기화됨
    llm = await LLM.create_async(
        "gpt-4o-mini",
        mcp_servers=McpStdioConfig(
            name="calculator",
            cmd="pyhub-llm mcp-server run calculator"
        )
    )

    try:
        response = await llm.ask_async(
            "2의 10제곱은 얼마인가요?"
        )
        print(f"답변: {response.text}")
    finally:
        await llm.close_mcp()


async def example_3_context_manager():
    """예제 3: 컨텍스트 매니저 사용"""
    print("\n=== 예제 3: 컨텍스트 매니저 사용 ===")

    # 컨텍스트 매니저를 사용하면 자동으로 연결/해제
    async with await LLM.create_async(
        "gpt-4o-mini",
        mcp_servers=McpStdioConfig(
            name="calculator",
            cmd="pyhub-llm mcp-server run calculator"
        )
    ) as llm:
        response = await llm.ask_async(
            "100에서 37을 빼고, 그 결과에 2를 곱하면?"
        )
        print(f"답변: {response.text}")
    # 여기서 자동으로 MCP 연결이 종료됨


async def example_4_multiple_servers():
    """예제 4: 여러 MCP 서버 사용"""
    print("\n=== 예제 4: 여러 MCP 서버 사용 ===")

    # 여러 MCP 서버 설정
    mcp_servers = [
        McpStdioConfig(
            name="calculator",
            cmd="pyhub-llm mcp-server run calculator",
            description="계산 도구"
        ),
        # 실제로 greeting 서버가 8888 포트에서 실행 중이어야 함
        # McpStreamableHttpConfig(
        #     name="greeting",
        #     url="http://localhost:8888/mcp",
        #     description="인사말 도구"
        # )
    ]

    async with await LLM.create_async("gpt-4o-mini", mcp_servers=mcp_servers) as llm:
        # 계산기 도구 사용
        calc_response = await llm.ask_async(
            "15 더하기 25는 얼마인가요?"
        )
        print(f"계산 결과: {calc_response.text}")

        # 인사말 도구 사용 (greeting 서버가 실행 중인 경우)
        # greeting_response = await llm.ask_async(
        #     "John에게 한국어로 인사해주세요."
        # )
        # print(f"인사말: {greeting_response.text}")


async def example_5_with_existing_tools():
    """예제 5: 기존 도구와 MCP 도구 함께 사용"""
    print("\n=== 예제 5: 기존 도구와 MCP 도구 함께 사용 ===")

    # 간단한 도구 정의
    def get_current_time():
        """현재 시간을 반환합니다."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # MCP와 일반 도구를 함께 사용
    async with await LLM.create_async(
        "gpt-4o-mini",
        tools=[get_current_time],  # 일반 도구
        mcp_servers=McpStdioConfig(  # MCP 도구
            name="calculator",
            cmd="pyhub-llm mcp-server run calculator"
        )
    ) as llm:
        response = await llm.ask_async(
            "현재 시간을 알려주고, 1시간은 몇 초인지 계산해주세요."
        )
        print(f"답변: {response.text}")


async def example_6_error_handling():
    """예제 6: 에러 처리"""
    print("\n=== 예제 6: 에러 처리 ===")

    # 잘못된 MCP 서버 설정
    invalid_config = McpStdioConfig(
        name="invalid",
        cmd="nonexistent-command",
        description="존재하지 않는 명령"
    )

    llm = LLM.create("gpt-4o-mini", mcp_servers=invalid_config)

    # MCP 연결 실패 시에도 LLM은 정상 작동
    await llm.initialize_mcp()  # 에러가 발생하지만 프로그램은 계속됨

    # MCP 도구 없이도 일반 질문은 가능
    response = await llm.ask_async("안녕하세요! 오늘 날씨가 좋네요.")
    print(f"답변: {response.text}")

    await llm.close_mcp()


async def main():
    """모든 예제 실행"""
    try:
        # 기본 사용법
        await example_1_basic_mcp()

        # create_async 사용
        await example_2_create_async()

        # 컨텍스트 매니저
        await example_3_context_manager()

        # 여러 서버
        await example_4_multiple_servers()

        # 기존 도구와 함께 사용
        await example_5_with_existing_tools()

        # 에러 처리
        await example_6_error_handling()

    except Exception as e:
        logger.error(f"예제 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    print("LLM MCP 통합 예제")
    print("=" * 50)
    asyncio.run(main())