"""MCP 연결 정책 사용 예제

MCP 연결 실패 시 동작을 제어하는 다양한 정책을 보여줍니다.

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import asyncio
import logging

from pyhub.llm import LLM
from pyhub.llm.mcp import McpStdioConfig, MCPConnectionPolicy, MCPConnectionError

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_optional_policy():
    """OPTIONAL 정책 (기본값) - MCP 실패해도 계속 진행"""
    print("\n=== OPTIONAL 정책 (기본값) ===")
    print("MCP 연결이 실패해도 LLM은 정상 작동합니다.")
    
    # 잘못된 MCP 서버 설정
    mcp_config = McpStdioConfig(
        name="invalid_server",
        cmd="nonexistent-command",
        description="존재하지 않는 서버"
    )
    
    # 정책을 명시하지 않으면 OPTIONAL이 기본값
    llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mcp_config)
    
    print(f"MCP 연결 상태: {llm._mcp_connected}")
    print(f"사용 가능한 도구 수: {len(llm._mcp_tools)}")
    
    # MCP 없이도 일반 대화는 가능
    response = await llm.ask_async("파이썬의 장점을 3가지 알려주세요.")
    print(f"응답: {response.text[:100]}...")
    
    await llm.close_mcp()


async def example_required_policy():
    """REQUIRED 정책 - MCP 연결 필수"""
    print("\n=== REQUIRED 정책 ===")
    print("MCP 연결이 실패하면 예외가 발생합니다.")
    
    # 잘못된 MCP 서버 설정
    mcp_config = McpStdioConfig(
        name="critical_server",
        cmd="nonexistent-command",
        description="중요한 서버 (반드시 필요)"
    )
    
    try:
        # REQUIRED 정책 설정
        _ = await LLM.create_async(
            "gpt-4o-mini",
            mcp_servers=mcp_config,
            mcp_policy=MCPConnectionPolicy.REQUIRED
        )
        print("❌ 이 메시지는 표시되지 않아야 합니다!")
        
    except MCPConnectionError as e:
        print(f"✅ 예상된 오류 발생: {e}")
        print(f"   실패한 서버: {e.failed_servers}")


async def example_warn_policy():
    """WARN 정책 - 경고만 표시하고 계속"""
    print("\n=== WARN 정책 ===")
    print("MCP 연결이 실패하면 경고를 표시하지만 계속 진행합니다.")
    
    # 잘못된 MCP 서버 설정
    mcp_config = McpStdioConfig(
        name="optional_server",
        cmd="nonexistent-command",
        description="선택적 서버"
    )
    
    # WARN 정책 설정
    llm = await LLM.create_async(
        "gpt-4o-mini",
        mcp_servers=mcp_config,
        mcp_policy=MCPConnectionPolicy.WARN
    )
    
    print(f"MCP 연결 상태: {llm._mcp_connected}")
    print("(위에 WARNING 로그가 표시되었을 것입니다)")
    
    await llm.close_mcp()


async def example_mixed_servers():
    """여러 서버 중 일부만 실패하는 경우"""
    print("\n=== 여러 서버 혼합 사용 ===")
    print("일부 서버만 실패해도 정책에 따라 동작이 달라집니다.")
    
    # 혼합 서버 설정
    mcp_configs = [
        McpStdioConfig(
            name="working_server",
            cmd="echo",  # 대부분의 시스템에서 작동
            description="작동하는 서버"
        ),
        McpStdioConfig(
            name="broken_server",
            cmd="nonexistent-command",
            description="작동하지 않는 서버"
        )
    ]
    
    # WARN 정책으로 생성
    llm = await LLM.create_async(
        "gpt-4o-mini",
        mcp_servers=mcp_configs,
        mcp_policy=MCPConnectionPolicy.WARN
    )
    
    print(f"MCP 연결 상태: {llm._mcp_connected}")
    
    if llm._mcp_client:
        failed_servers = list(llm._mcp_client._connection_errors.keys())
        successful_servers = [
            name for name in llm._mcp_client._clients.keys()
        ]
        print(f"성공한 서버: {successful_servers}")
        print(f"실패한 서버: {failed_servers}")
    
    await llm.close_mcp()


async def example_server_specific_policy():
    """서버별 정책 설정"""
    print("\n=== 서버별 정책 설정 ===")
    print("각 서버마다 다른 정책을 설정할 수 있습니다.")
    
    # 서버별로 다른 정책 설정
    mcp_configs = [
        McpStdioConfig(
            name="critical_calculator",
            cmd="uvx --from pyhub-llm pyhub-llm-calculator-server",
            description="필수 계산기 서버",
            policy=MCPConnectionPolicy.REQUIRED  # 이 서버는 필수
        ),
        McpStdioConfig(
            name="optional_logger",
            cmd="nonexistent-logger",
            description="선택적 로거 서버",
            policy=MCPConnectionPolicy.OPTIONAL  # 이 서버는 선택적
        )
    ]
    
    # 전체 정책은 REQUIRED로 설정
    # 하나라도 REQUIRED 서버가 실패하면 예외 발생
    try:
        llm = await LLM.create_async(
            "gpt-4o-mini",
            mcp_servers=mcp_configs,
            mcp_policy=MCPConnectionPolicy.REQUIRED
        )
        print("일부 서버가 성공했습니다.")
        await llm.close_mcp()
        
    except MCPConnectionError as e:
        print(f"필수 서버 연결 실패: {e}")


async def main():
    """모든 예제 실행"""
    print("MCP 연결 정책 예제")
    print("=" * 50)
    
    try:
        # 각 정책 예제 실행
        await example_optional_policy()
        await example_required_policy()
        await example_warn_policy()
        await example_mixed_servers()
        await example_server_specific_policy()
        
        print("\n" + "=" * 50)
        print("모든 예제 실행 완료!")
        
    except Exception as e:
        logger.error(f"예제 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())