"""
MCP 서버 CLI 엔트리포인트

사용법:
    python -m pyhub.llm.mcp.servers <server_name>
    python -m pyhub.llm.mcp.servers calculator
"""
import asyncio
import sys
from typing import Optional


def print_usage():
    """사용법을 출력합니다"""
    print("사용법: python -m pyhub.llm.mcp.servers <server_name>")
    print("\n사용 가능한 서버:")
    print("  calculator - 기본 계산 기능을 제공하는 서버")


async def run_server(server_name: str):
    """지정된 서버를 실행합니다"""
    if server_name == "calculator":
        from .calculator import main
        await main()
    else:
        print(f"오류: 알 수 없는 서버 '{server_name}'")
        print()
        print_usage()
        sys.exit(1)


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    server_name = sys.argv[1]
    
    if server_name in ["-h", "--help"]:
        print_usage()
        sys.exit(0)
    
    try:
        asyncio.run(run_server(server_name))
    except KeyboardInterrupt:
        # Ctrl+C로 종료 시 깔끔하게 종료
        sys.exit(0)


if __name__ == "__main__":
    main()