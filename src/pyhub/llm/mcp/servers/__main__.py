"""
MCP 서버 모듈 직접 실행 엔트리포인트

사용법:
    python -m pyhub.llm.mcp.servers <server_name>
    python -m pyhub.llm.mcp.servers calculator
    python -m pyhub.llm.mcp.servers greeting
    python -m pyhub.llm.mcp.servers greeting --port 8080
"""

import subprocess
import sys
from pathlib import Path


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python -m pyhub.llm.mcp.servers <server_name> [options]")
        print("\n사용 가능한 서버:")
        print("  calculator - 기본 계산 기능을 제공하는 서버")
        print("  greeting   - 다국어 인사말을 생성하는 서버")
        print("\n옵션:")
        print("  --port PORT - greeting 서버의 포트 지정 (기본값: 8000)")
        sys.exit(1)

    server_name = sys.argv[1]

    if server_name in ["-h", "--help"]:
        print("사용법: python -m pyhub.llm.mcp.servers <server_name> [options]")
        print("\n사용 가능한 서버:")
        print("  calculator - 기본 계산 기능을 제공하는 서버")
        print("  greeting   - 다국어 인사말을 생성하는 서버")
        print("\n옵션:")
        print("  --port PORT - greeting 서버의 포트 지정 (기본값: 8000)")
        sys.exit(0)

    if server_name == "calculator":
        # calculator는 직접 import해서 실행
        import asyncio

        from .calculator import main as calc_main

        try:
            asyncio.run(calc_main())
        except KeyboardInterrupt:
            sys.exit(0)
    elif server_name == "greeting":
        # greeting은 subprocess로 실행하며 추가 인자 전달
        current_dir = Path(__file__).parent
        greeting_path = current_dir / "greeting.py"

        # sys.argv[2:] 로 나머지 인자들을 전달
        cmd = [sys.executable, str(greeting_path)] + sys.argv[2:]

        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            sys.exit(0)
        except subprocess.CalledProcessError:
            sys.exit(1)
    else:
        print(f"오류: 알 수 없는 서버 '{server_name}'")
        print("\n사용 가능한 서버:")
        print("  calculator - 기본 계산 기능을 제공하는 서버")
        print("  greeting   - 다국어 인사말을 생성하는 서버")
        sys.exit(1)


if __name__ == "__main__":
    main()
