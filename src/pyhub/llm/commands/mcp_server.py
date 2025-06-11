"""
MCP 서버 실행 명령
"""

import subprocess
import sys
from typing import Optional

import typer

app = typer.Typer(help="MCP 서버 관리")


@app.command("run")
def run(
    server_name: str = typer.Argument(
        ...,
        help="실행할 서버 이름 (예: calculator, greeting)",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="서버 포트 (greeting 서버에만 적용됨, 기본값: 8000)",
    ),
):
    """
    내장 MCP 서버를 실행합니다.

    사용 가능한 서버:
    - calculator: 기본 계산 기능을 제공하는 서버
    - greeting: 다국어 인사말을 생성하는 서버

    예시:
    - pyhub-llm mcp-server run calculator
    - pyhub-llm mcp-server run greeting
    - pyhub-llm mcp-server run greeting --port 8080
    """
    # 모든 로직을 python -m pyhub.llm.mcp.servers 명령으로 위임
    cmd = [sys.executable, "-m", "pyhub.llm.mcp.servers", server_name]

    # port 옵션이 있으면 추가
    if port is not None:
        cmd.extend(["--port", str(port)])

    try:
        # subprocess로 실행하여 모든 출력을 그대로 전달
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        # Ctrl+C로 종료 시 깔끔하게 종료
        sys.exit(0)
    except subprocess.CalledProcessError:
        # 에러는 이미 subprocess에서 출력되므로 추가 메시지 없이 종료
        raise typer.Exit(1)


@app.command("list")
def list_servers():
    """사용 가능한 MCP 서버 목록을 표시합니다."""
    typer.echo("사용 가능한 MCP 서버:")
    typer.echo("  - calculator: 기본 계산 기능을 제공하는 서버")
    typer.echo("  - greeting: 다국어 인사말을 생성하는 서버")
