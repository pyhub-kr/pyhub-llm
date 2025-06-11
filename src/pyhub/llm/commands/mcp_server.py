"""
MCP 서버 실행 명령
"""
import asyncio
import sys
from typing import Optional

import typer

app = typer.Typer(help="MCP 서버 관리")


@app.command("run")
def run(
    server_name: str = typer.Argument(
        ...,
        help="실행할 서버 이름 (예: calculator)",
    ),
):
    """
    내장 MCP 서버를 실행합니다.
    
    사용 가능한 서버:
    - calculator: 기본 계산 기능을 제공하는 서버
    """
    if server_name == "calculator":
        from pyhub.llm.mcp.servers.calculator import main
        
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            # Ctrl+C로 종료 시 깔끔하게 종료
            sys.exit(0)
    else:
        typer.echo(f"오류: 알 수 없는 서버 '{server_name}'", err=True)
        typer.echo("\n사용 가능한 서버:", err=True)
        typer.echo("  - calculator: 기본 계산 기능을 제공하는 서버", err=True)
        raise typer.Exit(1)


@app.command("list")
def list_servers():
    """사용 가능한 MCP 서버 목록을 표시합니다."""
    typer.echo("사용 가능한 MCP 서버:")
    typer.echo("  - calculator: 기본 계산 기능을 제공하는 서버")