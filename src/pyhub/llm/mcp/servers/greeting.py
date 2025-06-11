"""
간단한 인사말 MCP 서버

FastMCP를 사용하여 다국어 인사말을 생성하는 MCP 서버입니다.
"""

import argparse

from fastmcp import FastMCP

# Create a FastMCP server instance
mcp = FastMCP("greeting-server")


@mcp.tool()
async def greeting(name: str, lang: str = "en") -> str:
    """Generate a greeting message in the specified language.

    Args:
        name: Name of the person to greet
        lang: Language code (en, ko, es, fr, ja). Defaults to "en"

    Returns:
        Greeting message in the specified language
    """
    # 언어별 인사말 템플릿
    greetings = {
        "en": f"Hello, {name}! Nice to meet you.",
        "ko": f"안녕하세요, {name}님! 만나서 반갑습니다.",
        "es": f"¡Hola, {name}! Mucho gusto.",
        "fr": f"Bonjour, {name}! Enchanté.",
        "ja": f"こんにちは、{name}さん！はじめまして。",
    }

    # 지원하지 않는 언어는 영어로 기본 설정
    return greetings.get(lang, greetings["en"])


if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Greeting MCP Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    args = parser.parse_args()

    # FastMCP 서버를 streamable-http transport로 실행
    print(f"Starting greeting MCP server on port {args.port}...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=args.port)
