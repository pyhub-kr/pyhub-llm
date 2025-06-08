"""MCP 설정 관리 모듈"""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MCPConfig:
    """MCP 설정 관리 클래스"""

    @staticmethod
    def get_default_config_path() -> Path:
        """기본 MCP 설정 파일 경로 반환

        Returns:
            ~/.pyhub-mcptools/mcp.toml 경로
        """
        home = Path.home()
        return home / ".pyhub-mcptools" / "mcp.toml"

    @staticmethod
    def get_default_config_dir() -> Path:
        """기본 MCP 설정 디렉터리 경로 반환

        Returns:
            ~/.pyhub-mcptools 경로
        """
        return MCPConfig.get_default_config_path().parent

    @staticmethod
    def ensure_config_dir() -> Path:
        """설정 디렉터리 생성 및 경로 반환

        Returns:
            생성된 설정 디렉터리 경로
        """
        config_dir = MCPConfig.get_default_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"MCP config directory ensured: {config_dir}")
        return config_dir

    @staticmethod
    def create_default_config_if_not_exists() -> Path:
        """기본 설정 파일이 없으면 생성

        Returns:
            설정 파일 경로
        """
        config_path = MCPConfig.get_default_config_path()

        if not config_path.exists():
            # 디렉터리 생성
            MCPConfig.ensure_config_dir()

            # 기본 설정 내용
            default_content = """# MCP (Model Context Protocol) 서버 설정
# 각 서버는 다양한 transport 방식을 지원합니다:
# - stdio: 표준 입출력을 통한 로컬 프로세스 통신
# - sse: Server-Sent Events를 통한 HTTP 기반 통신  
# - streamable_http: 스트리밍 HTTP 통신
# - websocket: WebSocket 기반 실시간 통신

# 예시 서버 설정들 (사용하려면 주석 해제하고 적절한 경로/URL로 수정)

# [servers.filesystem]
# transport = "stdio"
# command = "python"
# args = ["/path/to/filesystem_server.py"]
# env = {"DEBUG" = "1"}
# description = "파일시스템 관리 도구"

# [servers.web_search]
# transport = "sse"
# url = "http://localhost:3000/mcp/sse"
# headers = {"Authorization" = "Bearer your-token"}
# description = "웹 검색 도구"

# [servers.calculator]
# transport = "streamable_http"
# url = "http://localhost:3000/mcp"
# timeout = 30
# description = "계산기 도구"

# [servers.weather]
# transport = "websocket"
# url = "ws://localhost:3000/mcp/ws"
# description = "날씨 정보 도구"

[default]
auto_load = true
timeout = 30
max_retries = 3
"""

            # UTF-8 인코딩으로 기본 설정 파일 생성
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(default_content)

            logger.info(f"Created default MCP config file: {config_path}")

        return config_path

    @staticmethod
    def read_config_file(file_path: Path) -> str:
        """설정 파일을 UTF-8로 안전하게 읽기 (Windows 호환)

        Args:
            file_path: 읽을 파일 경로

        Returns:
            파일 내용

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            UnicodeDecodeError: UTF-8 디코딩 실패 시
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError as e:
            logger.error(f"UTF-8 decoding failed for {file_path}: {e}")
            raise

    @staticmethod
    def write_config_file(file_path: Path, content: str):
        """설정 파일을 UTF-8로 안전하게 쓰기 (Windows 호환)

        Args:
            file_path: 쓸 파일 경로
            content: 파일 내용
        """
        # 디렉터리 생성
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # UTF-8 인코딩으로 저장
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.debug(f"Config file written: {file_path}")

    @staticmethod
    def validate_server_config(config: Dict[str, Any]) -> bool:
        """서버 설정의 유효성 검사

        Args:
            config: 서버 설정 딕셔너리

        Returns:
            설정이 유효한지 여부
        """
        transport = config.get("transport", "stdio")

        if transport == "stdio":
            return "command" in config
        elif transport in ["sse", "streamable_http", "websocket"]:
            return "url" in config
        else:
            logger.warning(f"Unknown transport type: {transport}")
            return False
