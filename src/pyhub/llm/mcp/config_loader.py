"""MCP 설정 로더 모듈 - 다양한 방식의 MCP 설정 로드 (UTF-8 인코딩 보장)"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# toml import with fallback
try:
    import toml
except ImportError:
    try:
        import tomllib as toml
    except ImportError:
        logger.warning("TOML library not available. Install with: pip install toml")
        toml = None


class MCPConfigLoader:
    """다양한 방식의 MCP 설정 로드 (UTF-8 인코딩 보장)"""

    @staticmethod
    def load_from_default_file() -> Dict[str, Any]:
        """기본 MCP 설정 파일에서 로드

        기본 경로: ~/.pyhub-mcptools/mcp.toml

        Returns:
            서버 설정 딕셔너리
        """
        from .config import MCPConfig

        default_path = MCPConfig.get_default_config_path()

        if default_path.exists():
            try:
                return MCPConfigLoader.load_from_file(str(default_path))
            except Exception as e:
                logger.error(f"Failed to load default MCP config: {e}")
                return {}
        else:
            logger.debug(f"Default MCP config file not found: {default_path}")
            return {}

    @staticmethod
    def load_from_file(config_path: str) -> Dict[str, Any]:
        """TOML 파일에서 MCP 설정 로드 (UTF-8 보장)

        Args:
            config_path: 설정 파일 경로

        Returns:
            서버 설정 딕셔너리

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            ValueError: TOML 파싱 실패 시
            ImportError: toml 라이브러리가 없는 경우
        """
        if toml is None:
            raise ImportError("TOML support requires 'toml' package. Install with: pip install toml")

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            # UTF-8 인코딩 명시적 지정
            with open(path, "r", encoding="utf-8") as f:
                if hasattr(toml, "load"):
                    # toml 라이브러리
                    config = toml.load(f)
                else:
                    # tomllib (Python 3.11+)
                    content = f.read()
                    config = toml.loads(content)

            servers = config.get("servers", {})
            logger.info(f"Loaded MCP config from {config_path}: {len(servers)} servers")

            return servers

        except Exception as e:
            logger.error(f"Failed to load MCP config from {config_path}: {e}")
            raise ValueError(f"Invalid TOML config file: {e}")

    @staticmethod
    def load_from_json(json_str: str) -> Dict[str, Any]:
        """JSON 문자열에서 MCP 설정 로드

        Args:
            json_str: JSON 형태의 설정 문자열

        Returns:
            서버 설정 딕셔너리

        Raises:
            ValueError: JSON 파싱 실패 시
        """
        try:
            config = json.loads(json_str)
            logger.info(f"Loaded MCP config from JSON: {len(config)} servers")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON config: {e}")
            raise ValueError(f"Invalid JSON config: {e}")

    @staticmethod
    def load_from_cli_args(
        mcp_stdio: Optional[List[str]] = None,
        mcp_sse: Optional[List[str]] = None,
        mcp_http: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """CLI 인자에서 MCP 설정 생성

        Args:
            mcp_stdio: STDIO 서버 명령어 리스트
            mcp_sse: SSE 서버 URL 리스트
            mcp_http: HTTP 서버 URL 리스트

        Returns:
            서버 설정 딕셔너리
        """
        config = {}

        # STDIO 서버들
        if mcp_stdio:
            for i, cmd in enumerate(mcp_stdio):
                parts = cmd.split()
                if not parts:
                    logger.warning(f"Empty STDIO command at index {i}")
                    continue

                server_name = f"stdio_server_{i}"
                config[server_name] = {
                    "transport": "stdio",
                    "command": parts[0],
                    "args": parts[1:] if len(parts) > 1 else [],
                    "description": f"CLI STDIO server: {cmd}",
                }

        # SSE 서버들
        if mcp_sse:
            for i, url in enumerate(mcp_sse):
                if not url.strip():
                    logger.warning(f"Empty SSE URL at index {i}")
                    continue

                server_name = f"sse_server_{i}"
                config[server_name] = {"transport": "sse", "url": url.strip(), "description": f"CLI SSE server: {url}"}

        # HTTP 서버들
        if mcp_http:
            for i, url in enumerate(mcp_http):
                if not url.strip():
                    logger.warning(f"Empty HTTP URL at index {i}")
                    continue

                server_name = f"http_server_{i}"
                config[server_name] = {
                    "transport": "streamable_http",
                    "url": url.strip(),
                    "description": f"CLI HTTP server: {url}",
                }

        if config:
            logger.info(f"Generated MCP config from CLI args: {len(config)} servers")

        return config

    @staticmethod
    def load_from_environment() -> Dict[str, Any]:
        """환경변수에서 MCP 설정 로드

        지원되는 환경변수 패턴:
        1. PYHUB_MCP_SERVERS_JSON: JSON 형태 직접 설정
        2. PYHUB_MCP_<SERVER>_<FIELD>: 개별 서버 설정
        3. PYHUB_MCP_SERVER_LIST: 활성화할 서버 목록

        Returns:
            서버 설정 딕셔너리
        """
        config = {}

        # 1. JSON 형태 직접 설정
        servers_json = os.getenv("PYHUB_MCP_SERVERS_JSON")
        if servers_json:
            try:
                json_config = json.loads(servers_json)
                config.update(json_config)
                logger.info(f"Loaded MCP config from PYHUB_MCP_SERVERS_JSON: {len(json_config)} servers")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in PYHUB_MCP_SERVERS_JSON: {e}")

        # 2. 개별 서버 설정 (PYHUB_MCP_<SERVER>_<FIELD> 패턴)
        server_configs = {}
        for env_name, env_value in os.environ.items():
            if env_name.startswith("PYHUB_MCP_") and "_" in env_name[10:]:
                parts = env_name[10:].split("_", 1)  # PYHUB_MCP_ 제거 후 첫 번째 _로 분할
                if len(parts) == 2:
                    server_name, field_name = parts
                    server_name = server_name.lower()
                    field_name = field_name.lower()

                    if server_name not in server_configs:
                        server_configs[server_name] = {}

                    # 특수 필드 처리
                    if field_name == "args":
                        # 콤마로 구분된 인자 리스트
                        server_configs[server_name]["args"] = [
                            arg.strip() for arg in env_value.split(",") if arg.strip()
                        ]
                    elif field_name == "env":
                        # 콤마로 구분된 환경변수 (KEY=VALUE,KEY2=VALUE2)
                        env_dict = {}
                        for pair in env_value.split(","):
                            if "=" in pair:
                                key, value = pair.split("=", 1)
                                env_dict[key.strip()] = value.strip()
                        server_configs[server_name]["env"] = env_dict
                    elif field_name == "headers":
                        # 콤마로 구분된 헤더 (KEY=VALUE,KEY2=VALUE2)
                        headers_dict = {}
                        for pair in env_value.split(","):
                            if "=" in pair:
                                key, value = pair.split("=", 1)
                                headers_dict[key.strip()] = value.strip()
                        server_configs[server_name]["headers"] = headers_dict
                    else:
                        server_configs[server_name][field_name] = env_value

        # 개별 서버 설정을 메인 config에 추가
        if server_configs:
            for server_name, server_config in server_configs.items():
                # description 자동 생성
                if "description" not in server_config:
                    transport = server_config.get("transport", "unknown")
                    server_config["description"] = f"Environment server: {server_name} ({transport})"

                config[server_name] = server_config

            logger.info(f"Loaded MCP config from environment variables: {len(server_configs)} servers")

        # 3. 서버 목록 필터링 (선택적)
        server_list = os.getenv("PYHUB_MCP_SERVER_LIST")
        if server_list and config:
            allowed_servers = [name.strip() for name in server_list.split(",") if name.strip()]
            filtered_config = {name: cfg for name, cfg in config.items() if name in allowed_servers}
            if len(filtered_config) != len(config):
                logger.info(
                    f"Filtered MCP servers by PYHUB_MCP_SERVER_LIST: {len(filtered_config)}/{len(config)} servers"
                )
            config = filtered_config

        return config

    @staticmethod
    def get_environment_config_path() -> Optional[str]:
        """환경변수에서 MCP 설정 파일 경로 가져오기

        Returns:
            설정 파일 경로 또는 None
        """
        return os.getenv("PYHUB_MCP_CONFIG")

    @staticmethod
    def is_default_config_disabled() -> bool:
        """환경변수에서 기본 설정 비활성화 여부 확인

        Returns:
            기본 설정 비활성화 여부
        """
        disable_str = os.getenv("PYHUB_MCP_DISABLE_DEFAULT", "").lower()
        return disable_str in ("true", "1", "yes", "on")

    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """여러 설정을 병합 (나중 설정이 우선순위)

        Args:
            *configs: 병합할 설정 딕셔너리들

        Returns:
            병합된 설정 딕셔너리
        """
        merged = {}

        for config in configs:
            if config:
                # 서버 이름 중복 시 나중 설정이 우선
                for server_name, server_config in config.items():
                    if server_name in merged:
                        logger.warning(f"Overriding MCP server config: {server_name}")
                    merged[server_name] = server_config

        if merged:
            logger.info(f"Merged MCP configs: {len(merged)} total servers")

        return merged

    @staticmethod
    def save_config_to_file(config: Dict[str, Any], file_path: Optional[str] = None):
        """설정을 TOML 파일로 저장 (UTF-8 보장)

        Args:
            config: 저장할 서버 설정 딕셔너리
            file_path: 저장할 파일 경로 (None이면 기본 경로 사용)

        Raises:
            ImportError: toml 라이브러리가 없는 경우
        """
        if toml is None:
            raise ImportError("TOML support requires 'toml' package. Install with: pip install toml")

        from .config import MCPConfig

        if file_path is None:
            save_path = MCPConfig.get_default_config_path()
        else:
            save_path = Path(file_path)

        # 디렉터리 생성
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # TOML 형식으로 변환 (servers 섹션으로 감싸기)
        toml_config = {"servers": config, "default": {"auto_load": True, "timeout": 30, "max_retries": 3}}

        # UTF-8 인코딩으로 저장
        with open(save_path, "w", encoding="utf-8") as f:
            if hasattr(toml, "dump"):
                toml.dump(toml_config, f)
            else:
                # tomllib는 dump를 지원하지 않음
                raise ImportError("Saving TOML requires 'toml' package (not tomllib)")

        logger.info(f"MCP config saved to: {save_path}")

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """설정 유효성 검사 및 정리

        Args:
            config: 검사할 설정 딕셔너리

        Returns:
            유효한 설정만 포함된 딕셔너리
        """
        from .config import MCPConfig

        valid_config = {}

        for server_name, server_config in config.items():
            if not isinstance(server_config, dict):
                logger.warning(f"Invalid server config for '{server_name}': not a dictionary")
                continue

            if MCPConfig.validate_server_config(server_config):
                valid_config[server_name] = server_config
            else:
                logger.warning(f"Invalid server config for '{server_name}': {server_config}")

        logger.info(f"Validated MCP config: {len(valid_config)}/{len(config)} servers valid")
        return valid_config
