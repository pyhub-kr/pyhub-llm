"""MCP 서버 설정 파일 로더"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .configs import McpConfig, create_mcp_config


def load_mcp_config(config_source: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]) -> List[McpConfig]:
    """
    MCP 서버 설정을 로드합니다.

    Args:
        config_source: 다음 중 하나
            - 설정 파일 경로 (JSON/YAML)
            - 설정 dict (mcpServers 키 포함)
            - MCP 서버 설정 리스트

    Returns:
        McpConfig 인스턴스 리스트

    Raises:
        FileNotFoundError: 파일이 존재하지 않음
        json.JSONDecodeError: JSON 파싱 실패
        yaml.YAMLError: YAML 파싱 실패
        ValueError: 설정 검증 실패

    Examples:
        >>> # 파일에서 로드
        >>> configs = load_mcp_config("mcp_config.json")

        >>> # dict에서 로드
        >>> configs = load_mcp_config({
        ...     "mcpServers": [
        ...         {"cmd": "python server.py"},
        ...         {"url": "http://localhost:8080"}
        ...     ]
        ... })

        >>> # 리스트에서 로드
        >>> configs = load_mcp_config([
        ...     {"cmd": "python server.py"},
        ...     {"url": "http://localhost:8080"}
        ... ])
    """
    # 1. 파일 경로인 경우
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        content = path.read_text(encoding="utf-8")

        # 파일 확장자에 따라 파서 선택
        if path.suffix.lower() in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError("YAML 파일을 읽으려면 PyYAML을 설치하세요: pip install PyYAML")
            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"YAML 파싱 실패: {e}")
        else:
            # JSON으로 파싱
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"JSON 파싱 실패: {e}", content, e.pos)

        return _process_config_data(data)

    # 2. 딕셔너리인 경우
    elif isinstance(config_source, dict):
        return _process_config_data(config_source)

    # 3. 리스트인 경우
    elif isinstance(config_source, list):
        return _process_config_list(config_source)

    else:
        raise TypeError(f"지원하지 않는 config_source 타입: {type(config_source)}")


def _process_config_data(data: Dict[str, Any]) -> List[McpConfig]:
    """설정 딕셔너리를 처리하여 McpConfig 리스트 반환"""
    if not isinstance(data, dict):
        raise ValueError("설정은 딕셔너리여야 합니다")

    # mcpServers 키가 있는 경우
    if "mcpServers" in data:
        servers_config = data["mcpServers"]
        if not isinstance(servers_config, list):
            raise ValueError("mcpServers는 리스트여야 합니다")
        return _process_config_list(servers_config)

    # mcpServers 키가 없는 경우, 전체를 단일 서버 설정으로 간주
    else:
        return [_create_config_from_dict(data)]


def _process_config_list(config_list: List[Dict[str, Any]]) -> List[McpConfig]:
    """설정 리스트를 처리하여 McpConfig 리스트 반환"""
    if not isinstance(config_list, list):
        raise ValueError("설정 리스트는 list여야 합니다")

    configs = []
    for i, config_dict in enumerate(config_list):
        try:
            config = _create_config_from_dict(config_dict)
            configs.append(config)
        except Exception as e:
            raise ValueError(f"설정 {i}번 처리 중 오류: {e}")

    return configs


def _create_config_from_dict(config_dict: Dict[str, Any]) -> McpConfig:
    """딕셔너리에서 McpConfig 생성"""
    if not isinstance(config_dict, dict):
        raise ValueError("개별 설정은 딕셔너리여야 합니다")

    # 필수 필드 검증 (type 필드가 있는 경우 검증)
    if "type" in config_dict:
        transport_type = config_dict.pop("type")  # type 필드는 제거하고 transport로 사용
        config_dict["transport"] = transport_type

    # transport별 필수 필드 검증
    transport = config_dict.get("transport")
    if transport == "stdio":
        if "cmd" not in config_dict and "command" not in config_dict:
            raise ValueError("stdio transport에는 'cmd' 또는 'command' 필드가 필요합니다")

        # command + args 형태를 cmd로 변환
        if "command" in config_dict:
            command = config_dict.pop("command")
            args = config_dict.pop("args", [])
            if args:
                if isinstance(args, list):
                    config_dict["cmd"] = [command] + args
                else:
                    raise ValueError("args는 리스트여야 합니다")
            else:
                config_dict["cmd"] = command

    elif transport in ("streamable_http", "websocket", "sse"):
        if "url" not in config_dict:
            raise ValueError(f"{transport} transport에는 'url' 필드가 필요합니다")

    # transport가 명시되지 않은 경우, 자동 감지를 위해 cmd나 url 중 하나는 있어야 함
    elif not transport:
        if "cmd" not in config_dict and "url" not in config_dict and "command" not in config_dict:
            raise ValueError("'cmd', 'command', 또는 'url' 중 하나는 반드시 지정되어야 합니다")

    # 환경 변수 정규화
    if "env" in config_dict:
        env = config_dict["env"]
        if env is not None and not isinstance(env, dict):
            raise ValueError("env는 딕셔너리여야 합니다")

    try:
        return create_mcp_config(config_dict)
    except Exception as e:
        raise ValueError(f"McpConfig 생성 실패: {e}")
