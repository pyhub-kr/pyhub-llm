"""MCP 서버 설정 파일 로더"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union
from urllib.parse import urlparse

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .configs import (
    McpServerConfig,
    McpStdioConfig,
    McpStreamableHttpConfig,
    McpWebSocketConfig,
    McpSseConfig
)


def load_mcp_config(
    config_source: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]
) -> List[McpServerConfig]:
    """
    MCP 서버 설정을 로드합니다.
    
    Args:
        config_source: 다음 중 하나
            - 설정 파일 경로 (JSON/YAML)
            - 설정 dict (mcpServers 키 포함)
            - MCP 서버 설정 리스트
    
    Returns:
        McpServerConfig 인스턴스 리스트
    
    Raises:
        FileNotFoundError: 파일이 존재하지 않음
        json.JSONDecodeError: JSON 파싱 실패
        yaml.YAMLError: YAML 파싱 실패
        ValueError: 설정 검증 실패
    """
    # 1. 파일 경로인 경우
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        content = path.read_text(encoding='utf-8')
        
        if path.suffix.lower() == '.json':
            data = json.loads(content)
        elif path.suffix.lower() in ('.yaml', '.yml'):
            if not HAS_YAML:
                raise ImportError(
                    "PyYAML is required to load YAML files. "
                    "Install it with: pip install pyhub-llm[mcp]"
                )
            data = yaml.safe_load(content)
        else:
            # 확장자로 판단할 수 없으면 JSON 시도 후 YAML 시도
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                if not HAS_YAML:
                    raise ValueError(
                        f"Unable to parse file as JSON: {path}. "
                        "To try YAML parsing, install PyYAML with: pip install pyhub-llm[mcp]"
                    )
                try:
                    data = yaml.safe_load(content)
                except yaml.YAMLError:
                    raise ValueError(f"Unable to parse file as JSON or YAML: {path}")
        
        # mcpServers 키 추출
        if isinstance(data, dict) and 'mcpServers' in data:
            server_configs = data['mcpServers']
        else:
            raise ValueError("Config file must contain 'mcpServers' key")
    
    # 2. dict인 경우
    elif isinstance(config_source, dict):
        if 'mcpServers' in config_source:
            server_configs = config_source['mcpServers']
        else:
            raise ValueError("Config dict must contain 'mcpServers' key")
    
    # 3. list인 경우
    elif isinstance(config_source, list):
        server_configs = config_source
    
    else:
        raise TypeError(f"Invalid config_source type: {type(config_source)}")
    
    # 서버 설정 리스트가 비어있으면 빈 리스트 반환
    if not server_configs:
        return []
    
    # 각 서버 설정을 McpServerConfig 인스턴스로 변환
    configs = []
    for idx, server_dict in enumerate(server_configs):
        try:
            # 유효성 검사
            validate_mcp_config(server_dict)
            
            # 정규화
            normalized = normalize_mcp_config(server_dict)
            
            # 타입별 인스턴스 생성
            config_type = normalized.pop('type')
            
            if config_type == 'stdio':
                config = McpStdioConfig(**normalized)
            elif config_type == 'streamable_http':
                config = McpStreamableHttpConfig(**normalized)
            elif config_type == 'websocket':
                config = McpWebSocketConfig(**normalized)
            elif config_type == 'sse':
                config = McpSseConfig(**normalized)
            else:
                # validate_mcp_config에서 이미 검사했으므로 여기 도달하면 안됨
                raise ValueError(f"Unknown type: {config_type}")
            
            configs.append(config)
            
        except Exception as e:
            raise ValueError(f"Error in server config at index {idx}: {str(e)}")
    
    return configs


def validate_mcp_config(config: Dict[str, Any]) -> None:
    """
    MCP 서버 설정의 유효성을 검사합니다.
    
    Args:
        config: 검증할 설정 dict
    
    Raises:
        ValueError: 유효성 검사 실패
    """
    # 필수 필드 검사
    if 'type' not in config:
        raise ValueError("Missing required field 'type'")
    if 'name' not in config:
        raise ValueError("Missing required field 'name'")
    
    # type 값 검사
    valid_types = {'stdio', 'streamable_http', 'websocket', 'sse'}
    if config['type'] not in valid_types:
        raise ValueError(f"Invalid type '{config['type']}'. Must be one of: {valid_types}")
    
    # 타입별 필수 필드 검사
    if config['type'] == 'stdio':
        if 'cmd' not in config:
            raise ValueError("'cmd' is required for stdio type")
    
    elif config['type'] in ('streamable_http', 'websocket', 'sse'):
        if 'url' not in config:
            raise ValueError(f"'url' is required for {config['type']} type")
        
        # URL 형식 검사
        try:
            parsed = urlparse(config['url'])
            if not parsed.scheme or not parsed.netloc:
                raise ValueError()
            
            # 타입별 스키마 검사
            if config['type'] == 'streamable_http' and parsed.scheme not in ('http', 'https'):
                raise ValueError("streamable_http requires http or https URL")
            elif config['type'] == 'websocket' and parsed.scheme not in ('ws', 'wss'):
                raise ValueError("websocket requires ws or wss URL")
            elif config['type'] == 'sse' and parsed.scheme not in ('http', 'https'):
                raise ValueError("sse requires http or https URL")
                
        except Exception:
            raise ValueError(f"Invalid URL format: {config.get('url')}")


def normalize_mcp_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP 서버 설정을 정규화합니다.
    
    Args:
        config: 정규화할 설정 dict
    
    Returns:
        정규화된 설정 dict
    """
    normalized = config.copy()
    
    # timeout 정규화 (문자열 -> 숫자)
    if 'timeout' in normalized:
        try:
            normalized['timeout'] = int(normalized['timeout'])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid timeout value: {normalized['timeout']}")
    
    # filter_tools 정규화
    if 'filter_tools' in normalized:
        filter_tools = normalized['filter_tools']
        if isinstance(filter_tools, str):
            # 쉼표로 구분된 문자열을 리스트로 변환
            normalized['filter_tools'] = [
                tool.strip() for tool in filter_tools.split(',') 
                if tool.strip()
            ]
        elif filter_tools is None:
            # None은 그대로 유지
            pass
        elif isinstance(filter_tools, list):
            # 리스트는 그대로 유지
            pass
        else:
            raise ValueError(f"Invalid filter_tools type: {type(filter_tools)}")
    
    # env 정규화 (모든 값을 문자열로)
    if 'env' in normalized and normalized['env'] is not None:
        env = normalized['env']
        if not isinstance(env, dict):
            raise ValueError(f"'env' must be a dict, got {type(env)}")
        
        normalized['env'] = {
            str(k): str(v) for k, v in env.items()
        }
    
    return normalized