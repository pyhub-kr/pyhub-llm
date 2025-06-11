"""Multi-server MCP client implementation with extended transport support."""

import asyncio
import logging
from typing import Any, Dict, List, Union

from pyhub.llm.agents.base import Tool

from .client import MCPClient
from .configs import McpServerConfig
from .loader import load_mcp_tools
from .transports import create_transport

logger = logging.getLogger(__name__)


class MultiServerMCPClient:
    """여러 MCP 서버를 동시에 관리하는 클라이언트 (모든 Transport 지원)"""

    def __init__(self, servers: Union[Dict[str, Dict[str, Any]], List[McpServerConfig]], prefix_tools: bool = False):
        """
        Args:
            servers: 서버 설정
                - Dict[str, Dict[str, Any]]: 기존 딕셔너리 방식 (하위 호환)
                - List[McpServerConfig]: 새로운 dataclass 방식 (권장)
                
            prefix_tools: 도구 이름에 서버 이름을 prefix로 추가할지 여부
            
        Examples:
            >>> # Dataclass 방식 (권장)
            >>> from pyhub.llm.mcp.configs import McpStdioConfig, McpStreamableHttpConfig
            >>> servers = [
            ...     McpStdioConfig(
            ...         name="calculator",
            ...         cmd="pyhub-llm mcp-server run calculator"
            ...     ),
            ...     McpStreamableHttpConfig(
            ...         name="greeting",
            ...         url="http://localhost:8888/mcp"
            ...     )
            ... ]
            >>> client = MultiServerMCPClient(servers)
            
            >>> # 기존 딕셔너리 방식 (하위 호환)
            >>> servers = {
            ...     "calculator": {
            ...         "transport": "stdio",
            ...         "command": "pyhub-llm",
            ...         "args": ["mcp-server", "run", "calculator"]
            ...     }
            ... }
            >>> client = MultiServerMCPClient(servers)
        """
        # Dataclass 리스트를 딕셔너리로 변환
        if isinstance(servers, list):
            self.servers = {}
            for config in servers:
                if isinstance(config, McpServerConfig):
                    self.servers[config.name] = config.to_dict()
                else:
                    raise TypeError(f"리스트 요소는 McpServerConfig여야 합니다: {type(config)}")
        else:
            self.servers = servers
            
        self.prefix_tools = prefix_tools
        self._clients: Dict[str, MCPClient] = {}
        self._active_connections: Dict[str, Any] = {}
        self._connection_errors: Dict[str, str] = {}
        self._connection_tasks: Dict[str, asyncio.Task] = {}

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        # 태스크를 사용하지 않고 직접 연결 - 컨텍스트 문제 해결
        for server_name, config in self.servers.items():
            await self._connect_server(server_name, config)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        # 각 연결을 순차적으로 종료 - 동일한 태스크에서 실행
        for server_name, connection_context in list(self._active_connections.items()):
            try:
                await connection_context.__aexit__(None, None, None)
                logger.info(f"Disconnected from '{server_name}'")
            except Exception as e:
                logger.error(f"Error disconnecting from '{server_name}': {e}")

        # 정리
        self._clients.clear()
        self._active_connections.clear()
        self._connection_tasks.clear()

    async def _connect_server(self, server_name: str, config: Dict[str, Any]):
        """개별 서버에 연결 (모든 Transport 지원)"""
        try:
            # Transport 타입 자동 추론
            from .transports import infer_transport_type

            # 설정 복사 (원본 수정 방지)
            config = config.copy()

            # transport 타입이 명시되지 않은 경우 추론
            if "transport" not in config:
                config["transport"] = infer_transport_type(config)

            transport_type = config["transport"]
            logger.debug(f"Connecting to '{server_name}' using {transport_type} transport")

            # Transport 생성 및 검증
            transport = create_transport(config)  # noqa

            # 클라이언트 생성 (새로운 설정 기반 방식)
            client = MCPClient(config)

            # 연결 컨텍스트 매니저 저장
            connection_context = client.connect()
            # 연결 시작
            await connection_context.__aenter__()

            self._clients[server_name] = client
            self._active_connections[server_name] = connection_context

            # 현재 태스크 저장 - 이제 필요없음
            # self._connection_tasks[server_name] = asyncio.current_task()

            # 연결 성공 로그 (transport 정보 포함)
            description = config.get("description", "")
            if description:
                logger.info(f"✅ Connected to '{server_name}' via {transport_type}: {description}")
            else:
                logger.info(f"✅ Connected to '{server_name}' via {transport_type}")

        except Exception as e:
            error_msg = str(e)
            self._connection_errors[server_name] = error_msg
            logger.error(f"❌ Failed to connect to '{server_name}': {error_msg}")
            # 연결 실패해도 예외를 던지지 않음 (다른 서버 연결 계속 진행)

    async def get_tools(self) -> List[Tool]:
        """모든 서버에서 도구를 가져와 통합"""
        all_tools = []

        # 각 서버에서 도구 가져오기
        for server_name, client in self._clients.items():
            try:
                # 서버 설정에서 필터 가져오기
                filter_tools = self.servers[server_name].get("filter_tools")

                # 도구 로드
                tools = await load_mcp_tools(client, filter_tools)

                # 도구 이름에 prefix 추가 (옵션)
                if self.prefix_tools:
                    for tool in tools:
                        original_name = tool.name
                        tool.name = f"{server_name}.{original_name}"
                        tool.description = f"[{server_name}] {tool.description}"
                        logger.debug(f"Renamed tool '{original_name}' to '{tool.name}'")

                all_tools.extend(tools)
                logger.info(f"Loaded {len(tools)} tools from '{server_name}'")

            except Exception as e:
                logger.error(f"Failed to load tools from '{server_name}': {e}")
                # 실패한 서버는 건너뛰고 계속 진행
                continue

        logger.info(f"Total {len(all_tools)} tools loaded from {len(self._clients)} servers")
        return all_tools

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """모든 서버에서 도구 정보를 가져와 Function Calling 형식으로 반환"""
        all_tools = []

        for server_name, client in self._clients.items():
            try:
                # MCP 도구 정보 가져오기
                mcp_tools = await client.list_tools()

                # 서버 정보를 도구에 추가
                for tool in mcp_tools:
                    tool["_mcp_server"] = server_name
                    tool["_mcp_config"] = self.servers[server_name]

                    # prefix 옵션 적용
                    if self.prefix_tools:
                        original_name = tool["name"]
                        tool["name"] = f"{server_name}.{original_name}"
                        tool["description"] = f"[{server_name}] {tool.get('description', '')}"

                all_tools.extend(mcp_tools)
                logger.debug(f"Loaded {len(mcp_tools)} tools from '{server_name}'")

            except Exception as e:
                logger.error(f"Failed to get tools from '{server_name}': {e}")
                continue

        logger.info(f"Total {len(all_tools)} tools loaded from {len(self._clients)} connected servers")
        return all_tools

    async def list_servers(self) -> Dict[str, Dict[str, Any]]:
        """연결된 서버 목록과 상태 반환"""
        server_status = {}

        for server_name, config in self.servers.items():
            transport_type = config.get("transport", "unknown")

            status = {
                "config": config,
                "transport": transport_type,
                "connected": server_name in self._clients,
                "tools_count": 0,
                "description": config.get("description", ""),
            }

            # 연결 오류가 있는 경우
            if server_name in self._connection_errors:
                status["error"] = self._connection_errors[server_name]

            # 연결된 경우 도구 수 확인
            if server_name in self._clients:
                try:
                    tools = await self._clients[server_name].list_tools()
                    status["tools_count"] = len(tools)
                except Exception as e:
                    status["error"] = str(e)

            server_status[server_name] = status

        return server_status

    def get_connected_servers(self) -> List[str]:
        """연결된 서버 이름 목록 반환"""
        return list(self._clients.keys())

    def get_failed_servers(self) -> Dict[str, str]:
        """연결 실패한 서버와 오류 메시지 반환"""
        return self._connection_errors.copy()


async def create_multi_server_client_from_config(config: Dict[str, Any]) -> MultiServerMCPClient:
    """설정에서 MultiServerMCPClient 생성"""
    servers = {}

    # mcp.servers 섹션에서 서버 설정 읽기
    mcp_config = config.get("mcp", {})
    servers_config = mcp_config.get("servers", {})

    for server_name, server_config in servers_config.items():
        # 필수 필드 확인
        if "command" not in server_config:
            logger.warning(f"Server '{server_name}' missing 'command', skipping")
            continue

        servers[server_name] = server_config

    if not servers:
        raise ValueError("No valid MCP servers found in configuration")

    # prefix_tools 옵션
    prefix_tools = mcp_config.get("prefix_tools", False)

    return MultiServerMCPClient(servers, prefix_tools=prefix_tools)
