from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyhub.llm import LLM
from pyhub.llm.base import BaseLLM
from pyhub.llm.mcp import McpStdioConfig, McpStreamableHttpConfig
from pyhub.llm.tools import Tool


@pytest.fixture
def mock_mcp_server_config():
    """테스트용 MCP 서버 설정"""
    return McpStdioConfig(name="test_calculator", cmd="python calculator.py", description="테스트 계산기")


@pytest.fixture
def mock_multiple_mcp_configs():
    """여러 MCP 서버 설정"""
    return [
        McpStdioConfig(name="calculator", cmd="python calculator.py"),
        McpStreamableHttpConfig(name="greeting", url="http://localhost:8888/mcp"),
    ]


class TestMCPIntegration:
    """MCP 통합 테스트"""

    def test_llm_create_with_mcp_servers(self, mock_mcp_server_config):
        """LLM 생성 시 mcp_servers 파라미터가 제대로 전달되는지 테스트"""
        llm = LLM.create("gpt-4o-mini", mcp_servers=mock_mcp_server_config)

        assert isinstance(llm, BaseLLM)
        assert hasattr(llm, "mcp_servers")
        assert len(llm.mcp_servers) == 1
        assert llm.mcp_servers[0] == mock_mcp_server_config
        assert llm._mcp_client is None
        assert llm._mcp_connected is False
        assert llm._mcp_tools == []

    def test_llm_create_with_multiple_mcp_servers(self, mock_multiple_mcp_configs):
        """여러 MCP 서버 설정이 제대로 처리되는지 테스트"""
        llm = LLM.create("gpt-4o-mini", mcp_servers=mock_multiple_mcp_configs)

        assert len(llm.mcp_servers) == 2
        assert llm.mcp_servers[0].name == "calculator"
        assert llm.mcp_servers[1].name == "greeting"

    @pytest.mark.asyncio
    async def test_create_async_without_mcp(self):
        """MCP 없이 create_async 호출 테스트"""
        llm = await LLM.create_async("gpt-4o-mini")

        assert isinstance(llm, BaseLLM)
        assert llm.mcp_servers == []
        assert llm._mcp_connected is False

    @pytest.mark.asyncio
    async def test_create_async_with_mcp(self, mock_mcp_server_config):
        """MCP와 함께 create_async 호출 테스트"""
        with patch.object(BaseLLM, "initialize_mcp", new=AsyncMock()) as mock_init:
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mock_mcp_server_config)

            assert isinstance(llm, BaseLLM)
            assert len(llm.mcp_servers) == 1
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_mcp_success(self, mock_mcp_server_config):
        """MCP 초기화 성공 테스트"""
        llm = LLM.create("gpt-4o-mini", mcp_servers=mock_mcp_server_config)

        # Mock MCP client and tools
        mock_client = AsyncMock()
        mock_tools = [
            MagicMock(name="add", description="두 숫자를 더합니다"),
            MagicMock(name="subtract", description="두 숫자를 뺍니다"),
        ]

        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiClient:
            mock_instance = MockMultiClient.return_value
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.get_tools = AsyncMock(return_value=mock_tools)

            await llm.initialize_mcp()

            assert llm._mcp_connected is True
            assert llm._mcp_client is not None
            assert len(llm._mcp_tools) == 2
            # 기본 도구에 MCP 도구가 추가되었는지 확인
            assert len(llm.default_tools) == 2

    @pytest.mark.asyncio
    async def test_initialize_mcp_already_connected(self, mock_mcp_server_config):
        """이미 연결된 상태에서 초기화 시도 테스트"""
        llm = LLM.create("gpt-4o-mini", mcp_servers=mock_mcp_server_config)
        llm._mcp_connected = True

        # 로그 경고가 발생하는지 확인
        with patch("pyhub.llm.base.logger") as mock_logger:
            await llm.initialize_mcp()
            mock_logger.warning.assert_called_with("MCP is already connected")

    @pytest.mark.asyncio
    async def test_initialize_mcp_failure(self, mock_mcp_server_config):
        """MCP 초기화 실패 테스트"""
        llm = LLM.create("gpt-4o-mini", mcp_servers=mock_mcp_server_config)

        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiClient:
            MockMultiClient.side_effect = Exception("Connection failed")

            # 예외가 발생해도 프로그램이 중단되지 않아야 함
            await llm.initialize_mcp()

            assert llm._mcp_connected is False
            assert llm._mcp_client is None
            assert llm._mcp_tools == []

    @pytest.mark.asyncio
    async def test_close_mcp_success(self, mock_mcp_server_config):
        """MCP 연결 종료 테스트"""
        llm = LLM.create("gpt-4o-mini", mcp_servers=mock_mcp_server_config)

        # 연결된 상태 시뮬레이션
        mock_client = AsyncMock()
        mock_client.__aexit__ = AsyncMock()
        llm._mcp_client = mock_client
        llm._mcp_connected = True
        llm._mcp_tools = [MagicMock(), MagicMock()]

        # 도구가 default_tools에 추가된 상태 시뮬레이션
        adapted_tools = [
            Tool(name="tool1", description="Tool 1", func=lambda: None),
            Tool(name="tool2", description="Tool 2", func=lambda: None),
        ]
        llm.default_tools = adapted_tools.copy()

        with patch("pyhub.llm.tools.ToolAdapter.adapt_tools", return_value=adapted_tools):
            await llm.close_mcp()

            assert llm._mcp_client is None
            assert llm._mcp_connected is False
            assert llm._mcp_tools == []
            mock_client.__aexit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_mcp_server_config):
        """비동기 컨텍스트 매니저 사용 테스트"""
        with patch.object(BaseLLM, "initialize_mcp", new=AsyncMock()) as mock_init:
            with patch.object(BaseLLM, "close_mcp", new=AsyncMock()) as mock_close:
                async with LLM.create("gpt-4o-mini", mcp_servers=mock_mcp_server_config) as llm:
                    mock_init.assert_called_once()
                    assert isinstance(llm, BaseLLM)

                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_tools_merged_with_default_tools(self, mock_mcp_server_config):
        """MCP 도구가 기본 도구와 병합되는지 테스트"""

        # 기본 도구를 가진 LLM 생성
        def default_tool():
            """기본 도구"""
            return "default"

        llm = LLM.create("gpt-4o-mini", tools=[default_tool], mcp_servers=mock_mcp_server_config)

        # 초기 상태 확인
        assert len(llm.default_tools) == 1
        assert llm.default_tools[0].name == "default_tool"

        # MCP 도구 시뮬레이션
        mock_mcp_tools = [MagicMock(name="mcp_tool1"), MagicMock(name="mcp_tool2")]

        # 초기 default_tools 개수 저장
        initial_tool_count = len(llm.default_tools)

        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiClient:
            mock_instance = MockMultiClient.return_value
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.get_tools = AsyncMock(return_value=mock_mcp_tools)

            # ToolAdapter는 초기화 시 한 번, MCP 도구 추가 시 한 번 호출됨
            with patch("pyhub.llm.tools.ToolAdapter.adapt_tools") as mock_adapt:
                # MCP 도구를 Tool 객체로 변환
                mcp_tool_objects = [
                    Tool(name="mcp_tool1", description="MCP Tool 1", func=lambda: None),
                    Tool(name="mcp_tool2", description="MCP Tool 2", func=lambda: None),
                ]
                # MCP 도구 변환만 mocking (기본 도구는 이미 변환됨)
                mock_adapt.return_value = mcp_tool_objects

                await llm.initialize_mcp()

                # 도구가 병합되었는지 확인
                assert len(llm.default_tools) == initial_tool_count + 2  # 기존 1개 + MCP 2개
                tool_names = [tool.name for tool in llm.default_tools]
                assert "default_tool" in tool_names
                assert "mcp_tool1" in tool_names
                assert "mcp_tool2" in tool_names
