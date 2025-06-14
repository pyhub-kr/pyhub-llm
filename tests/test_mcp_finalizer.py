"""MCP Finalizer 테스트"""

import asyncio
import weakref
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyhub.llm import LLM


class TestMCPFinalizer:
    """MCP Finalizer 기능 테스트"""

    @pytest.mark.asyncio
    async def test_finalizer_registration(self):
        """Finalizer 등록 확인"""
        # MCP 설정
        mcp_config = [{"type": "stdio", "name": "test_server", "cmd": ["echo", "test"]}]

        # Mock registry
        with patch("pyhub.llm.resource_manager.register_mcp_instance") as mock_register:
            mock_finalizer = MagicMock()
            mock_register.return_value = mock_finalizer

            # LLM 생성 (MCP 설정 포함)
            llm = LLM.create("gpt-4o-mini", mcp_servers=mcp_config)

            # Finalizer가 등록되었는지 확인
            mock_register.assert_called_once_with(llm)
            assert llm._finalizer == mock_finalizer

    @pytest.mark.asyncio
    async def test_no_finalizer_without_mcp(self):
        """MCP 없을 때 finalizer 등록 안됨"""
        # MCP 설정 없이 LLM 생성
        llm = LLM.create("gpt-4o-mini")

        # Finalizer가 없어야 함
        assert llm._finalizer is None

    @pytest.mark.asyncio
    async def test_cleanup_timeout(self):
        """Cleanup 타임아웃 테스트"""
        # Mock MultiServerMCPClient
        mock_client = AsyncMock()

        # __aexit__이 지연되는 상황
        async def slow_exit(*args):
            await asyncio.sleep(10)

        mock_client.__aexit__.side_effect = slow_exit

        # LLM 인스턴스 생성
        llm = LLM.create("gpt-4o-mini")
        llm._mcp_client = mock_client
        llm._mcp_connected = True

        # 타임아웃으로 종료되어야 함
        import time

        start = time.time()
        await llm.close_mcp(timeout=1.0)  # 1초 타임아웃
        elapsed = time.time() - start

        # 1초 정도에 종료되어야 함
        assert 0.9 < elapsed < 1.5
        assert llm._mcp_client is None
        assert llm._mcp_connected is False

    @pytest.mark.asyncio
    async def test_resource_registry_cleanup(self):
        """리소스 레지스트리 정리 테스트"""
        from pyhub.llm.resource_manager import MCPResourceRegistry

        registry = MCPResourceRegistry()

        # Mock 인스턴스
        mock_instance = MagicMock()
        mock_instance.close_mcp = AsyncMock()

        # 등록
        instance_id = id(mock_instance)
        weak_ref = weakref.ref(mock_instance)
        registry._instances[instance_id] = weak_ref

        # 정리
        await registry._cleanup_instance(instance_id)

        # close_mcp가 호출되어야 함
        mock_instance.close_mcp.assert_called_once()

    @pytest.mark.asyncio
    async def test_registry_timeout_handling(self):
        """레지스트리 타임아웃 처리"""
        from pyhub.llm.resource_manager import MCPResourceRegistry

        registry = MCPResourceRegistry()

        # Mock 인스턴스 (close_mcp가 지연됨)
        mock_instance = MagicMock()

        async def slow_close():
            await asyncio.sleep(10)

        mock_instance.close_mcp = slow_close

        # 등록
        instance_id = id(mock_instance)
        weak_ref = weakref.ref(mock_instance)
        registry._instances[instance_id] = weak_ref

        # 타임아웃으로 정리되어야 함
        import time

        start = time.time()
        await registry._cleanup_instance(instance_id)
        elapsed = time.time() - start

        # 5초 타임아웃이 적용되어야 함
        assert 4.9 < elapsed < 5.5
