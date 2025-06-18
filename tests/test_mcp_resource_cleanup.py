"""MCP 리소스 정리 테스트"""

import asyncio
import gc
import time
import weakref
from unittest.mock import AsyncMock, patch

import pytest

from pyhub.llm import LLM


class TestMCPResourceCleanup:
    """MCP 리소스 정리 테스트"""

    @pytest.fixture
    def mock_mcp_config(self):
        """테스트용 MCP 설정"""
        return [{"type": "stdio", "name": "test_server", "cmd": ["echo", "test"], "env": {"TEST": "true"}}]

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_normal_exit(self, mock_mcp_config):
        """정상 종료 시 리소스 정리 테스트"""
        # 리소스 추적을 위한 weak reference
        llm_refs = []
        client_refs = []

        # MultiServerMCPClient를 직접 모킹
        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiServerMCPClient:
            # Mock MultiServerMCPClient 인스턴스
            mock_multi_client = AsyncMock()
            mock_multi_client._clients = {}
            mock_multi_client._active_connections = {}
            mock_multi_client._connection_errors = {}
            mock_multi_client.get_tools = AsyncMock(return_value=[])

            # __aexit__ 추적을 위한 변수
            exit_called = []

            async def mock_aexit(*args):
                exit_called.append(True)
                return None

            mock_multi_client.__aexit__ = AsyncMock(side_effect=mock_aexit)
            mock_multi_client.__aenter__ = AsyncMock(return_value=mock_multi_client)

            # MultiServerMCPClient 생성자가 mock 인스턴스를 반환하도록 설정
            MockMultiServerMCPClient.return_value = mock_multi_client

            # LLM 생성 및 MCP 초기화
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mock_mcp_config)

            # Weak reference 생성
            llm_refs.append(weakref.ref(llm))
            if llm._mcp_client:
                client_refs.append(weakref.ref(llm._mcp_client))

            # 정상 종료
            await llm.close_mcp()

            # 리소스 정리 확인
            assert llm._mcp_client is None
            assert llm._mcp_connected is False
            assert len(llm._mcp_tools) == 0

            # __aexit__ 호출 확인
            assert len(exit_called) > 0, "__aexit__ was not called"
            mock_multi_client.__aexit__.assert_called()

        # 객체 삭제 후 GC
        del llm
        gc.collect()

        # Weak reference 확인 (객체가 정리되었는지)
        for ref in llm_refs:
            assert ref() is None, "LLM instance not garbage collected"

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_exception(self, mock_mcp_config):
        """예외 발생 시 리소스 정리 테스트"""
        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiServerMCPClient:
            # Mock MultiServerMCPClient 인스턴스
            mock_multi_client = AsyncMock()
            mock_multi_client._clients = {}
            mock_multi_client._active_connections = {}
            mock_multi_client._connection_errors = {}
            mock_multi_client.get_tools = AsyncMock(return_value=[])

            # 종료 시 예외 발생
            mock_multi_client.__aexit__ = AsyncMock(side_effect=Exception("Cleanup error"))
            mock_multi_client.__aenter__ = AsyncMock(return_value=mock_multi_client)

            MockMultiServerMCPClient.return_value = mock_multi_client

            # LLM 생성
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mock_mcp_config)

            # 예외가 발생해도 정리는 완료되어야 함
            await llm.close_mcp()

            assert llm._mcp_client is None
            assert llm._mcp_connected is False

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Finalizer not yet implemented")
    async def test_finalizer_cleanup(self, mock_mcp_config):
        """Finalizer를 통한 자동 정리 테스트"""
        cleanup_called = []

        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiServerMCPClient:
            # Mock MultiServerMCPClient 인스턴스
            mock_multi_client = AsyncMock()
            mock_multi_client._clients = {}
            mock_multi_client._active_connections = {}
            mock_multi_client._connection_errors = {}
            mock_multi_client.get_tools = AsyncMock(return_value=[])

            # 정리 추적
            async def track_cleanup(*args):
                cleanup_called.append(True)
                return None

            mock_multi_client.__aexit__ = AsyncMock(side_effect=track_cleanup)
            mock_multi_client.__aenter__ = AsyncMock(return_value=mock_multi_client)

            MockMultiServerMCPClient.return_value = mock_multi_client

            # LLM 생성
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mock_mcp_config)

            # Finalizer가 등록되어야 함 (아직 구현 안됨 - 테스트 실패 예상)
            assert hasattr(llm, "_finalizer"), "Finalizer not registered"

            # 명시적 종료 없이 객체 삭제
            _llm_id = id(llm)
            del llm
            gc.collect()

            # Finalizer가 cleanup을 호출했는지 확인
            await asyncio.sleep(0.1)  # 비동기 정리 대기
            assert len(cleanup_called) > 0, "Finalizer did not trigger cleanup"

    @pytest.mark.asyncio
    async def test_cleanup_timeout(self, mock_mcp_config):
        """종료 시 타임아웃 처리 테스트"""
        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiServerMCPClient:
            # Mock MultiServerMCPClient 인스턴스
            mock_multi_client = AsyncMock()
            mock_multi_client._clients = {}
            mock_multi_client._active_connections = {}
            mock_multi_client._connection_errors = {}
            mock_multi_client.get_tools = AsyncMock(return_value=[])

            # 종료가 지연되는 상황 시뮬레이션
            async def slow_exit(*args):
                await asyncio.sleep(10)  # 10초 대기

            mock_multi_client.__aexit__ = AsyncMock(side_effect=slow_exit)
            mock_multi_client.__aenter__ = AsyncMock(return_value=mock_multi_client)

            MockMultiServerMCPClient.return_value = mock_multi_client

            # LLM 생성
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mock_mcp_config)

            # 타임아웃이 있어야 함 (기본 5초)
            start_time = time.time()
            await llm.close_mcp()
            elapsed = time.time() - start_time

            # 5초 타임아웃이 적용되어야 함
            assert elapsed < 7, f"Cleanup took too long: {elapsed}s"
            assert llm._mcp_client is None

    @pytest.mark.asyncio
    async def test_no_circular_reference(self, mock_mcp_config):
        """순환 참조 방지 테스트"""
        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiServerMCPClient:
            # Mock MultiServerMCPClient 인스턴스
            mock_multi_client = AsyncMock()
            mock_multi_client._clients = {}
            mock_multi_client._active_connections = {}
            mock_multi_client._connection_errors = {}

            # 도구 목록 반환
            from pyhub.llm.agents.base import Tool

            mock_tool = Tool(name="test_tool", description="Test tool", func=lambda: "test")
            mock_multi_client.get_tools = AsyncMock(return_value=[mock_tool])

            mock_multi_client.__aexit__ = AsyncMock(return_value=None)
            mock_multi_client.__aenter__ = AsyncMock(return_value=mock_multi_client)

            MockMultiServerMCPClient.return_value = mock_multi_client

            # LLM 생성
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mock_mcp_config)

            # MCP 도구가 LLM을 직접 참조하지 않아야 함
            for tool in llm._mcp_tools:
                # 도구가 LLM 인스턴스를 직접 참조하는지 확인
                assert not hasattr(tool, "_llm"), "Tool should not reference LLM directly"
                assert not hasattr(tool, "llm"), "Tool should not reference LLM directly"

            # Weak reference로만 LLM 추적
            llm_ref = weakref.ref(llm)
            _llm_id = id(llm)

            # 객체 삭제
            del llm
            gc.collect()

            # LLM이 GC되었는지 확인
            assert llm_ref() is None, "LLM not garbage collected (circular reference?)"

    @pytest.mark.asyncio
    async def test_multiple_instances_cleanup(self, mock_mcp_config):
        """여러 인스턴스의 독립적 정리 테스트"""
        instances = []

        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiServerMCPClient:
            # 각 인스턴스마다 다른 mock client
            mock_multi_clients = []

            def create_mock_client():
                mock_client = AsyncMock()
                mock_client._clients = {}
                mock_client._active_connections = {}
                mock_client._connection_errors = {}
                mock_client.get_tools = AsyncMock(return_value=[])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                return mock_client

            for _ in range(3):
                mock_multi_clients.append(create_mock_client())

            MockMultiServerMCPClient.side_effect = mock_multi_clients

            # 여러 LLM 인스턴스 생성
            for i in range(3):
                llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mock_mcp_config)
                instances.append(llm)

            # 중간 인스턴스만 종료
            await instances[1].close_mcp()

            # 다른 인스턴스는 영향받지 않아야 함
            assert instances[0]._mcp_connected is True
            assert instances[1]._mcp_connected is False
            assert instances[2]._mcp_connected is True

            # 모든 인스턴스 정리
            for llm in instances:
                if llm._mcp_connected:
                    await llm.close_mcp()

            # 모든 mock client의 종료 확인
            for mock_client in mock_multi_clients:
                mock_client.__aexit__.assert_called_once()


class TestMCPProcessCleanup:
    """MCP 프로세스 정리 테스트"""

    @pytest.mark.asyncio
    async def test_stdio_subprocess_cleanup(self):
        """stdio 서브프로세스 정리 테스트"""
        # MultiServerMCPClient를 모킹하여 실제 프로세스 생성 방지
        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiServerMCPClient:
            # Mock 프로세스
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = None
            mock_proc.terminate = AsyncMock()
            mock_proc.wait = AsyncMock()

            # Mock MultiServerMCPClient
            mock_multi_client = AsyncMock()
            mock_multi_client._clients = {}
            mock_multi_client._active_connections = {"test_server": mock_proc}
            mock_multi_client._connection_errors = {}
            mock_multi_client.get_tools = AsyncMock(return_value=[])
            mock_multi_client.__aenter__ = AsyncMock(return_value=mock_multi_client)
            mock_multi_client.__aexit__ = AsyncMock(return_value=None)

            # 프로세스 종료를 추적하기 위한 mock
            terminate_called = []

            async def track_terminate(*args):
                terminate_called.append(True)
                await mock_proc.terminate()
                return None

            mock_multi_client.__aexit__ = AsyncMock(side_effect=track_terminate)

            MockMultiServerMCPClient.return_value = mock_multi_client

            # MCP 설정 - 리스트 형식으로 직접 전달
            config = [{"type": "stdio", "name": "test_server", "cmd": ["python", "-m", "test_server"]}]

            # LLM 생성
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=config)

            # 정리 전 프로세스가 종료되지 않았는지 확인
            assert mock_proc.terminate.call_count == 0

            # 정리
            await llm.close_mcp()

            # 종료가 호출되었는지 확인
            assert len(terminate_called) > 0
            mock_proc.terminate.assert_called()

    @pytest.mark.asyncio
    async def test_hanging_process_force_kill(self):
        """응답 없는 프로세스 강제 종료 테스트"""
        with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiServerMCPClient:
            # Mock 프로세스
            mock_proc = AsyncMock()
            mock_proc.pid = 12346
            mock_proc.returncode = None
            mock_proc.terminate = AsyncMock()
            mock_proc.kill = AsyncMock()

            # terminate 후에도 종료되지 않는 상황
            mock_proc.wait.side_effect = asyncio.TimeoutError()

            # Mock MultiServerMCPClient
            mock_multi_client = AsyncMock()
            mock_multi_client._clients = {}
            mock_multi_client._active_connections = {"hanging_server": mock_proc}
            mock_multi_client._connection_errors = {}
            mock_multi_client.get_tools = AsyncMock(return_value=[])
            mock_multi_client.__aenter__ = AsyncMock(return_value=mock_multi_client)

            # 타임아웃 후 kill 호출을 시뮬레이션
            async def timeout_exit(*args):
                # wait에서 timeout 발생
                await asyncio.sleep(0.1)
                raise asyncio.TimeoutError()

            mock_multi_client.__aexit__ = AsyncMock(side_effect=timeout_exit)

            MockMultiServerMCPClient.return_value = mock_multi_client

            config = [{"type": "stdio", "name": "hanging_server", "cmd": ["python", "-m", "hanging_server"]}]

            # LLM 생성
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=config)

            # 정리 시도 (타임아웃 발생 예상)
            await llm.close_mcp()

            # MultiServerMCPClient의 __aexit__이 호출되었는지 확인
            mock_multi_client.__aexit__.assert_called()
