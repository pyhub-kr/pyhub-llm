"""MCP 시그널 핸들링 테스트"""

import asyncio
import signal
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyhub.llm import LLM
from pyhub.llm.resource_manager import MCPResourceRegistry


class TestSignalHandling:
    """시그널 핸들링 테스트"""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """각 테스트 전후에 레지스트리 리셋"""
        MCPResourceRegistry.reset()
        yield
        MCPResourceRegistry.reset()

    def test_signal_handler_registration(self):
        """시그널 핸들러 등록 테스트"""
        # 새 레지스트리 인스턴스 (테스트 모드에서는 시그널 핸들러 비활성화)
        registry = MCPResourceRegistry()

        # 수동으로 시그널 핸들러 활성화
        registry.enable_signal_handlers()

        # 시그널 핸들러가 등록되었는지 확인
        if sys.platform != "win32":
            # Unix/Linux/macOS
            assert signal.SIGTERM in registry._original_handlers
            assert signal.SIGINT in registry._original_handlers
        else:
            # Windows
            assert signal.SIGINT in registry._original_handlers

    @pytest.mark.asyncio
    async def test_sigint_handling(self):
        """SIGINT 처리 테스트"""
        # Mock 인스턴스
        mock_instance = MagicMock()
        mock_instance.close_mcp = AsyncMock()

        # Registry에 등록
        registry = MCPResourceRegistry()
        instance_id = id(mock_instance)
        registry._instances[instance_id] = weakref.ref(mock_instance)

        # 시그널 핸들러 호출
        with patch("sys.exit") as mock_exit:
            with patch.object(registry, "_async_cleanup_all", new_callable=AsyncMock) as mock_cleanup:
                # 시그널 핸들러 직접 호출
                registry._signal_handler(signal.SIGINT, None)

                # cleanup이 시작되어야 함
                await asyncio.sleep(0.1)  # 비동기 작업 대기
                mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_on_signal(self):
        """시그널 수신 시 graceful shutdown 테스트"""
        # 실제 LLM 인스턴스 생성
        mcp_config = [{"type": "stdio", "name": "test_server", "cmd": ["echo", "test"]}]

        # Mock MCP client
        with patch("pyhub.llm.mcp.multi_client.create_multi_server_client_from_config") as mock_create:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get_tools.return_value = []
            mock_create.return_value = mock_client

            # LLM 생성
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mcp_config)

            # Registry 확인
            registry = MCPResourceRegistry()
            assert id(llm) in registry._instances

            # 시그널 시뮬레이션
            with patch("sys.exit") as mock_exit:
                # cleanup 추적
                exit_called = []

                async def track_exit(*args):
                    exit_called.append(True)
                    return None

                mock_client.__aexit__.side_effect = track_exit

                # 시그널 핸들러 호출
                registry._signal_handler(signal.SIGINT, None)

                # 비동기 cleanup 대기
                await asyncio.sleep(0.2)

                # cleanup이 호출되어야 함
                assert len(exit_called) > 0
                assert llm._mcp_client is None

    @pytest.mark.asyncio
    async def test_multiple_instances_signal_cleanup(self):
        """여러 인스턴스가 있을 때 시그널 처리"""
        instances = []
        mock_clients = []

        # Mock 설정
        with patch("pyhub.llm.mcp.multi_client.create_multi_server_client_from_config") as mock_create:
            # 3개의 인스턴스 생성
            for i in range(3):
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get_tools.return_value = []
                mock_clients.append(mock_client)

            mock_create.side_effect = mock_clients

            # LLM 인스턴스들 생성
            mcp_config = [{"type": "stdio", "name": f"server_{i}", "cmd": ["echo", f"test_{i}"]}]

            for i in range(3):
                llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mcp_config)
                instances.append(llm)

            # Registry 확인
            registry = MCPResourceRegistry()
            for llm in instances:
                assert id(llm) in registry._instances

            # 시그널 핸들러 호출
            with patch("sys.exit") as mock_exit:
                registry._signal_handler(signal.SIGINT, None)

                # 비동기 cleanup 대기
                await asyncio.sleep(0.2)

                # 모든 인스턴스가 정리되어야 함
                for mock_client in mock_clients:
                    mock_client.__aexit__.assert_called()

                for llm in instances:
                    assert llm._mcp_client is None

    def test_original_handler_preservation(self):
        """기존 시그널 핸들러 보존 테스트"""
        # 원본 핸들러
        original_called = []

        def original_handler(signum, frame):
            original_called.append(signum)

        # 원본 핸들러 설정
        old_handler = signal.signal(signal.SIGINT, original_handler)

        try:
            # Registry 생성
            registry = MCPResourceRegistry()

            # 수동으로 시그널 핸들러 등록
            registry.enable_signal_handlers()

            # 원본 핸들러가 저장되었는지 확인
            assert registry._original_handlers[signal.SIGINT] == original_handler

            # 시그널 핸들러 호출
            with patch("sys.exit"):
                with patch.object(registry, "_async_cleanup_all", new_callable=AsyncMock):
                    registry._signal_handler(signal.SIGINT, None)

                    # 원본 핸들러도 호출되어야 함
                    assert len(original_called) > 0
                    assert original_called[0] == signal.SIGINT
        finally:
            # 원래 핸들러 복원
            signal.signal(signal.SIGINT, old_handler)

    @pytest.mark.asyncio
    async def test_cleanup_timeout_on_signal(self):
        """시그널 처리 시 전체 타임아웃"""
        # Mock 인스턴스 (cleanup이 오래 걸림)
        slow_instances = []

        for i in range(3):
            mock_instance = MagicMock()

            async def slow_cleanup():
                await asyncio.sleep(20)  # 20초

            mock_instance.close_mcp = slow_cleanup
            slow_instances.append(mock_instance)

        # Registry에 등록
        registry = MCPResourceRegistry()
        for instance in slow_instances:
            instance_id = id(instance)
            registry._instances[instance_id] = weakref.ref(instance)

        # Cleanup 시작
        start = time.time()
        await registry._async_cleanup_all()
        elapsed = time.time() - start

        # 10초 타임아웃이 적용되어야 함
        assert 9.9 < elapsed < 10.5


# weakref import 추가
import weakref
