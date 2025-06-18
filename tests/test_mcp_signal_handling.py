"""MCP 시그널 핸들링 테스트"""

import asyncio
import signal
import sys
import time
import weakref
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        import weakref

        # Mock 인스턴스
        mock_instance = MagicMock()
        mock_instance.close_mcp = AsyncMock()

        # Registry에 등록
        registry = MCPResourceRegistry()
        instance_id = id(mock_instance)
        registry._instances[instance_id] = weakref.ref(mock_instance)

        # 시그널 핸들러 호출
        with patch("sys.exit") as _mock_exit:
            with patch.object(registry, "_async_cleanup_all", new_callable=AsyncMock) as mock_cleanup:
                # 시그널 핸들러 직접 호출
                registry._signal_handler(signal.SIGINT, None)

                # cleanup이 시작되어야 함
                await asyncio.sleep(0.1)  # 비동기 작업 대기
                mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_on_signal(self):
        """시그널 수신 시 graceful shutdown 테스트"""
        # Registry 인스턴스 가져오기 (singleton)
        registry = MCPResourceRegistry()

        # Mock 인스턴스 생성
        mock_llm = MagicMock()
        mock_llm._mcp_connected = True
        mock_llm._mcp_client = AsyncMock()
        mock_llm._mcp_tools = []
        mock_llm.close_mcp = AsyncMock()

        # Registry에 수동 등록
        llm_id = id(mock_llm)
        registry._instances[llm_id] = weakref.ref(mock_llm)

        # Registry 확인
        assert llm_id in registry._instances

        # 시그널 시뮬레이션
        with patch("sys.exit") as _mock_exit:
            # 시그널 핸들러 호출
            registry._signal_handler(signal.SIGINT, None)

            # 비동기 cleanup 대기
            await asyncio.sleep(0.2)

            # cleanup이 호출되어야 함
            mock_llm.close_mcp.assert_called()

    @pytest.mark.asyncio
    async def test_multiple_instances_signal_cleanup(self):
        """여러 인스턴스가 있을 때 시그널 처리"""
        # Registry 인스턴스 가져오기 (singleton)
        registry = MCPResourceRegistry()

        # Mock 인스턴스들 생성
        instances = []
        for i in range(3):
            mock_llm = MagicMock()
            mock_llm._mcp_connected = True
            mock_llm._mcp_client = AsyncMock()
            mock_llm._mcp_tools = []
            mock_llm.close_mcp = AsyncMock()
            instances.append(mock_llm)

            # Registry에 수동 등록
            llm_id = id(mock_llm)
            registry._instances[llm_id] = weakref.ref(mock_llm)

        # Registry 확인
        for llm in instances:
            assert id(llm) in registry._instances

        # 시그널 핸들러 호출
        with patch("sys.exit") as _mock_exit:
            registry._signal_handler(signal.SIGINT, None)

            # 비동기 cleanup 대기
            await asyncio.sleep(0.2)

            # 모든 인스턴스가 정리되어야 함
            for llm in instances:
                llm.close_mcp.assert_called()

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
        import weakref

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

        # 5초 타임아웃이 적용되어야 함
        assert 4.9 < elapsed < 5.5
