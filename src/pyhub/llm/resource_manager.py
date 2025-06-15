"""MCP 리소스 관리를 위한 유틸리티"""

import asyncio
import atexit
import logging
import os
import signal
import sys
import weakref
from functools import wraps
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class MCPResourceRegistry:
    """전역 MCP 리소스 레지스트리"""

    _instance = None
    _instances: Dict[int, weakref.ref] = {}
    _cleanup_tasks: Set[asyncio.Task] = set()
    _shutdown_event: Optional[asyncio.Event] = None
    _original_handlers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._instances = {}
        self._cleanup_tasks = set()
        self._shutdown_event = None

        # atexit 핸들러 등록
        atexit.register(self._atexit_cleanup)

        # 시그널 핸들러 등록 (테스트 환경에서는 비활성화)
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            self._register_signal_handlers()

    def _register_signal_handlers(self):
        """시그널 핸들러 등록"""
        if sys.platform == "win32":
            # Windows에서는 SIGTERM이 없음
            signals = [signal.SIGINT]
        else:
            signals = [signal.SIGINT, signal.SIGTERM]

        for sig in signals:
            # 기존 핸들러 저장
            self._original_handlers[sig] = signal.signal(sig, self._signal_handler)

    def enable_signal_handlers(self):
        """시그널 핸들러를 수동으로 활성화 (테스트용)"""
        self._register_signal_handlers()

    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")

        # 비동기 cleanup 시작
        try:
            _ = asyncio.get_running_loop()
            asyncio.create_task(self._async_cleanup_all())
        except RuntimeError:
            # 실행 중인 이벤트 루프가 없는 경우
            asyncio.run(self._async_cleanup_all())

        # 원래 핸들러 호출
        original_handler = self._original_handlers.get(signum)
        if original_handler and original_handler != signal.SIG_DFL:
            original_handler(signum, frame)
        else:
            sys.exit(0)

    def register(self, instance: Any) -> weakref.finalize:
        """인스턴스 등록 및 finalizer 생성"""
        instance_id = id(instance)
        weak_ref = weakref.ref(instance)
        self._instances[instance_id] = weak_ref

        # Finalizer 생성
        def cleanup():
            logger.debug(f"Finalizer called for instance {instance_id}")
            # 비동기 cleanup을 동기 컨텍스트에서 실행
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프가 있으면 태스크 생성
                    task = asyncio.create_task(self._cleanup_instance(instance_id))
                    self._cleanup_tasks.add(task)
                    task.add_done_callback(self._cleanup_tasks.discard)
                else:
                    # 루프가 없으면 새로 생성
                    asyncio.run(self._cleanup_instance(instance_id))
            except Exception as e:
                logger.error(f"Error in finalizer: {e}")
            finally:
                # 레지스트리에서 제거
                self._instances.pop(instance_id, None)

        finalizer = weakref.finalize(instance, cleanup)
        return finalizer

    async def _cleanup_instance(self, instance_id: int):
        """특정 인스턴스 정리"""
        weak_ref = self._instances.get(instance_id)
        if weak_ref:
            instance = weak_ref()
            if instance and hasattr(instance, "close_mcp"):
                try:
                    await asyncio.wait_for(instance.close_mcp(), timeout=5.0)  # 5초 타임아웃
                    logger.debug(f"Successfully cleaned up instance {instance_id}")
                except asyncio.TimeoutError:
                    logger.warning(f"Cleanup timeout for instance {instance_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up instance {instance_id}: {e}")

    async def _async_cleanup_all(self):
        """모든 인스턴스 비동기 정리"""
        logger.info("Starting cleanup of all MCP instances...")

        cleanup_tasks = []
        for instance_id, weak_ref in list(self._instances.items()):
            instance = weak_ref()
            if instance:
                cleanup_tasks.append(self._cleanup_instance(instance_id))

        if cleanup_tasks:
            # 모든 cleanup 동시 실행 (최대 10초 대기)
            try:
                await asyncio.wait_for(asyncio.gather(*cleanup_tasks, return_exceptions=True), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Global cleanup timeout")

        logger.info("MCP cleanup completed")

    def _atexit_cleanup(self):
        """프로세스 종료 시 정리"""
        logger.debug("atexit handler called")

        # 남은 인스턴스가 있으면 정리
        if self._instances:
            try:
                # 새 이벤트 루프에서 정리 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._async_cleanup_all())
                loop.close()
            except Exception as e:
                logger.error(f"Error in atexit cleanup: {e}")

    def unregister(self, instance_id: int):
        """인스턴스 등록 해제"""
        self._instances.pop(instance_id, None)

    @classmethod
    def reset(cls):
        """레지스트리 리셋 (테스트용)"""
        if cls._instance:
            # 기존 시그널 핸들러 복원
            for sig, handler in cls._instance._original_handlers.items():
                if handler is not None:
                    signal.signal(sig, handler)
            cls._instance._original_handlers.clear()
            cls._instance._instances.clear()
            cls._instance._cleanup_tasks.clear()
        cls._instance = None


# 전역 레지스트리 인스턴스
_registry = MCPResourceRegistry()


def register_mcp_instance(instance):
    """MCP 인스턴스를 전역 레지스트리에 등록"""
    return _registry.register(instance)


def with_resource_cleanup(timeout: float = 5.0):
    """리소스 정리 타임아웃을 적용하는 데코레이터"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"{func.__name__} timed out after {timeout}s")
                raise

        return wrapper

    return decorator
