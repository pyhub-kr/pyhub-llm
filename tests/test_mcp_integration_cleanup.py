"""MCP 통합 정리 테스트"""

import asyncio
import gc
import os
import subprocess
import sys
import tempfile
import weakref
from unittest.mock import AsyncMock, patch

import pytest

from pyhub.llm import LLM
from pyhub.llm.resource_manager import MCPResourceRegistry


@pytest.mark.integration
@pytest.mark.skip(reason="Echo server implementation causes test hang - needs proper JSON-RPC protocol")
class TestMCPIntegrationCleanup:
    """MCP 통합 정리 테스트 (실제 서버 사용)"""

    @pytest.fixture
    def echo_server_script(self):
        """간단한 에코 서버 스크립트 생성"""
        script = """
import sys
import json

def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        # 간단한 JSONRPC 응답
        try:
            request = json.loads(line)
            if request.get("method") == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {"capabilities": {}}
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": None
                }
            
            output = json.dumps(response)
            sys.stdout.write(f"Content-Length: {len(output)}\\r\\n\\r\\n{output}")
            sys.stdout.flush()
        except:
            pass

if __name__ == "__main__":
    main()
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        yield script_path

        # Cleanup
        try:
            os.unlink(script_path)
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_stdio_server_lifecycle(self, echo_server_script):
        """STDIO 서버 생명주기 테스트"""
        # MCP 설정
        mcp_config = [{"type": "stdio", "name": "echo_test", "cmd": [sys.executable, echo_server_script]}]

        # 프로세스 추적
        initial_processes = self._get_python_processes()

        # LLM 생성 및 MCP 초기화
        async with await LLM.create_async("gpt-4o-mini", mcp_servers=mcp_config) as llm:
            # MCP가 연결되었는지 확인
            assert llm._mcp_connected

            # 새 프로세스가 생성되었는지 확인
            during_processes = self._get_python_processes()
            assert len(during_processes) > len(initial_processes)

        # 컨텍스트 종료 후 잠시 대기
        await asyncio.sleep(0.5)

        # 프로세스가 정리되었는지 확인
        final_processes = self._get_python_processes()
        assert len(final_processes) <= len(initial_processes) + 1  # 여유 1개

    @pytest.mark.asyncio
    async def test_multiple_servers_cleanup(self, echo_server_script):
        """여러 서버 동시 정리 테스트"""
        # 여러 MCP 서버 설정
        mcp_configs = []
        for i in range(3):
            mcp_configs.append({"type": "stdio", "name": f"echo_test_{i}", "cmd": [sys.executable, echo_server_script]})

        initial_processes = self._get_python_processes()

        # 여러 LLM 인스턴스 생성
        instances = []
        for i in range(3):
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=[mcp_configs[i]])
            instances.append(llm)

        # 모든 인스턴스가 연결되었는지 확인
        for llm in instances:
            assert llm._mcp_connected

        # 프로세스 수 확인
        during_processes = self._get_python_processes()
        assert len(during_processes) >= len(initial_processes) + 3

        # 모든 인스턴스 정리
        cleanup_tasks = [llm.close_mcp() for llm in instances]
        await asyncio.gather(*cleanup_tasks)

        # 잠시 대기
        await asyncio.sleep(0.5)

        # 프로세스가 정리되었는지 확인
        final_processes = self._get_python_processes()
        assert len(final_processes) <= len(initial_processes) + 1

    @pytest.mark.asyncio
    async def test_crash_recovery(self, echo_server_script):
        """서버 크래시 시 복구 테스트"""
        # 크래시하는 서버 스크립트
        crash_script = """
import sys
import time

# 2초 후 크래시
time.sleep(2)
sys.exit(1)
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(crash_script)
            crash_script_path = f.name

        try:
            mcp_config = [{"type": "stdio", "name": "crash_test", "cmd": [sys.executable, crash_script_path]}]

            # LLM 생성
            llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mcp_config)

            # 서버가 크래시할 때까지 대기
            await asyncio.sleep(3)

            # 정리 시도 (에러 없이 처리되어야 함)
            await llm.close_mcp()

            # 정리 확인
            assert llm._mcp_client is None
            assert not llm._mcp_connected

        finally:
            try:
                os.unlink(crash_script_path)
            except Exception:
                pass

    def _get_python_processes(self):
        """현재 실행 중인 Python 프로세스 목록"""
        try:
            # ps 명령 사용 (Unix/Linux/macOS)
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)

            # Python 프로세스 필터링
            python_processes = []
            for line in result.stdout.split("\n"):
                if "python" in line.lower():
                    python_processes.append(line)

            return python_processes
        except Exception:
            # Windows 또는 ps 명령이 없는 경우
            return []


class TestMCPCleanupScenarios:
    """다양한 정리 시나리오 테스트"""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_signal(self):
        """시그널을 통한 graceful shutdown 테스트"""
        # 이 테스트는 실제 시그널을 보내기 어려우므로 시뮬레이션
        registry = MCPResourceRegistry()

        # Mock 인스턴스 생성
        mock_instances = []
        with patch("pyhub.llm.resource_manager.register_mcp_instance"):
            for i in range(3):
                mcp_config = [{"type": "stdio", "name": f"test_{i}", "cmd": ["echo", "test"]}]

                llm = LLM.create("gpt-4o-mini", mcp_servers=mcp_config)
                mock_instances.append(llm)

                # 수동으로 레지스트리에 추가
                llm._mcp_connected = True
                llm._mcp_client = AsyncMock()
                registry._instances[id(llm)] = weakref.ref(llm)

        # 시그널 핸들러 직접 호출
        with patch("sys.exit"):
            # Cleanup 시뮬레이션
            await registry._async_cleanup_all()

        # 모든 인스턴스가 정리되었는지 확인
        for llm in mock_instances:
            # _async_cleanup_all이 close_mcp_connection을 호출했는지 확인
            if hasattr(llm, "close_mcp_connection"):
                llm._mcp_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """메모리 누수 방지 테스트"""
        import weakref

        # 인스턴스 추적
        instances_refs = []

        # 많은 인스턴스 생성 및 삭제
        for i in range(10):
            with patch("pyhub.llm.resource_manager.register_mcp_instance"):
                with patch("pyhub.llm.mcp.MultiServerMCPClient") as MockMultiServerMCPClient:
                    mock_multi_client = AsyncMock()
                    mock_multi_client._clients = {}
                    mock_multi_client._active_connections = {}
                    mock_multi_client._connection_errors = {}
                    mock_multi_client.get_tools = AsyncMock(return_value=[])
                    MockMultiServerMCPClient.return_value = mock_multi_client

                    mcp_config = [{"type": "stdio", "name": f"test_{i}", "cmd": ["echo", "test"]}]

                    llm = LLM.create("gpt-4o-mini", mcp_servers=mcp_config)
                    instances_refs.append(weakref.ref(llm))

                    # 즉시 삭제
                    del llm

        # 가비지 컬렉션
        gc.collect()
        await asyncio.sleep(0.1)

        # 모든 인스턴스가 정리되었는지 확인
        alive_count = sum(1 for ref in instances_refs if ref() is not None)
        assert alive_count == 0, f"{alive_count} instances still alive"

    @pytest.mark.asyncio
    async def test_concurrent_operations_during_cleanup(self):
        """정리 중 동시 작업 테스트"""
        # Mock 설정
        with patch("pyhub.llm.resource_manager.register_mcp_instance"):
            mcp_config = [{"type": "stdio", "name": "test_server", "cmd": ["echo", "test"]}]

            llm = LLM.create("gpt-4o-mini", mcp_servers=mcp_config)
            llm._mcp_connected = True

            # Mock 클라이언트 (정리가 오래 걸림)
            mock_client = AsyncMock()

            async def slow_exit(*args):
                await asyncio.sleep(2)

            mock_client.__aexit__ = slow_exit
            llm._mcp_client = mock_client

            # 동시에 여러 정리 시도
            cleanup_tasks = [
                llm.close_mcp(timeout=1.0),
                llm.close_mcp(timeout=1.0),
                llm.close_mcp(timeout=1.0),
            ]

            # 모두 완료될 때까지 대기
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            # 정리 완료 확인
            assert llm._mcp_client is None
            assert not llm._mcp_connected

            # 여러 번 호출해도 문제없이 처리됨
            assert all(result is None for result in results)
