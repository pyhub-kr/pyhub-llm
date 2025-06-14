"""실제 시나리오 테스트"""

import asyncio
import os
import sys
import time
from pyhub.llm import LLM
from pyhub.llm.resource_manager import MCPResourceRegistry

async def test_basic_usage():
    """기본 사용 시나리오"""
    print("=== Test 1: Basic Usage ===")
    
    # MCP 없는 LLM
    llm = LLM.create("gpt-4o-mini")
    print(f"LLM created without MCP")
    print(f"Has finalizer: {llm._finalizer is not None}")
    print(f"Registry instances: {len(MCPResourceRegistry()._instances)}")
    
    # 삭제
    del llm
    print("LLM deleted\n")

async def test_mcp_with_error():
    """MCP 연결 실패 시나리오"""
    print("=== Test 2: MCP Connection Error ===")
    
    # 잘못된 MCP 설정
    mcp_config = [{
        "type": "stdio",
        "name": "test_server",
        "cmd": ["nonexistent_command"]
    }]
    
    try:
        llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mcp_config)
        print(f"LLM created with MCP (may have connection error)")
        print(f"MCP connected: {llm._mcp_connected}")
        print(f"Has finalizer: {llm._finalizer is not None}")
        print(f"Registry instances: {len(MCPResourceRegistry()._instances)}")
        
        # 정리
        await llm.close_mcp()
        print("MCP closed")
    except Exception as e:
        print(f"Error: {e}")
    print()

async def test_timeout_behavior():
    """타임아웃 동작 테스트"""
    print("=== Test 3: Timeout Behavior ===")
    
    # Mock slow cleanup
    class SlowCleanupLLM:
        def __init__(self):
            self._mcp_client = self
            self._mcp_connected = True
            self._mcp_tools = []
            
        async def __aexit__(self, *args):
            print("Starting slow cleanup...")
            await asyncio.sleep(10)  # 10초 대기
            print("Slow cleanup done")
    
    llm = SlowCleanupLLM()
    
    # 타임아웃 테스트
    start = time.time()
    try:
        from pyhub.llm.base import BaseLLM
        # close_mcp 메서드 직접 호출
        await BaseLLM.close_mcp(llm, timeout=2.0)
    except Exception as e:
        print(f"Error: {e}")
    
    elapsed = time.time() - start
    print(f"Cleanup took {elapsed:.2f} seconds (expected ~2s)")
    print()

async def test_registry_behavior():
    """레지스트리 동작 테스트"""
    print("=== Test 4: Registry Behavior ===")
    
    registry = MCPResourceRegistry()
    print(f"Initial instances: {len(registry._instances)}")
    
    # Mock 인스턴스 등록
    class MockLLM:
        async def close_mcp(self):
            print("Mock cleanup called")
    
    mock = MockLLM()
    weak_ref = weakref.ref(mock)
    registry._instances[id(mock)] = weak_ref
    
    print(f"After registration: {len(registry._instances)}")
    
    # 정리
    await registry._cleanup_instance(id(mock))
    
    # 인스턴스 삭제
    del mock
    print(f"Weak ref alive: {weak_ref() is not None}")
    print()

async def test_signal_handler():
    """시그널 핸들러 테스트"""
    print("=== Test 5: Signal Handler ===")
    
    registry = MCPResourceRegistry()
    
    # 테스트 환경 확인
    is_pytest = os.environ.get('PYTEST_CURRENT_TEST')
    print(f"Running in pytest: {is_pytest is not None}")
    print(f"Signal handlers registered: {len(registry._original_handlers)}")
    
    # 수동 활성화
    if is_pytest:
        registry.enable_signal_handlers()
        print(f"After manual enable: {len(registry._original_handlers)}")
    
    print()

import weakref

async def main():
    """모든 테스트 실행"""
    
    await test_basic_usage()
    await test_mcp_with_error()
    await test_timeout_behavior()
    await test_registry_behavior()
    await test_signal_handler()
    
    print("=== Final State ===")
    registry = MCPResourceRegistry()
    print(f"Total instances in registry: {len(registry._instances)}")
    print(f"Active instances: {sum(1 for ref in registry._instances.values() if ref() is not None)}")

if __name__ == "__main__":
    asyncio.run(main())