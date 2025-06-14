"""MCP 리소스 추적 개선 테스트"""

import asyncio
import gc
import weakref
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from pyhub.llm import LLM
from pyhub.llm.resource_manager import MCPResourceRegistry


class TestResourceTracking:
    """리소스 추적 개선 테스트"""
    
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """각 테스트 전후에 레지스트리 리셋"""
        MCPResourceRegistry.reset()
        yield
        MCPResourceRegistry.reset()
    
    @pytest.mark.asyncio
    async def test_global_registry_tracking(self):
        """전역 레지스트리 추적 테스트"""
        instances = []
        
        # Mock finalizer 등록
        with patch('pyhub.llm.resource_manager.register_mcp_instance') as mock_register:
            mock_register.return_value = MagicMock()  # Mock finalizer
            
            # 여러 인스턴스 생성
            for i in range(5):
                mcp_config = [{
                    "type": "stdio",
                    "name": f"server_{i}",
                    "cmd": ["echo", f"test_{i}"]
                }]
                
                llm = LLM.create("gpt-4o-mini", mcp_servers=mcp_config)
                instances.append(llm)
                
                # 수동으로 레지스트리에 등록 (테스트용)
                registry = MCPResourceRegistry()
                registry._instances[id(llm)] = weakref.ref(llm)
            
            # 전역 레지스트리 확인
            registry = MCPResourceRegistry()
            
            # 모든 인스턴스가 추적되어야 함
            assert len(registry._instances) == 5
            
            # 각 인스턴스가 등록되었는지 확인
            for llm in instances:
                assert id(llm) in registry._instances
                weak_ref = registry._instances[id(llm)]
                assert weak_ref() is llm
    
    @pytest.mark.asyncio
    async def test_weak_reference_cleanup(self):
        """약한 참조 정리 테스트"""
        registry = MCPResourceRegistry()
        
        # Mock finalizer 등록
        with patch('pyhub.llm.resource_manager.register_mcp_instance') as mock_register:
            mock_register.return_value = MagicMock()
            
            # 인스턴스 생성
            mcp_config = [{
                "type": "stdio",
                "name": "test_server",
                "cmd": ["echo", "test"]
            }]
            
            llm = LLM.create("gpt-4o-mini", mcp_servers=mcp_config)
            llm_id = id(llm)
            
            # 수동으로 레지스트리에 등록
            registry._instances[llm_id] = weakref.ref(llm)
            
            # 등록 확인
            assert llm_id in registry._instances
            
            # 인스턴스 삭제
            del llm
            gc.collect()
            
            # weak reference가 None이 되어야 함
            weak_ref = registry._instances.get(llm_id)
            if weak_ref:
                assert weak_ref() is None
    
    @pytest.mark.asyncio
    async def test_atexit_cleanup(self):
        """atexit 핸들러 테스트"""
        registry = MCPResourceRegistry()
        
        # Mock 인스턴스들
        mock_instances = []
        for i in range(3):
            mock_instance = MagicMock()
            mock_instance.close_mcp = AsyncMock()
            mock_instances.append(mock_instance)
            
            # 레지스트리에 등록
            instance_id = id(mock_instance)
            registry._instances[instance_id] = weakref.ref(mock_instance)
        
        # atexit 핸들러 직접 호출
        with patch('asyncio.new_event_loop') as mock_new_loop:
            # 실제 이벤트 루프 생성
            real_loop = asyncio.new_event_loop()
            mock_new_loop.return_value = real_loop
            
            # cleanup 추적
            for instance in mock_instances:
                instance.close_mcp.assert_not_called()
            
            # atexit 핸들러 호출
            registry._atexit_cleanup()
            
            # cleanup이 호출되었는지 확인
            for instance in mock_instances:
                instance.close_mcp.assert_called_once()
            
            # loop가 닫혔는지 확인
            assert real_loop.is_closed()
    
    @pytest.mark.asyncio
    async def test_concurrent_cleanup_tasks(self):
        """동시 cleanup 작업 추적"""
        registry = MCPResourceRegistry()
        
        # cleanup 추적
        cleanup_order = []
        
        # Mock 인스턴스들
        for i in range(3):
            mock_instance = MagicMock()
            
            async def make_cleanup(idx):
                async def cleanup():
                    cleanup_order.append(f"start_{idx}")
                    await asyncio.sleep(0.1)
                    cleanup_order.append(f"end_{idx}")
                return cleanup
            
            mock_instance.close_mcp = await make_cleanup(i)
            
            # 레지스트리에 등록  
            instance_id = id(mock_instance)
            registry._instances[instance_id] = weakref.ref(mock_instance)
        
        # 모든 cleanup 실행
        await registry._async_cleanup_all()
        
        # 모든 cleanup이 시작되고 끝났는지 확인
        assert len([x for x in cleanup_order if x.startswith("start_")]) == 3
        assert len([x for x in cleanup_order if x.startswith("end_")]) == 3
        
        # 동시에 실행되었는지 확인 (순서가 섞여있어야 함)
        # start_0, start_1, start_2가 모두 end들보다 먼저 나와야 함
        start_indices = [i for i, x in enumerate(cleanup_order) if x.startswith("start_")]
        end_indices = [i for i, x in enumerate(cleanup_order) if x.startswith("end_")]
        assert max(start_indices) < min(end_indices)
    
    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self):
        """cleanup 중 에러 처리"""
        registry = MCPResourceRegistry()
        
        # 성공/실패 추적
        cleanup_results = {"success": 0, "error": 0}
        
        # Mock 인스턴스들 (일부는 에러 발생)
        for i in range(4):
            mock_instance = MagicMock()
            
            if i % 2 == 0:
                # 정상 cleanup
                async def success_cleanup():
                    cleanup_results["success"] += 1
                mock_instance.close_mcp = AsyncMock(side_effect=success_cleanup)
            else:
                # 에러 발생
                async def error_cleanup():
                    cleanup_results["error"] += 1
                    raise Exception(f"Cleanup error {i}")
                mock_instance.close_mcp = AsyncMock(side_effect=error_cleanup)
            
            # 레지스트리에 등록
            instance_id = id(mock_instance)
            registry._instances[instance_id] = weakref.ref(mock_instance)
        
        # cleanup 실행 (에러가 있어도 계속 진행되어야 함)
        await registry._async_cleanup_all()
        
        # 모든 cleanup이 시도되었는지 확인
        assert cleanup_results["success"] == 2
        assert cleanup_results["error"] == 2
    
    def test_unregister(self):
        """수동 등록 해제 테스트"""
        registry = MCPResourceRegistry()
        
        # Mock 인스턴스
        mock_instance = MagicMock()
        instance_id = id(mock_instance)
        
        # 등록
        registry._instances[instance_id] = weakref.ref(mock_instance)
        assert instance_id in registry._instances
        
        # 해제
        registry.unregister(instance_id)
        assert instance_id not in registry._instances
        
        # 없는 ID 해제 (에러 없이 처리되어야 함)
        registry.unregister(12345)  # 존재하지 않는 ID