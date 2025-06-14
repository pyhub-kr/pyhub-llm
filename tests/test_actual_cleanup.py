"""실제 cleanup 동작 확인"""

import asyncio
from pyhub.llm import LLM
from pyhub.llm.resource_manager import MCPResourceRegistry


async def test_actual_cleanup():
    """실제 cleanup 동작 테스트"""
    print("=== Actual Cleanup Test ===")
    
    # MCP 없는 LLM
    llm = LLM.create("gpt-4o-mini")
    print(f"LLM without MCP created")
    print(f"Has MCP client: {llm._mcp_client is not None}")
    print(f"MCP connected: {llm._mcp_connected}")
    
    # Registry 확인
    registry = MCPResourceRegistry()
    print(f"Registry instances: {len(registry._instances)}")
    
    # MCP 있는 LLM (실패할 것)
    mcp_config = [{
        "type": "stdio",
        "name": "test_server",
        "cmd": ["nonexistent_command"]
    }]
    
    try:
        llm_with_mcp = await LLM.create_async("gpt-4o-mini", mcp_servers=mcp_config)
        print(f"\nLLM with MCP created")
        print(f"Has MCP client: {llm_with_mcp._mcp_client is not None}")
        print(f"MCP connected: {llm_with_mcp._mcp_connected}")
        print(f"Has finalizer: {llm_with_mcp._finalizer is not None}")
        
        # 명시적 cleanup
        await llm_with_mcp.close_mcp()
        print(f"\nAfter close_mcp:")
        print(f"Has MCP client: {llm_with_mcp._mcp_client is not None}")
        print(f"MCP connected: {llm_with_mcp._mcp_connected}")
        
    except Exception as e:
        print(f"\nError: {e}")
    
    print(f"\nFinal registry instances: {len(registry._instances)}")


if __name__ == "__main__":
    asyncio.run(test_actual_cleanup())