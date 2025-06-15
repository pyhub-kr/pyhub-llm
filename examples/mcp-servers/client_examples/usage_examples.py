#!/usr/bin/env python3
"""
MCP Client Usage Examples with pyhub-llm

This script shows practical examples of using MCP servers
with pyhub-llm for real AI applications.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add pyhub-llm to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from pyhub.llm import LLMFactory
from pyhub.llm.mcp import MCPClient, load_mcp_tools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def example_1_basic_calculator():
    """Example 1: Basic calculator with AI assistant."""
    logger.info("=== Example 1: AI Calculator Assistant ===")

    try:
        # Configure STDIO MCP server
        config = {
            "command": "python3",
            "args": [str(Path(__file__).parent.parent / "stdio" / "calculator_server.py")]
        }

        # Load calculator tools
        tools = await load_mcp_tools(config)
        logger.info(f"Loaded tools: {[tool.name for tool in tools]}")

        # Create LLM with tools
        llm = LLMFactory.create("mock", tools=tools)

        # Test math questions
        questions = [
            "What is 25 + 17?",
            "Calculate 8 * 7",
            "Divide 144 by 12",
            "What's 15 - 8?",
            "Evaluate the expression: (10 + 5) * 2"
        ]

        for question in questions:
            try:
                response = await llm.ask_async(question)
                logger.info(f"Q: {question}")
                logger.info(f"A: {response.text}\n")
            except Exception as e:
                logger.error(f"Error with question '{question}': {e}")

    except Exception as e:
        logger.error(f"Example 1 failed: {e}")


async def example_2_multi_server_setup():
    """Example 2: Using multiple MCP servers simultaneously."""
    logger.info("=== Example 2: Multi-Server Setup ===")

    servers = [
        {
            "name": "Calculator",
            "config": {
                "command": "python3",
                "args": [str(Path(__file__).parent.parent / "stdio" / "calculator_server.py")]
            }
        }
        # Note: Add more servers when available
        # {
        #     "name": "WebSearch",
        #     "config": {"url": "http://localhost:8004/mcp/sse"}
        # }
    ]

    all_tools = []

    for server in servers:
        try:
            logger.info(f"Loading tools from {server['name']} server...")
            tools = await load_mcp_tools(server['config'])
            all_tools.extend(tools)
            logger.info(f"Added {len(tools)} tools from {server['name']}")
        except Exception as e:
            logger.warning(f"Failed to load {server['name']} server: {e}")

    if all_tools:
        logger.info(f"Total tools available: {len(all_tools)}")

        # Create AI assistant with all tools
        llm = LLMFactory.create("mock", tools=all_tools)

        # Complex query requiring multiple tools
        query = "Calculate the result of 15 * 8, then add 23 to it"

        try:
            response = await llm.ask_async(query)
            logger.info(f"Multi-tool query: {query}")
            logger.info(f"Result: {response.text}")
        except Exception as e:
            logger.error(f"Multi-tool query failed: {e}")


async def example_3_error_handling():
    """Example 3: Robust error handling with MCP."""
    logger.info("=== Example 3: Error Handling ===")

    # Test error scenarios
    error_configs = [
        {
            "name": "Invalid command",
            "config": {"command": "nonexistent_command"}
        },
        {
            "name": "Invalid URL",
            "config": {"url": "http://localhost:9999/invalid"}
        },
        {
            "name": "Missing required field",
            "config": {"transport": "stdio"}  # Missing command
        }
    ]

    for test_case in error_configs:
        try:
            logger.info(f"Testing: {test_case['name']}")
            client = MCPClient(test_case['config'])

            # Try to connect (will likely fail)
            async with client.connect() as conn:
                tools = await conn.list_tools()
                logger.info(f"Unexpected success: {len(tools)} tools found")

        except Exception as e:
            logger.info(f"Expected error: {type(e).__name__}: {e}")

    # Test tool execution errors
    try:
        config = {
            "command": "python3",
            "args": [str(Path(__file__).parent.parent / "stdio" / "calculator_server.py")]
        }

        async with MCPClient(config).connect() as client:
            # Test division by zero
            result = await client.execute_tool("divide", {"a": 10, "b": 0})
            logger.info(f"Division by zero result: {result}")

            # Test invalid expression
            result = await client.execute_tool("calculate_expression", {"expression": "import os"})
            logger.info(f"Invalid expression result: {result}")

    except Exception as e:
        logger.error(f"Tool error testing failed: {e}")


async def example_4_advanced_ai_assistant():
    """Example 4: Advanced AI assistant with conversation context."""
    logger.info("=== Example 4: Advanced AI Assistant ===")

    try:
        # Load calculator tools
        config = {
            "command": "python3",
            "args": [str(Path(__file__).parent.parent / "stdio" / "calculator_server.py")]
        }

        tools = await load_mcp_tools(config)

        # Create LLM with history enabled
        llm = LLMFactory.create(
            "mock",
            tools=tools,
            system_prompt="You are a helpful math tutor. Use the calculator tools to solve problems step by step."
        )

        # Conversation with context
        conversation = [
            "I need to calculate my monthly budget. I earn $3000 per month.",
            "If I spend $800 on rent, how much do I have left?",
            "From the remaining amount, I want to save 20%. How much is that?",
            "How much will I have for other expenses?"
        ]

        for message in conversation:
            try:
                response = await llm.ask_async(message, use_history=True)
                logger.info(f"User: {message}")
                logger.info(f"Assistant: {response.text}\n")
            except Exception as e:
                logger.error(f"Conversation error: {e}")

    except Exception as e:
        logger.error(f"Example 4 failed: {e}")


async def example_5_performance_testing():
    """Example 5: Performance testing with concurrent requests."""
    logger.info("=== Example 5: Performance Testing ===")

    try:
        config = {
            "command": "python3",
            "args": [str(Path(__file__).parent.parent / "stdio" / "calculator_server.py")]
        }

        # Test concurrent tool calls
        async def test_calculation(client, operation, a, b):
            result = await client.execute_tool(operation, {"a": a, "b": b})
            return f"{operation}({a}, {b}) = {result}"

        async with MCPClient(config).connect() as client:
            # Prepare concurrent tasks
            tasks = [
                test_calculation(client, "add", 10, 5),
                test_calculation(client, "multiply", 7, 8),
                test_calculation(client, "subtract", 20, 13),
                test_calculation(client, "divide", 56, 7),
                test_calculation(client, "add", 100, 200),
            ]

            # Run concurrently
            import time
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            logger.info(f"Completed {len(tasks)} operations in {end_time - start_time:.2f} seconds")

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {result}")
                else:
                    logger.info(f"Task {i}: {result}")

    except Exception as e:
        logger.error(f"Performance testing failed: {e}")


async def main():
    """Run all examples."""
    examples = [
        example_1_basic_calculator,
        example_2_multi_server_setup,
        example_3_error_handling,
        example_4_advanced_ai_assistant,
        example_5_performance_testing
    ]

    for i, example in enumerate(examples, 1):
        try:
            await example()
            logger.info(f"✅ Example {i} completed")
        except Exception as e:
            logger.error(f"❌ Example {i} failed: {e}")

        logger.info("-" * 50)

    logger.info("🎉 All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())