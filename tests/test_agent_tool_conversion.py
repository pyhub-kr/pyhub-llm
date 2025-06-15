"""Test automatic tool conversion in agents"""

from unittest.mock import Mock

from pyhub.llm.agents import ReactAgent
from pyhub.llm.agents.base import Tool
from pyhub.llm.agents.tools import Calculator


def test_agent_accepts_tool_instances():
    """Test that agent accepts Tool instances"""
    mock_llm = Mock()

    tool = Tool(name="test_tool", description="Test tool", func=lambda x: f"Result: {x}")

    agent = ReactAgent(llm=mock_llm, tools=[tool])

    assert len(agent.tools) == 1
    assert agent.tools[0].name == "test_tool"
    assert isinstance(agent.tools[0], Tool)


def test_agent_converts_functions_to_tools():
    """Test that agent converts regular functions to Tool objects"""
    mock_llm = Mock()

    def calculate(x: int, y: int) -> int:
        """Add two numbers"""
        return x + y

    def get_weather(city: str) -> str:
        """Get weather for a city"""
        return f"Weather in {city}: Sunny"

    agent = ReactAgent(llm=mock_llm, tools=[calculate, get_weather])

    assert len(agent.tools) == 2
    assert all(isinstance(tool, Tool) for tool in agent.tools)

    # Check tool names (function names)
    tool_names = [tool.name for tool in agent.tools]
    assert "calculate" in tool_names
    assert "get_weather" in tool_names

    # Check tool descriptions (from docstrings)
    calculate_tool = next(t for t in agent.tools if t.name == "calculate")
    assert "Add two numbers" in calculate_tool.description


def test_agent_converts_mixed_tools():
    """Test that agent handles mixed tool types"""
    mock_llm = Mock()

    # Mix of different tool types
    def simple_func(x: str) -> str:
        """Simple function"""
        return x.upper()

    tool_instance = Tool(name="explicit_tool", description="Explicit tool", func=lambda: "explicit")

    calculator = Calculator()

    agent = ReactAgent(llm=mock_llm, tools=[simple_func, tool_instance, calculator])

    assert len(agent.tools) == 3
    assert all(isinstance(tool, Tool) for tool in agent.tools)

    # Verify all tools are accessible
    assert agent.get_tool("simple_func") is not None
    assert agent.get_tool("explicit_tool") is not None
    assert agent.get_tool("calculator") is not None


def test_agent_tool_execution_after_conversion():
    """Test that converted tools can be executed"""
    mock_llm = Mock()

    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    agent = ReactAgent(llm=mock_llm, tools=[add])

    # Get the converted tool
    add_tool = agent.get_tool("add")
    assert add_tool is not None

    # Execute the tool
    result = add_tool.func(a=5, b=3)
    assert result == 8


def test_agent_with_callable_classes():
    """Test that agent handles callable classes"""
    mock_llm = Mock()

    class WeatherService:
        """Weather service tool"""

        def __call__(self, city: str) -> str:
            return f"Weather in {city}: Rainy"

    weather_service = WeatherService()

    agent = ReactAgent(llm=mock_llm, tools=[weather_service])

    assert len(agent.tools) == 1
    assert agent.tools[0].name == "WeatherService"
    assert "Weather service tool" in agent.tools[0].description

    # Test execution
    result = agent.tools[0].func(city="Seoul")
    assert result == "Weather in Seoul: Rainy"


def test_agent_empty_tools_list():
    """Test agent with empty tools list"""
    mock_llm = Mock()

    agent = ReactAgent(llm=mock_llm, tools=[])

    assert len(agent.tools) == 0
    assert len(agent._tool_map) == 0
