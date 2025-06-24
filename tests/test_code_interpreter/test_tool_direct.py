"""Direct testing of CodeInterpreter tool without LLM."""

import pytest
from pyhub.llm.agents.tools import CodeInterpreter


class TestCodeInterpreterDirect:
    """Test CodeInterpreter tool directly without LLM."""
    
    @pytest.fixture
    def tool(self):
        """Create CodeInterpreter instance."""
        return CodeInterpreter(backend="local")
    
    def test_simple_print(self, tool):
        """Test simple print statement."""
        result = tool.run("print('Hello, World!')")
        assert "Hello, World!" in result
        assert "Output:" in result
        assert "Execution time:" in result
    
    def test_variable_assignment(self, tool):
        """Test variable assignment and calculation."""
        code = """
x = 10
y = 20
result = x + y
print(f"Result: {result}")
"""
        result = tool.run(code)
        assert "Result: 30" in result
    
    def test_multi_line_output(self, tool):
        """Test multiple print statements."""
        code = """
print("Line 1")
print("Line 2")
print("Line 3")
"""
        result = tool.run(code)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
    
    def test_math_operations(self, tool):
        """Test various math operations."""
        code = """
import math

# Basic operations
print(f"Addition: {5 + 3}")
print(f"Multiplication: {5 * 3}")
print(f"Division: {15 / 3}")
print(f"Power: {2 ** 8}")

# Math functions
print(f"Square root of 16: {math.sqrt(16)}")
print(f"Sin of pi/2: {math.sin(math.pi/2)}")
"""
        result = tool.run(code)
        assert "Addition: 8" in result
        assert "Multiplication: 15" in result
        assert "Division: 5.0" in result
        assert "Power: 256" in result
        assert "Square root of 16: 4.0" in result
        assert "Sin of pi/2: 1.0" in result
    
    def test_list_comprehension(self, tool):
        """Test list comprehension and operations."""
        code = """
# List comprehension
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")
print(f"Sum: {sum(squares)}")
print(f"Max: {max(squares)}")
"""
        result = tool.run(code)
        assert "[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]" in result
        assert "Sum: 285" in result
        assert "Max: 81" in result
    
    def test_function_definition(self, tool):
        """Test function definition and call."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(f"5! = {factorial(5)}")
print(f"10! = {factorial(10)}")
"""
        result = tool.run(code)
        assert "5! = 120" in result
        assert "10! = 3628800" in result