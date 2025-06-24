"""Test context preservation across multiple CodeInterpreter runs."""

import pytest
from pyhub.llm.agents.tools import CodeInterpreter


class TestCodeInterpreterContext:
    """Test that context is maintained across multiple runs."""
    
    @pytest.fixture
    def tool(self):
        """Create CodeInterpreter instance."""
        return CodeInterpreter(backend="local")
    
    def test_variable_persistence(self, tool):
        """Test that variables persist across runs."""
        session_id = "var_persist"
        
        # Run 1: Define variables
        result1 = tool.run("x = 42\ny = 'hello'", session_id=session_id)
        
        # Run 2: Use variables
        result2 = tool.run("print(f'x = {x}, y = {y}')", session_id=session_id)
        assert "x = 42, y = hello" in result2
        
        # Run 3: Modify and use
        result3 = tool.run("x = x * 2\nprint(f'x doubled = {x}')", session_id=session_id)
        assert "x doubled = 84" in result3
    
    def test_import_persistence(self, tool):
        """Test that imports persist across runs."""
        session_id = "import_persist"
        
        # Run 1: Import libraries
        result1 = tool.run("""
import math
import random
import json
print("Libraries imported")
""", session_id=session_id)
        assert "Libraries imported" in result1
        
        # Run 2: Use imported libraries without re-importing
        result2 = tool.run("""
print(f"Pi: {math.pi}")
print(f"Random: {random.randint(1, 10)}")
data = json.dumps({"key": "value"})
print(f"JSON: {data}")
""", session_id=session_id)
        assert "Pi: 3.14159" in result2
        assert "Random:" in result2
        assert '"key": "value"' in result2
    
    def test_function_persistence(self, tool):
        """Test that functions persist across runs."""
        session_id = "func_persist"
        
        # Run 1: Define functions
        result1 = tool.run("""
def greet(name):
    return f"Hello, {name}!"

def calculate(a, b):
    return a * b + 10

print("Functions defined")
""", session_id=session_id)
        assert "Functions defined" in result1
        
        # Run 2: Use functions
        result2 = tool.run("""
print(greet("Alice"))
print(greet("Bob"))
print(f"Calculate: {calculate(5, 3)}")
""", session_id=session_id)
        assert "Hello, Alice!" in result2
        assert "Hello, Bob!" in result2
        assert "Calculate: 25" in result2
    
    def test_class_persistence(self, tool):
        """Test that classes persist across runs."""
        session_id = "class_persist"
        
        # Run 1: Define class
        result1 = tool.run("""
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
        return self.count
    
    def get_count(self):
        return self.count

counter = Counter()
print("Counter created")
""", session_id=session_id)
        assert "Counter created" in result1
        
        # Run 2: Use class instance
        result2 = tool.run("""
print(f"Initial: {counter.get_count()}")
counter.increment()
counter.increment()
print(f"After 2 increments: {counter.get_count()}")
""", session_id=session_id)
        assert "Initial: 0" in result2
        assert "After 2 increments: 2" in result2
        
        # Run 3: Continue using instance
        result3 = tool.run("""
for i in range(3):
    counter.increment()
print(f"Final count: {counter.get_count()}")
""", session_id=session_id)
        assert "Final count: 5" in result3
    
    def test_data_accumulation(self, tool):
        """Test accumulating data across runs."""
        session_id = "data_accum"
        
        # Run 1: Initialize list
        result1 = tool.run("""
data = []
print("Data list initialized")
""", session_id=session_id)
        
        # Run 2-4: Add data in multiple runs
        for i in range(3):
            code = f"""
data.append({i * 10})
print(f"Added {{data[-1]}}, list is now: {{data}}")
"""
            result = tool.run(code, session_id=session_id)
            assert f"[{', '.join(str(j*10) for j in range(i+1))}]" in result
        
        # Run 5: Process accumulated data
        result5 = tool.run("""
print(f"Total items: {len(data)}")
print(f"Sum: {sum(data)}")
print(f"Average: {sum(data) / len(data)}")
""", session_id=session_id)
        assert "Total items: 3" in result5
        assert "Sum: 30" in result5
        assert "Average: 10.0" in result5
    
    def test_pandas_dataframe_evolution(self, tool):
        """Test building a DataFrame step by step."""
        session_id = "df_evolution"
        
        # Run 1: Create initial DataFrame
        result1 = tool.run("""
import pandas as pd
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
print("Initial DataFrame:")
print(df)
""", session_id=session_id)
        assert "Initial DataFrame:" in result1
        assert "A" in result1 and "B" in result1
        
        # Run 2: Add new column
        result2 = tool.run("""
df['C'] = df['A'] + df['B']
print("After adding column C:")
print(df)
""", session_id=session_id)
        assert "C" in result2
        assert "5" in result2  # 1+4
        assert "7" in result2  # 2+5
        assert "9" in result2  # 3+6
        
        # Run 3: Add row
        result3 = tool.run("""
new_row = pd.DataFrame({'A': [4], 'B': [7], 'C': [11]})
df = pd.concat([df, new_row], ignore_index=True)
print("After adding row:")
print(df)
print(f"Shape: {df.shape}")
""", session_id=session_id)
        assert "Shape: (4, 3)" in result3
        assert "11" in result3
    
    def test_matplotlib_figure_persistence(self, tool):
        """Test that matplotlib figures can be built incrementally."""
        session_id = "fig_persist"
        
        # Run 1: Create figure
        result1 = tool.run("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
print("Figure created")
""", session_id=session_id)
        assert "Figure created" in result1
        
        # Run 2: Add first plot
        result2 = tool.run("""
ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label='Series 1')
ax.set_xlabel('X axis')
print("First series added")
""", session_id=session_id)
        assert "First series added" in result2
        
        # Run 3: Add second plot
        result3 = tool.run("""
ax.plot([1, 2, 3, 4], [2, 3, 4, 1], label='Series 2', linestyle='--')
ax.set_ylabel('Y axis')
ax.set_title('Multi-step Plot')
ax.legend()
print("Second series added")
""", session_id=session_id)
        assert "Second series added" in result3
        
        # Run 4: Save figure
        result4 = tool.run("""
plt.savefig('incremental_plot.png')
print("Plot saved")
""", session_id=session_id)
        assert "Plot saved" in result4
        assert "incremental_plot.png" in result4
    
    def test_error_recovery_context(self, tool):
        """Test that context is preserved after errors."""
        session_id = "error_recovery"
        
        # Run 1: Set up variables
        result1 = tool.run("""
x = 100
y = 200
data = [1, 2, 3, 4, 5]
print("Variables set")
""", session_id=session_id)
        assert "Variables set" in result1
        
        # Run 2: Cause an error
        result2 = tool.run("z = 1 / 0", session_id=session_id)
        assert "Error" in result2 or "ZeroDivisionError" in result2
        
        # Run 3: Verify context still exists
        result3 = tool.run("""
print(f"x = {x}")
print(f"y = {y}")
print(f"data = {data}")
print(f"Sum of data: {sum(data)}")
""", session_id=session_id)
        assert "x = 100" in result3
        assert "y = 200" in result3
        assert "data = [1, 2, 3, 4, 5]" in result3
        assert "Sum of data: 15" in result3