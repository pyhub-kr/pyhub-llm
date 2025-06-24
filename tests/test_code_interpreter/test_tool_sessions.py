"""Test session isolation in CodeInterpreter tool."""

import pytest
from pyhub.llm.agents.tools import CodeInterpreter


class TestCodeInterpreterSessions:
    """Test that different sessions are properly isolated."""
    
    @pytest.fixture
    def tool(self):
        """Create CodeInterpreter instance."""
        return CodeInterpreter(backend="local")
    
    def test_session_isolation(self, tool):
        """Test that different sessions don't share state."""
        # Session 1: Define variables
        result1_s1 = tool.run("x = 100\ny = 'session1'", session_id="session1")
        
        # Session 2: Define different variables
        result2_s2 = tool.run("x = 200\ny = 'session2'", session_id="session2")
        
        # Session 1: Check variables unchanged
        result3_s1 = tool.run("print(f'Session 1: x={x}, y={y}')", session_id="session1")
        assert "Session 1: x=100, y=session1" in result3_s1
        
        # Session 2: Check variables are different
        result4_s2 = tool.run("print(f'Session 2: x={x}, y={y}')", session_id="session2")
        assert "Session 2: x=200, y=session2" in result4_s2
    
    def test_no_session_id_isolation(self, tool):
        """Test that runs without session_id don't share state."""
        # First run without session_id
        result1 = tool.run("isolated_var = 'first'")
        
        # Second run without session_id should not see the variable
        result2 = tool.run("print(isolated_var)")
        assert "Error" in result2 or "NameError" in result2
    
    def test_multiple_sessions_parallel_work(self, tool):
        """Test multiple sessions working on different tasks."""
        sessions = {
            "math_session": "import math\nresult = math.sqrt(16)",
            "string_session": "text = 'hello'\nresult = text.upper()",
            "list_session": "nums = [1, 2, 3]\nresult = sum(nums)"
        }
        
        # Initialize all sessions
        for session_id, code in sessions.items():
            tool.run(code, session_id=session_id)
        
        # Check results in each session
        results = {}
        results['math'] = tool.run("print(f'Math result: {result}')", session_id="math_session")
        results['string'] = tool.run("print(f'String result: {result}')", session_id="string_session")
        results['list'] = tool.run("print(f'List result: {result}')", session_id="list_session")
        
        assert "Math result: 4.0" in results['math']
        assert "String result: HELLO" in results['string']
        assert "List result: 6" in results['list']
    
    def test_session_with_imports(self, tool):
        """Test that imports are isolated between sessions."""
        # Session 1: Import json
        tool.run("import json", session_id="json_session")
        result1 = tool.run("print(json.dumps({'key': 'value'}))", session_id="json_session")
        assert '"key": "value"' in result1
        
        # Session 2: json should not be available
        result2 = tool.run("print(json.dumps({'test': 'data'}))", session_id="other_session")
        assert "Error" in result2 or "NameError" in result2
    
    def test_session_cleanup(self, tool):
        """Test session cleanup behavior."""
        session_id = "cleanup_test"
        
        # Create data in session
        tool.run("large_list = list(range(1000))", session_id=session_id)
        result1 = tool.run("print(f'List length: {len(large_list)}')", session_id=session_id)
        assert "List length: 1000" in result1
        
        # Clear session (simulate cleanup)
        tool.clear_session(session_id)
        
        # Variable should no longer exist
        result2 = tool.run("print(large_list)", session_id=session_id)
        assert "Error" in result2 or "NameError" in result2
    
    def test_concurrent_pandas_sessions(self, tool):
        """Test pandas operations in different sessions."""
        # Session 1: Create one DataFrame
        tool.run("""
import pandas as pd
df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
""", session_id="pandas1")
        
        # Session 2: Create different DataFrame
        tool.run("""
import pandas as pd
df1 = pd.DataFrame({
    'X': [10, 20, 30],
    'Y': [40, 50, 60]
})
""", session_id="pandas2")
        
        # Check Session 1 has correct columns
        result1 = tool.run("print(df1.columns.tolist())", session_id="pandas1")
        assert "['A', 'B']" in result1
        
        # Check Session 2 has different columns
        result2 = tool.run("print(df1.columns.tolist())", session_id="pandas2")
        assert "['X', 'Y']" in result2
    
    def test_matplotlib_session_isolation(self, tool):
        """Test matplotlib figures are isolated between sessions."""
        # Session 1: Create a plot
        tool.run("""
import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots()
ax1.plot([1, 2, 3], [1, 4, 9])
ax1.set_title('Session 1 Plot')
""", session_id="plot1")
        
        # Session 2: Create different plot
        tool.run("""
import matplotlib.pyplot as plt
fig2, ax2 = plt.subplots()
ax2.plot([1, 2, 3], [3, 2, 1])
ax2.set_title('Session 2 Plot')
""", session_id="plot2")
        
        # Save plots from each session
        result1 = tool.run("plt.savefig('session1_plot.png')\nprint('Saved session 1')", 
                          session_id="plot1")
        assert "Saved session 1" in result1
        
        result2 = tool.run("plt.savefig('session2_plot.png')\nprint('Saved session 2')", 
                          session_id="plot2")
        assert "Saved session 2" in result2
    
    def test_session_state_persistence(self, tool):
        """Test that session state persists across multiple runs."""
        session_id = "persistent_state"
        
        # Run 1: Initialize counter
        tool.run("""
class StateCounter:
    def __init__(self):
        self.count = 0
        self.history = []
    
    def increment(self, amount=1):
        self.count += amount
        self.history.append(self.count)
        return self.count

counter = StateCounter()
""", session_id=session_id)
        
        # Run 2-5: Increment counter
        for i in range(1, 5):
            result = tool.run(f"""
new_count = counter.increment({i})
print(f'Run {i}: count = {{new_count}}, history = {{counter.history}}')
""", session_id=session_id)
            assert f"count = {sum(range(1, i+1))}" in result
        
        # Final check
        result_final = tool.run("""
print(f'Final count: {counter.count}')
print(f'History length: {len(counter.history)}')
print(f'Total increments: {sum(counter.history)}')
""", session_id=session_id)
        assert "Final count: 10" in result_final  # 1+2+3+4
        assert "History length: 4" in result_final
    
    def test_different_backends_same_session(self, tool):
        """Test that same session_id with different tool instances are isolated."""
        # Create two separate tool instances
        tool1 = CodeInterpreter(backend="local")
        tool2 = CodeInterpreter(backend="local")
        
        session_id = "shared_session_id"
        
        # Set different values in each tool instance
        tool1.run("shared_var = 'tool1'", session_id=session_id)
        tool2.run("shared_var = 'tool2'", session_id=session_id)
        
        # Check values remain separate
        result1 = tool1.run("print(f'Tool1: {shared_var}')", session_id=session_id)
        result2 = tool2.run("print(f'Tool2: {shared_var}')", session_id=session_id)
        
        assert "Tool1: tool1" in result1
        assert "Tool2: tool2" in result2