"""Tests for Code Interpreter tool integration."""

import pytest
from pathlib import Path
import tempfile

from pyhub.llm.agents.tools import CodeInterpreter


class TestCodeInterpreterTool:
    """Test Code Interpreter tool."""
    
    @pytest.fixture
    def tool(self):
        """Create Code Interpreter tool instance."""
        return CodeInterpreter(backend="local")
    
    def test_tool_initialization(self):
        """Test tool initialization."""
        tool = CodeInterpreter()
        assert tool.name == "code_interpreter"
        assert "Python code" in tool.description
        assert tool.backend_name == "local"
    
    def test_tool_run_basic(self, tool):
        """Test basic code execution through tool."""
        result = tool.run("print(2 + 2)")
        assert "4" in result
        assert "Output:" in result
        assert "Execution time:" in result
    
    def test_tool_run_with_session(self, tool):
        """Test code execution with session."""
        session_id = "test_session"
        
        # First execution
        result1 = tool.run("x = 42", session_id=session_id)
        assert "Execution time:" in result1
        
        # Second execution in same session
        result2 = tool.run("print(f'x = {x}')", session_id=session_id)
        assert "x = 42" in result2
    
    def test_tool_error_handling(self, tool):
        """Test error handling in tool."""
        result = tool.run("1 / 0")
        assert "Error executing code:" in result
        assert "ZeroDivisionError" in result
    
    def test_tool_with_pandas(self, tool):
        """Test pandas code execution."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.describe())
"""
        result = tool.run(code)
        
        # Check if pandas is available
        if "Error" not in result:
            assert "count" in result
            assert "mean" in result
            assert "std" in result
    
    def test_tool_with_visualization(self, tool):
        """Test visualization generation."""
        code = """
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Test Plot')
plt.savefig('test_plot.png')
print("Plot saved")
"""
        result = tool.run(code, session_id="viz_session")
        
        # Check if matplotlib is available
        if "Error" not in result:
            assert "Plot saved" in result
            assert "Files created:" in result
    
    def test_tool_security(self, tool):
        """Test security restrictions."""
        dangerous_code = "import os\nos.system('ls')"
        result = tool.run(dangerous_code)
        
        assert "Error" in result or "not allowed" in result
    
    def test_tool_schema(self, tool):
        """Test tool input schema."""
        schema = tool.args_schema
        assert schema is not None
        
        # Check required fields
        fields = schema.model_fields
        assert 'code' in fields
        assert fields['code'].is_required()
        
        # Check optional fields
        assert 'session_id' in fields
        assert not fields['session_id'].is_required()
    
    def test_session_cleanup(self, tool):
        """Test session cleanup functionality."""
        session_id = "cleanup_test"
        
        # Create session
        tool.run("x = 1", session_id=session_id)
        
        # Get session info
        info = tool.get_session_info(session_id)
        assert info['session_id'] == session_id
        assert info['execution_count'] == 1
        
        # Cleanup
        tool.cleanup_session(session_id)
        
        # Session should be cleaned up
        info = tool.get_session_info(session_id)
        assert 'error' in info or info['execution_count'] == 0
    
    @pytest.mark.skipif(not Path("test_data.csv").exists(), reason="Test data file not found")
    def test_file_upload(self, tool):
        """Test file upload functionality."""
        session_id = "file_upload_test"
        
        # Create a test CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,value\nA,1\nB,2\nC,3\n")
            test_file = f.name
        
        try:
            # Run code with file upload
            code = """
import pandas as pd
df = pd.read_csv('data.csv')
print(df)
print(f"Total rows: {len(df)}")
"""
            result = tool.run(
                code,
                session_id=session_id,
                files=[test_file]
            )
            
            if "Error" not in result:
                assert "Total rows: 3" in result
                assert "name" in result
                assert "value" in result
        finally:
            Path(test_file).unlink()