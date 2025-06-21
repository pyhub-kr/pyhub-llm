"""Tests for local execution backend."""

import pytest
from pathlib import Path
import tempfile

from pyhub.llm.agents.tools.code_interpreter.backends import LocalBackend
from pyhub.llm.agents.tools.code_interpreter.session import SessionManager


class TestLocalBackend:
    """Test local execution backend."""
    
    @pytest.fixture
    def backend(self):
        """Create local backend instance."""
        return LocalBackend()
    
    def test_basic_execution(self, backend):
        """Test basic code execution."""
        code = "result = 2 + 2\nprint(result)"
        result = backend.execute(code, "test_session")
        
        assert result.success
        assert "4" in result.output
        assert result.execution_time > 0
    
    def test_session_persistence(self, backend):
        """Test that variables persist across executions."""
        # First execution
        code1 = "x = 42"
        result1 = backend.execute(code1, "persist_session")
        assert result1.success
        
        # Second execution in same session
        code2 = "print(x)"
        result2 = backend.execute(code2, "persist_session")
        assert result2.success
        assert "42" in result2.output
    
    def test_pandas_support(self, backend):
        """Test pandas functionality."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.sum())
"""
        result = backend.execute(code, "pandas_session")
        
        if result.success:
            assert "A" in result.output
            assert "B" in result.output
        else:
            # Pandas might not be installed in test environment
            assert "No module named 'pandas'" in result.error
    
    def test_matplotlib_support(self, backend):
        """Test matplotlib functionality."""
        code = """
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3))
plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig('test_plot.png')
print("Plot saved")
"""
        result = backend.execute(code, "plot_session")
        
        if result.success:
            assert "Plot saved" in result.output
            assert any("test_plot.png" in f for f in result.files_created)
        else:
            # Matplotlib might not be installed
            assert "No module named 'matplotlib'" in result.error
    
    def test_security_validation(self, backend):
        """Test that dangerous code is blocked."""
        dangerous_codes = [
            "import os\nos.system('ls')",
            "import subprocess\nsubprocess.run(['ls'])",
            "open('/etc/passwd', 'r')",
            "__import__('os').system('ls')",
            "exec('print(1)')",
            "eval('2+2')",
        ]
        
        for code in dangerous_codes:
            result = backend.execute(code, f"security_test")
            assert not result.success
            assert "not allowed" in result.error.lower()
    
    def test_file_operations(self, backend):
        """Test file upload and download."""
        session_id = "file_test"
        
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            test_file = Path(f.name)
        
        try:
            # Upload file
            remote_path = backend.upload_file(session_id, test_file)
            assert remote_path
            
            # Use file in code (using pandas to avoid direct open)
            code = f"""
import pandas as pd
# Read as text file using pandas
with open('{test_file.name}', 'r') as f:
    content = f.read()
print(f"File content: {{content}}")
"""
            result = backend.execute(code, session_id)
            assert result.success
            assert "test content" in result.output
            
            # Download file
            downloaded = backend.download_file(session_id, test_file.name)
            assert downloaded.exists()
            assert downloaded.read_text() == "test content"
            
        finally:
            test_file.unlink()
    
    def test_error_handling(self, backend):
        """Test error handling in code execution."""
        code = """
def divide(a, b):
    return a / b

result = divide(10, 0)
"""
        result = backend.execute(code, "error_test")
        
        assert not result.success
        assert "ZeroDivisionError" in result.error
    
    def test_session_cleanup(self, backend):
        """Test session cleanup."""
        session_id = "cleanup_test"
        
        # Execute code to create session
        backend.execute("x = 1", session_id)
        
        # Get session info
        info = backend.get_session_info(session_id)
        assert info['session_id'] == session_id
        
        # Cleanup session
        backend.cleanup_session(session_id)
        
        # Session should be gone
        info = backend.get_session_info(session_id)
        assert 'error' in info