"""Tests for Docker execution backend."""

import pytest
from pathlib import Path
import tempfile

# Skip all tests if docker is not available
docker = pytest.importorskip("docker")

try:
    from pyhub.llm.agents.tools.code_interpreter.backends import DockerBackend
except ImportError:
    pytest.skip("DockerBackend not available", allow_module_level=True)

from pyhub.llm.agents.tools.code_interpreter.session import SessionManager


class TestDockerBackend:
    """Test Docker execution backend."""
    
    @pytest.fixture
    def backend(self):
        """Create Docker backend instance."""
        try:
            backend = DockerBackend()
            if not backend.is_available():
                pytest.skip("Docker is not available")
            return backend
        except Exception as e:
            pytest.skip(f"Docker backend initialization failed: {e}")
    
    def test_docker_available(self):
        """Test Docker availability check."""
        backend = DockerBackend()
        # This test will pass if Docker is available, otherwise skip
        if not backend.is_available():
            pytest.skip("Docker is not running")
    
    def test_basic_execution(self, backend):
        """Test basic code execution in Docker."""
        code = "result = 2 + 2\nprint(result)"
        result = backend.execute(code, "test_session")
        
        assert result.success
        assert "4" in result.output
        assert result.execution_time >= 0
    
    def test_session_isolation(self, backend):
        """Test that sessions are isolated in Docker."""
        # Session 1
        code1 = "x = 42"
        result1 = backend.execute(code1, "session1")
        assert result1.success
        
        # Session 2 should not see x from session 1
        code2 = "try:\n    print(x)\nexcept NameError:\n    print('x not defined')"
        result2 = backend.execute(code2, "session2")
        assert result2.success
        assert "x not defined" in result2.output
    
    def test_resource_limits(self, backend):
        """Test that resource limits are enforced."""
        # Try to allocate too much memory
        code = """
import numpy as np
try:
    # Try to allocate 1GB (should fail with 512MB limit)
    big_array = np.zeros((1024, 1024, 1024), dtype=np.uint8)
    print("Memory allocation succeeded")
except MemoryError:
    print("Memory limit enforced")
"""
        result = backend.execute(code, "memory_test")
        
        # Either it fails to allocate or gets killed by Docker
        if result.success:
            assert "Memory limit enforced" in result.output
    
    def test_network_isolation(self, backend):
        """Test that network access is blocked."""
        code = """
import socket
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Network access allowed")
except Exception as e:
    print(f"Network blocked: {e}")
"""
        result = backend.execute(code, "network_test")
        
        # Network operations should fail
        if result.success:
            assert "Network blocked" in result.output
    
    def test_pandas_in_docker(self, backend):
        """Test pandas functionality in Docker."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.sum())
print("Pandas works!")
"""
        result = backend.execute(code, "pandas_docker")
        
        assert result.success
        assert "A" in result.output
        assert "B" in result.output
        assert "Pandas works!" in result.output
    
    def test_matplotlib_in_docker(self, backend):
        """Test matplotlib functionality in Docker."""
        code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.savefig('/tmp/sine_wave.png')
print("Plot saved successfully")
"""
        result = backend.execute(code, "plot_docker")
        
        assert result.success
        assert "Plot saved successfully" in result.output
        assert any("figure" in f for f in result.files_created)
    
    def test_container_cleanup(self, backend):
        """Test that containers are properly cleaned up."""
        session_id = "cleanup_test"
        
        # Execute code
        backend.execute("x = 1", session_id)
        
        # Check no containers are left running
        containers = backend.client.containers.list()
        session_containers = [c for c in containers if f"code_interpreter_{session_id}" in c.name]
        assert len(session_containers) == 0
    
    def test_file_persistence_in_session(self, backend):
        """Test file operations within Docker session."""
        session_id = "file_docker_test"
        
        # Create a file
        code1 = """
with open('/tmp/test.txt', 'w') as f:
    f.write('Hello from Docker')
print("File created")
"""
        result1 = backend.execute(code1, session_id)
        assert result1.success
        
        # Read the file in same session (new container)
        code2 = """
with open('/tmp/test.txt', 'r') as f:
    content = f.read()
print(f"Content: {content}")
"""
        result2 = backend.execute(code2, session_id)
        # Files don't persist between container runs by default
        # This should fail or show empty
        assert not result2.success or "Hello from Docker" not in result2.output