"""Docker-based code execution backend for Code Interpreter."""

import os
import json
import tempfile
import uuid
from typing import Any, Dict, Optional
import docker
from docker.errors import DockerException, ContainerError, ImageNotFound
import tarfile
import io

from .base import CodeExecutionBackend, ExecutionResult
from ..session import SessionManager


class DockerBackend(CodeExecutionBackend):
    """Docker-based code execution backend with container isolation."""
    
    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        image_name: str = "python:3.9-slim",
        network_mode: str = "none",
        memory_limit: str = "512m",
        cpu_quota: int = 50000,  # 50% of one CPU
        session_timeout: int = 3600,
        remote_docker_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize Docker backend.
        
        Args:
            image_name: Docker image to use
            network_mode: Docker network mode (default: none for security)
            memory_limit: Memory limit for containers
            cpu_quota: CPU quota for containers (microseconds per 100ms)
            session_timeout: Session timeout in seconds
            remote_docker_url: Optional remote Docker daemon URL
        """
        self.image_name = image_name
        self.network_mode = network_mode
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.session_timeout = session_timeout
        
        # Initialize Docker client
        if remote_docker_url:
            self.client = docker.DockerClient(base_url=remote_docker_url)
        else:
            self.client = docker.from_env()
            
        # Session manager for persistent state
        self.session_manager = session_manager or SessionManager(timeout=session_timeout)
        
        # Ensure base image exists
        self._ensure_image()
        
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
        
    def _ensure_image(self):
        """Ensure the Docker image exists, pull if necessary."""
        try:
            self.client.images.get(self.image_name)
        except ImageNotFound:
            print(f"Pulling Docker image: {self.image_name}")
            self.client.images.pull(self.image_name)
            
    def _create_container(self, session_id: str) -> docker.models.containers.Container:
        """Create a new container for the session."""
        container_name = f"code_interpreter_{session_id}_{uuid.uuid4().hex[:8]}"
        
        # Prepare container with required libraries
        container = self.client.containers.create(
            self.image_name,
            name=container_name,
            command="sleep infinity",  # Keep container running
            detach=True,
            network_mode=self.network_mode,
            mem_limit=self.memory_limit,
            cpu_quota=self.cpu_quota,
            cpu_period=100000,  # 100ms
            remove=True,
            environment={
                "PYTHONUNBUFFERED": "1",
                "SESSION_ID": session_id
            }
        )
        
        # Start the container
        container.start()
        
        # Install required packages
        packages = [
            "pandas==2.0.3",
            "numpy==1.24.3",
            "matplotlib==3.7.2",
            "seaborn==0.12.2",
            "scipy==1.10.1",
            "scikit-learn==1.3.0"
        ]
        
        install_cmd = f"pip install --no-cache-dir {' '.join(packages)}"
        result = container.exec_run(install_cmd)
        if result.exit_code != 0:
            container.stop()
            container.remove()
            raise RuntimeError(f"Failed to install packages: {result.output.decode()}")
            
        return container
        
    def _restore_session_state(self, container: docker.models.containers.Container, session_id: str):
        """Restore session state in container."""
        session = self.session_manager.get_session(session_id)
        if not session.variables:
            return
            
        # Create state restoration script
        restore_script = "import pickle\nimport base64\n"
        
        for var_name, var_data in session.variables.items():
            if isinstance(var_data, dict) and "_pickle_data" in var_data:
                restore_script += f"{var_name} = pickle.loads(base64.b64decode('{var_data['_pickle_data']}'))\n"
            else:
                restore_script += f"{var_name} = {json.dumps(var_data)}\n"
                
        # Execute restoration script
        self._execute_in_container(container, restore_script)
        
    def _execute_in_container(
        self,
        container: docker.models.containers.Container,
        code: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute code in container."""
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()  # Ensure content is written to disk
            temp_path = f.name
            
        try:
            # Create tar archive with the code directly (no need to read temp file)
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tarinfo = tarfile.TarInfo(name='script.py')
                tarinfo.size = len(code.encode())
                tar.addfile(tarinfo, io.BytesIO(code.encode()))
            
            tar_stream.seek(0)
            container.put_archive('/tmp', tar_stream)
                
            # Execute the script
            exec_result = container.exec_run(
                "python /tmp/script.py",
                stdout=True,
                stderr=True,
                stream=False
            )
            
            return {
                "exit_code": exec_result.exit_code,
                "output": exec_result.output.decode('utf-8')
            }
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    def _save_session_state(self, container: docker.models.containers.Container, session_id: str):
        """Save session state from container."""
        # Script to export session variables
        export_script = """
import json
import pickle
import base64
import types

# Get all user-defined variables
variables = {}
for name, value in globals().items():
    if not name.startswith('_') and not isinstance(value, types.ModuleType):
        try:
            # Try JSON serialization first
            json.dumps(value)
            variables[name] = value
        except:
            # Fall back to pickle
            try:
                variables[name] = {
                    '_pickle_data': base64.b64encode(pickle.dumps(value)).decode(),
                    '_type': str(type(value))
                }
            except:
                # Skip unserializable objects
                pass
                
print("__SESSION_STATE_START__")
print(json.dumps(variables))
print("__SESSION_STATE_END__")
"""
        
        result = self._execute_in_container(container, export_script)
        
        if result["exit_code"] == 0 and "__SESSION_STATE_START__" in result["output"]:
            # Extract state data
            output = result["output"]
            start_idx = output.find("__SESSION_STATE_START__") + len("__SESSION_STATE_START__")
            end_idx = output.find("__SESSION_STATE_END__")
            
            if start_idx < end_idx:
                state_json = output[start_idx:end_idx].strip()
                try:
                    variables = json.loads(state_json)
                    session = self.session_manager.get_session(session_id)
                    session.variables.update(variables)
                except json.JSONDecodeError:
                    pass
                    
    def execute(self, code: str, session_id: str, timeout: Optional[float] = None) -> ExecutionResult:
        """Execute code in a Docker container.
        
        Args:
            code: Python code to execute
            session_id: Session identifier
            timeout: Execution timeout in seconds
            
        Returns:
            ExecutionResult with output and any generated files
        """
        container = None
        
        try:
            # Create or get container for session
            container = self._create_container(session_id)
            
            # Restore session state
            self._restore_session_state(container, session_id)
            
            # Wrap code to capture output and handle matplotlib
            wrapped_code = f"""
import sys
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Capture stdout
old_stdout = sys.stdout
sys.stdout = io.StringIO()

try:
{self._indent_code(code)}
    
    # Save any plots
    import os
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(f'/tmp/figure_{{i}}.png', dpi=150, bbox_inches='tight')
        
finally:
    # Restore stdout and get output
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    print(output)
"""
            
            # Execute the code
            result = self._execute_in_container(container, wrapped_code, timeout)
            
            # Save session state
            self._save_session_state(container, session_id)
            
            # Collect generated files
            files = {}
            
            # Check for matplotlib figures
            for i in range(10):  # Check up to 10 figures
                fig_path = f"/tmp/figure_{i}.png"
                try:
                    tar_stream = container.get_archive(fig_path)[0]
                    with tarfile.open(fileobj=io.BytesIO(b''.join(tar_stream))) as tar:
                        for member in tar.getmembers():
                            f = tar.extractfile(member)
                            if f:
                                files[f"figure_{i}.png"] = f.read()
                                f.close()
                except:
                    break
                    
            # Update session
            session = self.session_manager.get_session(session_id)
            session.execution_count += 1
            session.files.update(files)
            
            return ExecutionResult(
                success=result["exit_code"] == 0,
                output=result["output"] if result["exit_code"] == 0 else "",
                error="" if result["exit_code"] == 0 else result["output"],
                files_created=list(files.keys()),
                execution_time=0.0,  # TODO: Measure actual time
                metadata={"files": files}
            )
            
        except ContainerError as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Container error: {str(e)}",
                files_created=[],
                execution_time=0.0
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Docker backend error: {str(e)}",
                files_created=[],
                execution_time=0.0
            )
        finally:
            # Clean up container
            if container:
                try:
                    container.stop(timeout=5)
                except Exception:
                    # Container might already be stopped
                    pass
                try:
                    container.remove(force=True)
                except Exception:
                    # Container might already be removed
                    pass
                    
    def _indent_code(self, code: str, indent: int = 4) -> str:
        """Indent code block."""
        lines = code.split('\n')
        return '\n'.join(' ' * indent + line if line.strip() else line for line in lines)
        
    def cleanup(self):
        """Cleanup any running containers."""
        # Stop all code interpreter containers
        for container in self.client.containers.list():
            if container.name.startswith("code_interpreter_"):
                try:
                    container.stop()
                    container.remove()
                except:
                    pass
                    
    def upload_file(self, session_id: str, local_path: Path, remote_path: Optional[str] = None) -> str:
        """Upload a file to the execution environment."""
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
            
        # Use container for this session
        container = self._create_container(session_id)
        
        try:
            # Determine remote path
            if not remote_path:
                remote_path = f"/tmp/{local_path.name}"
                
            # Create tar archive
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tar.add(str(local_path), arcname=local_path.name)
            tar_stream.seek(0)
            
            # Upload to container
            container.put_archive(os.path.dirname(remote_path), tar_stream)
            
            # Save file info in session
            session = self.session_manager.get_session(session_id)
            session.files[remote_path] = str(local_path)
            
            return remote_path
            
        finally:
            container.stop()
            container.remove()
            
    def download_file(self, session_id: str, remote_path: str, local_path: Optional[Path] = None) -> Path:
        """Download a file from the execution environment."""
        # Check if file exists in session
        session = self.session_manager.get_session(session_id)
        
        if remote_path in session.files:
            # File was created during execution
            file_data = session.files[remote_path]
            if isinstance(file_data, bytes):
                # Binary file data
                if not local_path:
                    local_path = Path(f"./{os.path.basename(remote_path)}")
                local_path.write_bytes(file_data)
                return local_path
                
        raise FileNotFoundError(f"File not found in session: {remote_path}")
        
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        session = self.session_manager.get_session(session_id)
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "execution_count": session.execution_count,
            "variables": list(session.variables.keys()),
            "created_files": list(session.files.keys()),
            "backend": "docker"
        }
        
    def cleanup_session(self, session_id: str) -> None:
        """Clean up resources for a session."""
        # Remove session from manager
        self.session_manager.remove_session(session_id)
        
        # Stop any containers for this session
        for container in self.client.containers.list():
            if f"code_interpreter_{session_id}" in container.name:
                try:
                    container.stop()
                    container.remove()
                except:
                    pass
                    
    def is_available(self) -> bool:
        """Check if Docker is available."""
        try:
            self.client.ping()
            return True
        except Exception:
            return False