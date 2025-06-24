"""Code Interpreter tool implementation."""

import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field

from pyhub.llm.agents.base import BaseTool
from .backends import CodeExecutionBackend, LocalBackend
from .session import SessionManager


class CodeInterpreterInput(BaseModel):
    """Input schema for Code Interpreter tool."""
    
    code: str = Field(..., description="Python code to execute")
    session_id: Optional[str] = Field(None, description="Session ID for maintaining state between executions")
    files: Optional[List[str]] = Field(None, description="List of file paths to make available in the execution environment")


class CodeInterpreter(BaseTool):
    """
    Execute Python code for data analysis and visualization.
    
    Supports multiple execution backends:
    - local: Restricted local Python environment
    - docker: Isolated Docker container (coming soon)
    - remote: Remote Docker server (coming soon)
    """
    
    def __init__(
        self,
        backend: str = "local",
        backend_config: Optional[Dict[str, Any]] = None,
        session_manager: Optional[SessionManager] = None,
        name: str = "code_interpreter",
        description: str = "Execute Python code for data analysis, calculations, and visualizations",
        include_usage_notes: bool = True
    ):
        """
        Initialize Code Interpreter tool.
        
        Args:
            backend: Execution backend ("local", "docker", "remote")
            backend_config: Configuration for the backend
            session_manager: Optional session manager instance
            name: Tool name
            description: Tool description
            include_usage_notes: Whether to include usage notes in description
        """
        # Add usage notes to description if requested
        if include_usage_notes:
            usage_notes = (
                "\n\nIMPORTANT USAGE NOTES:\n"
                "- For matplotlib plots: Always use plt.savefig('filename.png') instead of plt.show()\n"
                "- Print confirmation messages after saving files\n"
                "- Use descriptive filenames for saved outputs"
            )
            full_description = description + usage_notes
        else:
            full_description = description
            
        super().__init__(name=name, description=full_description)
        
        self.backend_name = backend
        self.backend_config = backend_config or {}
        self.session_manager = session_manager or SessionManager()
        
        # Initialize backend
        self._init_backend()
        
        # Set input schema
        self.args_schema = CodeInterpreterInput
    
    def _init_backend(self):
        """Initialize the execution backend."""
        if self.backend_name == "local":
            self.backend = LocalBackend(
                session_manager=self.session_manager,
                **self.backend_config
            )
        elif self.backend_name == "docker":
            try:
                from .backends import DockerBackend
                self.backend = DockerBackend(
                    session_manager=self.session_manager,
                    **self.backend_config
                )
            except ImportError:
                raise ImportError("Docker backend requires 'docker' package. Install with: pip install docker")
        elif self.backend_name == "remote":
            # TODO: Implement remote Docker backend
            raise NotImplementedError("Remote Docker backend not yet implemented")
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")
        
        # Check if backend is available
        if not self.backend.is_available():
            raise RuntimeError(f"Backend '{self.backend_name}' is not available")
    
    def run(
        self,
        code: str,
        session_id: Optional[str] = None,
        files: Optional[List[str]] = None
    ) -> str:
        """
        Execute Python code.
        
        Args:
            code: Python code to execute
            session_id: Optional session ID for state persistence
            files: Optional list of files to upload to execution environment
            
        Returns:
            Formatted execution result
        """
        # Use provided session_id or create a new one
        if not session_id:
            session_id = self.session_manager.create_session().session_id
        
        # Upload files if provided
        if files:
            for file_path in files:
                try:
                    self.backend.upload_file(session_id, Path(file_path))
                except Exception as e:
                    return f"Error uploading file {file_path}: {str(e)}"
        
        # Execute code
        result = self.backend.execute(code, session_id)
        
        # Format response
        return self._format_result(result)
    
    def _format_result(self, result) -> str:
        """
        Format execution result for LLM response.
        
        Args:
            result: ExecutionResult from backend
            
        Returns:
            Formatted string response
        """
        if result.success:
            output_parts = []
            
            # Add main output
            if result.output:
                output_parts.append(f"Output:\n{result.output}")
            
            # Add any stderr output (warnings, etc.)
            if result.error:
                output_parts.append(f"Warnings:\n{result.error}")
            
            # Add file information
            if result.files_created:
                output_parts.append(f"Files created: {', '.join(result.files_created)}")
            
            # Add execution time
            output_parts.append(f"Execution time: {result.execution_time:.3f}s")
            
            return "\n\n".join(output_parts)
        else:
            return f"Error executing code:\n{result.error}"
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information
        """
        return self.backend.get_session_info(session_id)
    
    def cleanup_session(self, session_id: str):
        """
        Clean up a session.
        
        Args:
            session_id: Session identifier
        """
        self.backend.cleanup_session(session_id)
    
    def download_file(self, session_id: str, file_path: str, local_path: Optional[str] = None) -> str:
        """
        Download a file from the execution environment.
        
        Args:
            session_id: Session identifier
            file_path: Path to file in execution environment
            local_path: Optional local destination path
            
        Returns:
            Path to downloaded file
        """
        result_path = self.backend.download_file(
            session_id,
            file_path,
            Path(local_path) if local_path else None
        )
        return str(result_path)
    
    def get_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return self.backend.get_sessions()
    
    def clear_session(self, session_id: str) -> None:
        """Clear a specific session."""
        self.backend.clear_session(session_id)
