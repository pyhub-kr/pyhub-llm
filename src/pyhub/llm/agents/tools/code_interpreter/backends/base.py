"""Base interface for code execution backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result of code execution."""
    
    success: bool
    output: str = ""
    error: str = ""
    files_created: List[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.files_created is None:
            self.files_created = []
        if self.metadata is None:
            self.metadata = {}


class CodeExecutionBackend(ABC):
    """Abstract base class for code execution backends."""
    
    @abstractmethod
    def execute(self, code: str, session_id: str, timeout: Optional[float] = None) -> ExecutionResult:
        """
        Execute Python code.
        
        Args:
            code: Python code to execute
            session_id: Session identifier for maintaining state
            timeout: Maximum execution time in seconds
            
        Returns:
            ExecutionResult containing output, errors, and metadata
        """
        pass
    
    @abstractmethod
    def upload_file(self, session_id: str, local_path: Path, remote_path: Optional[str] = None) -> str:
        """
        Upload a file to the execution environment.
        
        Args:
            session_id: Session identifier
            local_path: Path to local file
            remote_path: Optional path in execution environment
            
        Returns:
            Path to file in execution environment
        """
        pass
    
    @abstractmethod
    def download_file(self, session_id: str, remote_path: str, local_path: Optional[Path] = None) -> Path:
        """
        Download a file from the execution environment.
        
        Args:
            session_id: Session identifier
            remote_path: Path in execution environment
            local_path: Optional local destination path
            
        Returns:
            Path to downloaded file
        """
        pass
    
    @abstractmethod
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary containing session information
        """
        pass
    
    @abstractmethod
    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up resources for a session.
        
        Args:
            session_id: Session identifier
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the backend is available and ready to use.
        
        Returns:
            True if backend is available
        """
        pass