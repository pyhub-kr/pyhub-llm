"""Session management for code execution."""

import json
import pickle
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
import tempfile
import shutil


class CodeSession:
    """Manages execution session state."""
    
    def __init__(self, session_id: Optional[str] = None, timeout_minutes: int = 60):
        """
        Initialize a code session.
        
        Args:
            session_id: Optional session identifier
            timeout_minutes: Session timeout in minutes
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.timeout_minutes = timeout_minutes
        
        # Session state
        self.variables: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.created_files: List[str] = []
        self.imports: Set[str] = set()
        
        # Working directory for this session
        self.work_dir = Path(tempfile.mkdtemp(prefix=f"code_session_{self.session_id}_"))
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return (datetime.now() - self.last_activity) > timedelta(minutes=self.timeout_minutes)
    
    def touch(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def add_execution(self, code: str, result: Any, execution_time: float):
        """
        Add execution to history.
        
        Args:
            code: Executed code
            result: Execution result
            execution_time: Time taken to execute
        """
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "code": code,
            "result": str(result)[:1000],  # Limit stored result size
            "execution_time": execution_time
        })
        self.touch()
    
    def add_file(self, file_path: str):
        """
        Track a created file.
        
        Args:
            file_path: Path to created file
        """
        self.created_files.append(file_path)
    
    def cleanup(self):
        """Clean up session resources."""
        # Remove working directory
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "timeout_minutes": self.timeout_minutes,
            "execution_count": len(self.execution_history),
            "created_files": self.created_files,
            "imports": list(self.imports),
            "work_dir": str(self.work_dir)
        }
    
    def save(self, path: Path):
        """
        Save session to file.
        
        Args:
            path: Path to save session
        """
        session_data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "timeout_minutes": self.timeout_minutes,
            "variables": {},  # Don't save actual variables for security
            "execution_history": self.execution_history,
            "created_files": self.created_files,
            "imports": list(self.imports)
        }
        
        with open(path, 'wb') as f:
            pickle.dump(session_data, f)
    
    @classmethod
    def load(cls, path: Path) -> "CodeSession":
        """
        Load session from file.
        
        Args:
            path: Path to load session from
            
        Returns:
            Loaded session
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        session = cls(session_id=data["session_id"])
        session.created_at = data["created_at"]
        session.last_activity = data["last_activity"]
        session.timeout_minutes = data["timeout_minutes"]
        session.execution_history = data["execution_history"]
        session.created_files = data["created_files"]
        session.imports = set(data["imports"])
        
        return session


class SessionManager:
    """Manages multiple code sessions."""
    
    def __init__(self, max_sessions: int = 100):
        """
        Initialize session manager.
        
        Args:
            max_sessions: Maximum number of concurrent sessions
        """
        self.sessions: Dict[str, CodeSession] = {}
        self.max_sessions = max_sessions
    
    def create_session(self, session_id: Optional[str] = None) -> CodeSession:
        """
        Create a new session.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Created session
        """
        # Clean up expired sessions
        self._cleanup_expired()
        
        # Check session limit
        if len(self.sessions) >= self.max_sessions:
            # Remove oldest session
            oldest_id = min(self.sessions.keys(), 
                          key=lambda k: self.sessions[k].last_activity)
            self.remove_session(oldest_id)
        
        session = CodeSession(session_id)
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[CodeSession]:
        """
        Get existing session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if exists and not expired
        """
        session = self.sessions.get(session_id)
        if session and not session.is_expired():
            return session
        elif session:
            # Remove expired session
            self.remove_session(session_id)
        return None
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> CodeSession:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Session
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session(session_id)
    
    def remove_session(self, session_id: str):
        """
        Remove and cleanup session.
        
        Args:
            session_id: Session identifier
        """
        session = self.sessions.pop(session_id, None)
        if session:
            session.cleanup()
    
    def _cleanup_expired(self):
        """Remove expired sessions."""
        expired = [sid for sid, session in self.sessions.items() 
                  if session.is_expired()]
        for sid in expired:
            self.remove_session(sid)
    
    def cleanup_all(self):
        """Clean up all sessions."""
        for session_id in list(self.sessions.keys()):
            self.remove_session(session_id)