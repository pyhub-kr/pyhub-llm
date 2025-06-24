"""Local execution backend with restricted environment."""

import io
import os
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile
import shutil

from .base import CodeExecutionBackend, ExecutionResult
from ..security import CodeSecurityValidator, SecurityError
from ..session import CodeSession, SessionManager


class LocalBackend(CodeExecutionBackend):
    """
    Local Python execution backend with security restrictions.
    
    This backend executes code in the local Python environment but with
    security restrictions and isolated namespaces.
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None,
                 additional_allowed_imports: Optional[set] = None):
        """
        Initialize local backend.
        
        Args:
            session_manager: Optional session manager instance
            additional_allowed_imports: Additional safe imports to allow
        """
        self.session_manager = session_manager or SessionManager()
        self.security_validator = CodeSecurityValidator(additional_allowed_imports)
        self._setup_safe_builtins()
    
    def _setup_safe_builtins(self):
        """Set up restricted builtins for code execution."""
        # Safe builtins - no file operations, imports, or dangerous functions
        self.safe_builtins = {
            # Type constructors
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'frozenset': frozenset, 'bytes': bytes, 'bytearray': bytearray,
            
            # Safe functions
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'sum': sum, 'min': min, 'max': max,
            'abs': abs, 'round': round, 'pow': pow,
            'sorted': sorted, 'reversed': reversed,
            'all': all, 'any': any,
            
            # Type checking
            'isinstance': isinstance, 'type': type,
            '__build_class__': __builtins__['__build_class__'],
            '__name__': '__main__',
            
            # String/repr
            'print': print, 'repr': repr, 'format': format,
            
            # Math
            'divmod': divmod,
            
            # Exceptions (read-only)
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'AttributeError': AttributeError,
            'NameError': NameError,
            'ZeroDivisionError': ZeroDivisionError,
            'ImportError': ImportError,
            'RuntimeError': RuntimeError,
            
            # Constants
            'True': True, 'False': False, 'None': None,
            
            # Safe file operations (will be wrapped)
            'open': None,  # Will be replaced with safe_open in _create_safe_globals
        }
    
    def _create_safe_globals(self, session: CodeSession) -> Dict[str, Any]:
        """
        Create safe global namespace for code execution.
        
        Args:
            session: Code session
            
        Returns:
            Safe globals dictionary
        """
        # Start with safe builtins
        safe_globals = {'__builtins__': self.safe_builtins.copy()}
        
        # Add safe import function that only allows whitelisted modules
        def safe_import(name, *args, **kwargs):
            if name in self.security_validator.allowed_imports:
                return __import__(name, *args, **kwargs)
            # Allow submodules of safe packages
            elif any(name.startswith(safe_pkg + '.') for safe_pkg in ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'scipy']):
                return __import__(name, *args, **kwargs)
            else:
                raise ImportError(f"Import of '{name}' is not allowed")
        
        safe_globals['__builtins__']['__import__'] = safe_import
        
        # Add safe file operations
        def safe_open(filename, mode='r', *args, **kwargs):
            """Safe open that only allows files in session directory."""
            file_path = Path(filename).resolve()
            session_dir = Path(session.work_dir).resolve()
            
            # Check if file is within session directory
            try:
                file_path.relative_to(session_dir)
            except ValueError:
                # Also allow created files in session
                if str(file_path) not in [str(Path(f).resolve()) for f in session.created_files]:
                    raise PermissionError(f"Access to '{filename}' is not allowed")
            
            # Only allow read and write modes, no execute
            allowed_modes = {'r', 'rb', 'r+', 'w', 'wb', 'w+', 'a', 'ab', 'a+'}
            if mode not in allowed_modes:
                raise ValueError(f"File mode '{mode}' is not allowed")
                
            return open(filename, mode, *args, **kwargs)
        
        safe_globals['__builtins__']['open'] = safe_open
        
        # Pre-import common modules for convenience
        preload_modules = {
            'pandas': 'pd',
            'numpy': 'np',
            'matplotlib': 'matplotlib',
            'matplotlib.pyplot': 'plt',
            'seaborn': 'sns',
        }
        
        for module_name, alias in preload_modules.items():
            if module_name in self.security_validator.allowed_imports or module_name.startswith('matplotlib'):
                try:
                    if module_name == "matplotlib":
                        import matplotlib
                        matplotlib.use('Agg')  # Non-interactive backend
                        safe_globals[alias] = matplotlib
                    elif module_name == "matplotlib.pyplot":
                        import matplotlib.pyplot as plt
                        safe_globals[alias] = plt
                    else:
                        module = __import__(module_name)
                        safe_globals[alias] = module
                except ImportError:
                    pass  # Module not available
        
        return safe_globals
    
    @contextmanager
    def _capture_output(self):
        """Context manager to capture stdout and stderr."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            yield stdout_buffer, stderr_buffer
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def execute(self, code: str, session_id: str, timeout: Optional[float] = 30.0) -> ExecutionResult:
        """
        Execute Python code in a restricted local environment.
        
        Args:
            code: Python code to execute
            session_id: Session identifier
            timeout: Maximum execution time (not enforced in local backend)
            
        Returns:
            ExecutionResult with output and metadata
        """
        start_time = time.time()
        
        # Get or create session
        session = self.session_manager.get_or_create_session(session_id)
        
        # Validate code security (but allow 'open' since we provide safe_open)
        try:
            is_safe, issues = self.security_validator.validate(code)
            # Filter out open() issues since we provide safe_open
            filtered_issues = [issue for issue in issues if "file operations" not in issue or "open" not in code]
            if filtered_issues:
                return ExecutionResult(
                    success=False,
                    error=f"Security validation failed:\n" + "\n".join(f"- {issue}" for issue in filtered_issues),
                    execution_time=time.time() - start_time
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Security validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
        
        # Create safe execution environment
        safe_globals = self._create_safe_globals(session)
        # Merge session variables and imported modules into namespace
        local_namespace = {}
        local_namespace.update(session.variables)
        
        # Set working directory to session directory
        original_cwd = Path.cwd()
        
        try:
            # Change to session working directory
            import os
            os.chdir(session.work_dir)
            
            # Execute code with output capture
            # Merge safe_globals and local_namespace for execution
            # This allows functions to call themselves and other definitions
            exec_namespace = safe_globals.copy()
            exec_namespace.update(local_namespace)
            
            with self._capture_output() as (stdout, stderr):
                exec(code, exec_namespace)
                
            # Extract new/modified variables back to local_namespace
            for key in exec_namespace:
                if key not in safe_globals or exec_namespace[key] != safe_globals.get(key):
                    local_namespace[key] = exec_namespace[key]
            
            # Update session variables (including imported modules)
            for key, value in local_namespace.items():
                if not key.startswith('_'):
                    if self._is_safe_variable(value) or self._is_module(value):
                        session.variables[key] = value
            
            # Get output
            output = stdout.getvalue()
            error_output = stderr.getvalue()
            
            # Check for created files
            files_created = []
            for file_path in session.work_dir.iterdir():
                if file_path.is_file() and str(file_path) not in session.created_files:
                    files_created.append(str(file_path))
                    session.add_file(str(file_path))
            
            execution_time = time.time() - start_time
            session.add_execution(code, output, execution_time)
            
            return ExecutionResult(
                success=True,
                output=output,
                error=error_output,
                files_created=files_created,
                execution_time=execution_time,
                metadata={
                    'session_id': session_id,
                    'variables': list(session.variables.keys())
                }
            )
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time,
                metadata={'session_id': session_id}
            )
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def _is_safe_variable(self, value: Any) -> bool:
        """Check if a variable is safe to store in session."""
        # Allow basic types and common data structures
        safe_types = (int, float, str, bool, list, dict, tuple, set, type(None))
        
        # Check pandas/numpy types if available
        try:
            import pandas as pd
            import numpy as np
            safe_types = safe_types + (pd.DataFrame, pd.Series, np.ndarray)
        except ImportError:
            pass
        
        if isinstance(value, safe_types):
            return True
        
        # Allow instances of user-defined classes (but not built-in/system classes)
        # Check if it's a class instance (has __class__ and it's not a built-in type)
        if hasattr(value, '__class__'):
            module = getattr(type(value), '__module__', None)
            # Allow if it's from __main__ (user-defined) or if it's a known safe type
            if module == '__main__' or (module and module.startswith(('pandas', 'numpy', 'matplotlib', 'sklearn', 'scipy', 'seaborn'))):
                return True
        
        return False
    
    def _is_module(self, value: Any) -> bool:
        """Check if a value is a module, function, or class."""
        import types
        return isinstance(value, (types.ModuleType, types.FunctionType, type)) or callable(value)
    
    def upload_file(self, session_id: str, local_path: Path, remote_path: Optional[str] = None) -> str:
        """
        Copy file to session working directory.
        
        Args:
            session_id: Session identifier
            local_path: Path to local file
            remote_path: Optional path in session directory
            
        Returns:
            Path to file in session directory
        """
        session = self.session_manager.get_or_create_session(session_id)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        # Determine destination path
        if remote_path:
            dest_path = session.work_dir / remote_path
        else:
            dest_path = session.work_dir / local_path.name
        
        # Create parent directories if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(local_path, dest_path)
        session.add_file(str(dest_path))
        
        return str(dest_path)
    
    def download_file(self, session_id: str, remote_path: str, local_path: Optional[Path] = None) -> Path:
        """
        Copy file from session directory.
        
        Args:
            session_id: Session identifier
            remote_path: Path in session directory
            local_path: Optional local destination
            
        Returns:
            Path to downloaded file
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # Resolve source path
        if Path(remote_path).is_absolute():
            src_path = Path(remote_path)
        else:
            src_path = session.work_dir / remote_path
        
        if not src_path.exists():
            raise FileNotFoundError(f"File not found in session: {remote_path}")
        
        # Determine destination
        if local_path is None:
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix=src_path.suffix)
            os.close(fd)
            local_path = Path(temp_path)
        
        # Copy file
        shutil.copy2(src_path, local_path)
        
        return local_path
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        return session.to_dict()
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        self.session_manager.remove_session(session_id)
    
    def is_available(self) -> bool:
        """Local backend is always available."""
        return True
    
    def get_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.session_manager.sessions.keys())
    
    def clear_session(self, session_id: str) -> None:
        """Clear a specific session."""
        self.session_manager.remove_session(session_id)