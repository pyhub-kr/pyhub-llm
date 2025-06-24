"""Code execution backends for Code Interpreter."""

from .base import CodeExecutionBackend, ExecutionResult
from .local import LocalBackend

# Optional Docker backend
try:
    from .docker import DockerBackend
    __all__ = ["CodeExecutionBackend", "ExecutionResult", "LocalBackend", "DockerBackend"]
except ImportError:
    __all__ = ["CodeExecutionBackend", "ExecutionResult", "LocalBackend"]