"""
Code Interpreter tool for executing Python code in various environments.

Supports multiple execution backends:
- Local: Restricted local execution
- Docker: Containerized execution
- Remote Docker: Remote server execution
"""

from .code_interpreter import CodeInterpreter

__all__ = ["CodeInterpreter"]