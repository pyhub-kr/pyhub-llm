"""Tests for code security validation."""

import pytest

from pyhub.llm.agents.tools.code_interpreter.security import CodeSecurityValidator, SecurityError


class TestCodeSecurityValidator:
    """Test code security validation."""
    
    @pytest.fixture
    def validator(self):
        """Create security validator instance."""
        return CodeSecurityValidator()
    
    def test_safe_code(self, validator):
        """Test that safe code passes validation."""
        safe_codes = [
            "x = 2 + 2",
            "import pandas as pd\ndf = pd.DataFrame()",
            "import numpy as np\narr = np.array([1, 2, 3])",
            "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])",
            "for i in range(10):\n    print(i)",
            "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
        ]
        
        for code in safe_codes:
            is_safe, issues = validator.validate(code)
            assert is_safe, f"Safe code rejected: {code}\nIssues: {issues}"
            assert len(issues) == 0
    
    def test_dangerous_imports(self, validator):
        """Test that dangerous imports are blocked."""
        dangerous_imports = [
            "import os",
            "import sys",
            "import subprocess",
            "from os import system",
            "import socket",
            "import urllib",
            "import requests",
            "__import__('os')",
        ]
        
        for code in dangerous_imports:
            is_safe, issues = validator.validate(code)
            assert not is_safe, f"Dangerous import allowed: {code}"
            assert len(issues) > 0
            assert any("not allowed" in issue for issue in issues)
    
    def test_dangerous_patterns(self, validator):
        """Test that dangerous patterns are detected."""
        dangerous_patterns = [
            "exec('print(1)')",
            "eval('2+2')",
            "compile('x=1', 'string', 'exec')",
            "open('/etc/passwd', 'r')",
            "file('/etc/passwd', 'r')",
            "globals()['__builtins__']",
            "locals()['__name__']",
            "dir(__builtins__)",
            "__import__('os').system('ls')",
        ]
        
        for code in dangerous_patterns:
            is_safe, issues = validator.validate(code)
            assert not is_safe, f"Dangerous pattern allowed: {code}"
            assert len(issues) > 0
    
    def test_file_path_validation(self, validator):
        """Test that suspicious file paths are detected."""
        suspicious_codes = [
            "pd.read_csv('../../../etc/passwd')",
            "pd.read_csv('~/sensitive_file.csv')",
            "pd.read_csv('/etc/shadow')",
            "pd.read_excel('/sys/something')",
        ]
        
        for code in suspicious_codes:
            is_safe, issues = validator.validate(code)
            assert not is_safe, f"Suspicious path allowed: {code}"
            assert any("suspicious" in issue.lower() for issue in issues)
    
    def test_safe_file_operations(self, validator):
        """Test that legitimate file operations are allowed."""
        safe_file_ops = [
            "pd.read_csv('data.csv')",
            "pd.read_excel('report.xlsx')",
            "df.to_csv('output.csv')",
            "plt.savefig('plot.png')",
        ]
        
        for code in safe_file_ops:
            is_safe, issues = validator.validate(code)
            assert is_safe, f"Safe file operation rejected: {code}\nIssues: {issues}"
    
    def test_private_attribute_access(self, validator):
        """Test that private attribute access is blocked."""
        private_access = [
            "obj.__dict__",
            "cls.__bases__",
            "func.__code__",
            "__builtins__.__dict__",
        ]
        
        for code in private_access:
            is_safe, issues = validator.validate(code)
            assert not is_safe, f"Private attribute access allowed: {code}"
            assert any("private" in issue.lower() for issue in issues)
    
    def test_additional_allowed_imports(self):
        """Test custom allowed imports."""
        validator = CodeSecurityValidator(additional_allowed_imports={"custom_module"})
        
        # Custom module should be allowed
        is_safe, issues = validator.validate("import custom_module")
        assert is_safe
        assert len(issues) == 0
        
        # Dangerous imports should still be blocked
        is_safe, issues = validator.validate("import os")
        assert not is_safe
    
    def test_code_sanitization(self, validator):
        """Test code sanitization."""
        code_with_comments = """
# This is a comment
x = 1  # Another comment
# import os  # This should be removed
y = 2
"""
        sanitized = validator.sanitize_code(code_with_comments)
        
        assert "# This is a comment" not in sanitized
        assert "x = 1" in sanitized
        assert "y = 2" in sanitized
        assert "import os" not in sanitized