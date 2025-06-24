"""Security validation for code execution."""

import re
from typing import List, Set, Tuple


class SecurityError(Exception):
    """Raised when code fails security validation."""
    pass


class CodeSecurityValidator:
    """Validates code for security risks before execution."""
    
    # Dangerous imports that should be blocked
    DANGEROUS_IMPORTS = {
        "os", "sys", "subprocess", "shutil", "socket", "urllib",
        "requests", "http", "ftplib", "telnetlib", "smtplib",
        "importlib", "__builtins__", "eval", "exec", "compile",
        "open", "file", "input", "raw_input"
    }
    
    # Allowed safe imports for data analysis
    ALLOWED_IMPORTS = {
        "pandas", "numpy", "matplotlib", "matplotlib.pyplot", "seaborn", "plotly",
        "scipy", "sklearn", "sklearn.datasets", "sklearn.model_selection",
        "sklearn.preprocessing", "sklearn.linear_model", "sklearn.ensemble",
        "sklearn.svm", "sklearn.metrics", "datetime", "json", "math", "random",
        "collections", "itertools", "functools", "re", "typing",
        "dataclasses", "enum", "decimal", "fractions", "statistics"
    }
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        # System access
        (r"os\s*\.", "Direct OS access is not allowed"),
        (r"sys\s*\.", "Direct system access is not allowed"),
        (r"subprocess\s*\.", "Subprocess execution is not allowed"),
        
        # File operations (we'll handle this through safe_open in backend)
        (r"open\s*\(", "Direct file operations are not allowed"),
        (r"file\s*\(", "Direct file operations are not allowed"),
        (r"__builtins__\s*\[", "Access to builtins is not allowed"),
        
        # Network access
        (r"socket\s*\.", "Network access is not allowed"),
        (r"urllib\s*\.", "Network access is not allowed"),
        (r"requests\s*\.", "Network access is not allowed"),
        
        # Code execution
        (r"exec\s*\(", "Dynamic code execution is not allowed"),
        (r"eval\s*\(", "Dynamic code evaluation is not allowed"),
        (r"compile\s*\(", "Code compilation is not allowed"),
        (r"__import__", "Dynamic imports are not allowed"),
        
        # Dangerous builtins
        (r"globals\s*\(", "Access to globals is not allowed"),
        (r"locals\s*\(", "Access to locals is not allowed"),
        (r"vars\s*\(", "Access to vars is not allowed"),
        (r"dir\s*\(", "Directory listing is not allowed"),
        
        # Private attribute access
        (r"\.__[_a-zA-Z0-9]+__", "Access to private attributes is not allowed"),
        (r"__builtins__", "Access to __builtins__ is not allowed"),
    ]
    
    def __init__(self, additional_allowed_imports: Set[str] = None):
        """
        Initialize security validator.
        
        Args:
            additional_allowed_imports: Additional safe imports to allow
        """
        self.allowed_imports = self.ALLOWED_IMPORTS.copy()
        if additional_allowed_imports:
            self.allowed_imports.update(additional_allowed_imports)
    
    def validate(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate code for security risks.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []
        
        # Check for dangerous patterns
        for pattern, message in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(message)
        
        # Check imports (including multiline and aliased imports)
        # First, normalize multiline imports by joining lines that end with backslash
        normalized_code = re.sub(r'\\\s*\n', ' ', code)
        
        # Check various import patterns
        import_patterns = [
            r'(?:from\s+(\S+)\s+import|import\s+(\S+))',  # Basic import
            r'from\s+(\S+)\s+import\s+\([^)]+\)',  # Multiline from import
            r'import\s+(\S+)\s+as\s+\w+',  # Import with alias
            r'from\s+(\S+)\s+import\s+\S+\s+as\s+\w+',  # From import with alias
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, normalized_code, re.MULTILINE | re.DOTALL):
                module = match.group(1) or match.group(2) or ''
                base_module = module.split('.')[0]
                
                if base_module in self.DANGEROUS_IMPORTS:
                    issues.append(f"Import of '{base_module}' is not allowed")
                elif base_module and module not in self.allowed_imports:
                    # Check if it's a relative import or standard safe module
                    # Allow internal modules for known safe packages
                    if module.startswith(('numpy.', 'pandas.', 'matplotlib.', 'seaborn.', 'sklearn.', 'scipy.')):
                        continue  # Allow submodules of safe packages
                    if not module.startswith('.') and base_module not in self.allowed_imports:
                        issues.append(f"Import of '{module}' is not in the allowed list")
        
        # Check for obfuscated imports using __import__
        if '__import__' in code:
            issues.append("Use of __import__ is not allowed (possible obfuscation)")
        
        # Check for attempts to access file system through pandas
        # (pandas file operations are allowed but with restrictions)
        if "read_" in code:
            # Check for suspicious file paths
            suspicious_paths = [r"\.\.\/", r"\.\.\\", r"~\/", r"\/etc", r"\/sys", r"\/proc"]
            for path_pattern in suspicious_paths:
                if re.search(path_pattern, code):
                    issues.append("Suspicious file path detected")
                    break
        
        return len(issues) == 0, issues
    
    def sanitize_code(self, code: str) -> str:
        """
        Basic code sanitization (removes comments and empty lines).
        
        Args:
            code: Python code to sanitize
            
        Returns:
            Sanitized code
        """
        lines = []
        for line in code.split('\n'):
            # Remove comments but preserve string literals
            if '#' in line:
                # Simple approach - doesn't handle # in strings perfectly
                line = line.split('#')[0].rstrip()
            if line.strip():
                lines.append(line)
        
        return '\n'.join(lines)