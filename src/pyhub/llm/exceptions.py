"""Exception classes for PyHub LLM."""

from typing import Optional


class LLMError(Exception):
    """Base exception for all LLM errors."""
    
    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.code = code


class APIError(LLMError):
    """API communication error."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(APIError):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class InvalidRequestError(APIError):
    """Invalid request parameters."""
    
    def __init__(self, message: str, param: Optional[str] = None):
        super().__init__(message, status_code=400)
        self.param = param


class AuthenticationError(APIError):
    """Authentication failed."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class ModelNotFoundError(LLMError):
    """Model not found or not supported."""
    
    def __init__(self, model: str, vendor: Optional[str] = None):
        message = f"Model '{model}' not found"
        if vendor:
            message += f" for vendor '{vendor}'"
        super().__init__(message)
        self.model = model
        self.vendor = vendor


class ProviderNotFoundError(LLMError):
    """Provider not found or not supported."""
    
    def __init__(self, provider: str):
        super().__init__(f"Provider '{provider}' not found or not supported")
        self.provider = provider


class ConfigurationError(LLMError):
    """Configuration error."""
    
    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")


class CacheError(LLMError):
    """Cache operation error."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(f"Cache error: {message}")
        self.operation = operation


class TemplateError(LLMError):
    """Template rendering error."""
    
    def __init__(self, message: str, template_name: Optional[str] = None):
        super().__init__(f"Template error: {message}")
        self.template_name = template_name