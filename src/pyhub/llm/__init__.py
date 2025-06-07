"""PyHub LLM - Standalone LLM library with multiple provider support."""

from .factory import LLM
from .base import BaseLLM
from .types import (
    LLMResponse,
    EmbeddingResponse,
    StreamResponse,
    FunctionCall,
    ToolCall,
    Usage,
)
from .exceptions import (
    LLMError,
    APIError,
    RateLimitError,
    InvalidRequestError,
    AuthenticationError,
)

__version__ = "0.1.0"

__all__ = [
    "LLM",
    "BaseLLM",
    "LLMResponse",
    "EmbeddingResponse",
    "StreamResponse",
    "FunctionCall",
    "ToolCall",
    "Usage",
    "LLMError",
    "APIError",
    "RateLimitError",
    "InvalidRequestError",
    "AuthenticationError",
]