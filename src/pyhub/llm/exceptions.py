from anthropic import RateLimitError as AnthropicRateLimitError
from openai import RateLimitError as OpenAIRateLimitError


class RateLimitError(AnthropicRateLimitError, OpenAIRateLimitError):
    pass


class LLMError(Exception):
    pass


class ValidationError(LLMError):
    """Raised when validation fails"""

    pass
