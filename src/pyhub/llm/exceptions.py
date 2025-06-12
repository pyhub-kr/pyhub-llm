# Optional imports for rate limit errors
try:
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    # anthropic이 설치되지 않은 경우 기본 Exception 사용
    class AnthropicRateLimitError(Exception):
        pass


try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    # openai가 설치되지 않은 경우 기본 Exception 사용
    class OpenAIRateLimitError(Exception):
        pass


class RateLimitError(AnthropicRateLimitError, OpenAIRateLimitError):
    """통합 RateLimitError - 설치된 provider의 RateLimitError를 상속"""

    pass


class LLMError(Exception):
    pass


class ValidationError(LLMError):
    """Raised when validation fails"""

    pass
