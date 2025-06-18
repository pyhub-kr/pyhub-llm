"""Retry and fallback functionality for LLM instances."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Union

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retry delays."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    JITTER = "jitter"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = False
    retry_on: Optional[List[Union[str, type]]] = None
    stop_on: Optional[List[Union[str, type]]] = None
    retry_condition: Optional[Callable[[Exception], bool]] = None
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
    on_failure: Optional[Callable[[Exception, int], None]] = None

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.initial_delay <= 0:
            raise ValueError("initial_delay must be positive")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        if self.backoff_factor <= 0:
            raise ValueError("backoff_factor must be positive")

        # Set default retry_on conditions if not specified
        if self.retry_on is None:
            self.retry_on = [
                # Common API errors
                ConnectionError,
                TimeoutError,
                # String patterns for common errors
                "rate limit",
                "too many requests",
                "429",
                "503",
                "504",
                "timeout",
                "connection",
            ]


@dataclass
class FallbackConfig:
    """Configuration for fallback LLMs."""

    fallback_llms: List[Any]  # List[BaseLLM]
    fallback_condition: Optional[Callable[[Exception], bool]] = None
    on_fallback: Optional[Callable[[Exception, Any], None]] = None

    def __post_init__(self):
        """Validate configuration values."""
        if not self.fallback_llms:
            raise ValueError("At least one fallback LLM must be provided")


class RetryError(Exception):
    """Exception raised when all retry attempts fail."""

    def __init__(self, message: str, attempts: int, last_error: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class FallbackError(Exception):
    """Exception raised when all fallback LLMs fail."""

    def __init__(self, message: str, errors: List[Exception]):
        super().__init__(message)
        self.errors = errors


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt based on backoff strategy."""
    if config.backoff_strategy == BackoffStrategy.FIXED:
        delay = config.initial_delay
    elif config.backoff_strategy == BackoffStrategy.LINEAR:
        delay = config.initial_delay * attempt
    elif config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
        delay = config.initial_delay * (config.backoff_factor ** (attempt - 1))
    elif config.backoff_strategy == BackoffStrategy.JITTER:
        # Exponential with full jitter
        max_delay = config.initial_delay * (config.backoff_factor ** (attempt - 1))
        delay = random.uniform(0, max_delay)
    else:
        delay = config.initial_delay

    # Apply jitter if enabled (except for JITTER and FIXED strategies)
    if config.jitter and config.backoff_strategy not in (BackoffStrategy.JITTER, BackoffStrategy.FIXED):
        jitter_factor = 0.1  # 10% jitter
        jitter = delay * jitter_factor * (2 * random.random() - 1)
        delay += jitter

    # Cap at max_delay
    return min(delay, config.max_delay)


def should_retry_error(error: Exception, config: RetryConfig) -> bool:
    """Determine if an error should trigger a retry."""
    # Check custom retry condition first
    if config.retry_condition:
        return config.retry_condition(error)

    # Check stop conditions
    if config.stop_on:
        for stop_condition in config.stop_on:
            if isinstance(stop_condition, type) and isinstance(error, stop_condition):
                return False
            elif isinstance(stop_condition, str) and stop_condition.lower() in str(error).lower():
                return False

    # Check retry conditions
    if config.retry_on:
        for retry_condition in config.retry_on:
            if isinstance(retry_condition, type) and isinstance(error, retry_condition):
                return True
            elif isinstance(retry_condition, str) and retry_condition.lower() in str(error).lower():
                return True
        # If retry_on is specified but no match, don't retry
        return False

    # Default: retry on any error
    return True


def should_fallback_error(error: Exception, config: FallbackConfig) -> bool:
    """Determine if an error should trigger fallback."""
    # Always fallback on RetryError (exhausted retries)
    if isinstance(error, RetryError):
        return True

    # Check custom fallback condition
    if config.fallback_condition:
        return config.fallback_condition(error)

    # Default fallback conditions
    fallback_keywords = [
        "context length",
        "maximum context",
        "token limit",
        "model not found",
        "invalid model",
        "not supported",
        "insufficient quota",
        "invalid api key",
        "unauthorized",
        "forbidden",
    ]

    error_message = str(error).lower()
    return any(keyword in error_message for keyword in fallback_keywords)


def create_dynamic_method(method_name: str, is_async: bool = False):
    """Create a dynamic method that applies retry/fallback logic."""
    if is_async:

        async def async_method(self, *args, **kwargs):
            if hasattr(self, "_retry_async_call"):
                # This is a RetryWrapper
                kwargs["raise_errors"] = True
                return await self._retry_async_call(getattr(self.llm, method_name), *args, **kwargs)
            else:
                # This is a FallbackWrapper
                return await self._fallback_async_call(method_name, *args, **kwargs)

        return async_method
    else:

        def sync_method(self, *args, **kwargs):
            if hasattr(self, "_retry_sync_call"):
                # This is a RetryWrapper
                kwargs["raise_errors"] = True
                return self._retry_sync_call(getattr(self.llm, method_name), *args, **kwargs)
            else:
                # This is a FallbackWrapper
                return self._fallback_sync_call(method_name, *args, **kwargs)

        return sync_method


class RetryWrapper:
    """Wrapper class that adds retry functionality to LLM instances."""

    def __init__(self, llm: Any, config: RetryConfig):  # llm: BaseLLM
        # Don't call super().__init__() to avoid BaseLLM initialization
        self.llm = llm
        self.config = config

        # Copy essential attributes from wrapped LLM
        self.model = llm.model
        self.api_key = getattr(llm, "api_key", None)
        self.base_url = getattr(llm, "base_url", None)

        # Preserve LLM state
        self._history = getattr(llm, "_history", [])
        self._stateless = getattr(llm, "_stateless", False)
        self._cache = getattr(llm, "_cache", None)

    def __getattr__(self, name):
        """Delegate attribute access to wrapped LLM."""
        return getattr(self.llm, name)

    async def _retry_async_call(self, coro_func, *args, **kwargs):
        """Execute async function with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as error:
                last_error = error

                if attempt == self.config.max_retries:
                    # Final attempt failed
                    if self.config.on_failure:
                        self.config.on_failure(error, attempt + 1)
                    raise RetryError(
                        f"All {self.config.max_retries + 1} attempts failed for {self.llm.model}. "
                        f"Last error: {type(error).__name__}: {error}",
                        attempt + 1,
                        error,
                    )

                if not should_retry_error(error, self.config):
                    # Error is not retryable
                    raise error

                # Calculate delay and wait
                delay = calculate_delay(attempt + 1, self.config)

                if self.config.on_retry:
                    self.config.on_retry(error, attempt + 1, delay)

                logger.warning(
                    f"Attempt {attempt + 1} failed with {type(error).__name__}: {error}. "
                    f"Retrying in {delay:.2f}s..."
                )

                await asyncio.sleep(delay)

        # This should never be reached
        raise last_error

    def _retry_sync_call(self, func, *args, **kwargs):
        """Execute sync function with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                last_error = error

                if attempt == self.config.max_retries:
                    # Final attempt failed
                    if self.config.on_failure:
                        self.config.on_failure(error, attempt + 1)
                    raise RetryError(
                        f"All {self.config.max_retries + 1} attempts failed for {self.llm.model}. "
                        f"Last error: {type(error).__name__}: {error}",
                        attempt + 1,
                        error,
                    )

                if not should_retry_error(error, self.config):
                    # Error is not retryable
                    raise error

                # Calculate delay and wait
                delay = calculate_delay(attempt + 1, self.config)

                if self.config.on_retry:
                    self.config.on_retry(error, attempt + 1, delay)

                logger.warning(
                    f"Attempt {attempt + 1} failed with {type(error).__name__}: {error}. "
                    f"Retrying in {delay:.2f}s..."
                )

                time.sleep(delay)

        # This should never be reached
        raise last_error


class FallbackWrapper:
    """Wrapper class that adds fallback functionality to LLM instances."""

    def __init__(self, llm: Any, config: FallbackConfig):  # llm: BaseLLM
        # Don't call super().__init__() to avoid BaseLLM initialization
        self.llm = llm
        self.config = config
        self.llm_chain = [llm] + config.fallback_llms

        # Copy essential attributes from wrapped LLM
        self.model = llm.model
        self.api_key = getattr(llm, "api_key", None)
        self.base_url = getattr(llm, "base_url", None)

        # Preserve LLM state
        self._history = getattr(llm, "_history", [])
        self._stateless = getattr(llm, "_stateless", False)
        self._cache = getattr(llm, "_cache", None)

    def __getattr__(self, name):
        """Delegate attribute access to wrapped LLM."""
        return getattr(self.llm, name)

    async def _fallback_async_call(self, method_name: str, *args, **kwargs):
        """Execute method with fallback logic asynchronously."""
        errors = []

        for i, llm in enumerate(self.llm_chain):
            try:
                method = getattr(llm, method_name)
                kwargs["raise_errors"] = True
                return await method(*args, **kwargs)
            except Exception as error:
                errors.append(error)

                if i == len(self.llm_chain) - 1:
                    # All LLMs failed
                    models = [llm.model for llm in self.llm_chain]
                    error_details = [f"{models[i]}: {type(e).__name__}: {e}" for i, e in enumerate(errors)]
                    raise FallbackError(
                        f"All {len(self.llm_chain)} LLMs failed. Details:\n"
                        + "\n".join(f"  - {detail}" for detail in error_details),
                        errors,
                    )

                if not should_fallback_error(error, self.config):
                    # Error should not trigger fallback
                    raise error

                # Log fallback
                if self.config.on_fallback:
                    self.config.on_fallback(error, self.llm_chain[i + 1])

                logger.warning(
                    f"LLM {i + 1}/{len(self.llm_chain)} failed with {type(error).__name__}: {error}. "
                    f"Falling back to next LLM..."
                )

    def _fallback_sync_call(self, method_name: str, *args, **kwargs):
        """Execute method with fallback logic synchronously."""
        errors = []

        for i, llm in enumerate(self.llm_chain):
            try:
                method = getattr(llm, method_name)
                kwargs["raise_errors"] = True
                return method(*args, **kwargs)
            except Exception as error:
                errors.append(error)

                if i == len(self.llm_chain) - 1:
                    # All LLMs failed
                    models = [llm.model for llm in self.llm_chain]
                    error_details = [f"{models[i]}: {type(e).__name__}: {e}" for i, e in enumerate(errors)]
                    raise FallbackError(
                        f"All {len(self.llm_chain)} LLMs failed. Details:\n"
                        + "\n".join(f"  - {detail}" for detail in error_details),
                        errors,
                    )

                if not should_fallback_error(error, self.config):
                    # Error should not trigger fallback
                    raise error

                # Log fallback
                if self.config.on_fallback:
                    self.config.on_fallback(error, self.llm_chain[i + 1])

                logger.warning(
                    f"LLM {i + 1}/{len(self.llm_chain)} failed with {type(error).__name__}: {error}. "
                    f"Falling back to next LLM..."
                )


# Define method names to wrap dynamically
WRAPPED_METHODS = {
    "ask": False,
    "ask_async": True,
    "messages": False,
    "messages_async": True,
    "embed": False,
    "embed_async": True,
    "generate_image": False,
    "generate_image_async": True,
    "ask_with_json": False,
    "ask_with_json_async": True,
    "ask_with_tools": False,
    "ask_with_tools_async": True,
}

# Add dynamic methods to wrappers
for method_name, is_async in WRAPPED_METHODS.items():
    dynamic_method = create_dynamic_method(method_name, is_async)
    setattr(RetryWrapper, method_name, dynamic_method)
    setattr(FallbackWrapper, method_name, dynamic_method)


# Add with_retry and with_fallbacks methods to preserve chaining capability
def with_retry(self, **kwargs):
    """Apply retry to this already-wrapped LLM."""
    # Get the with_retry method from the wrapped LLM
    from .base import BaseLLM

    # Create a new wrapper that wraps this one
    return BaseLLM.with_retry(self, **kwargs)


def with_fallbacks(self, fallback_llms, **kwargs):
    """Apply fallback to this already-wrapped LLM."""
    # Get the with_fallbacks method from the wrapped LLM
    from .base import BaseLLM

    # Create a new wrapper that wraps this one
    return BaseLLM.with_fallbacks(self, fallback_llms, **kwargs)


# Add these methods to both wrapper classes
setattr(RetryWrapper, "with_retry", with_retry)
setattr(RetryWrapper, "with_fallbacks", with_fallbacks)
setattr(FallbackWrapper, "with_retry", with_retry)
setattr(FallbackWrapper, "with_fallbacks", with_fallbacks)
