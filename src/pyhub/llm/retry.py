"""Retry and fallback strategies for LLM API calls.

This module provides LangChain-inspired retry functionality with:
- Exponential backoff strategies
- Custom retry conditions
- Fallback LLM instance support
- Pythonic method chaining
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retry logic."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    JITTER = "jitter"  # Exponential with random jitter


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Examples:
        # Basic exponential backoff
        config = RetryConfig(max_retries=3, initial_delay=1.0)

        # Custom retry condition
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            backoff_strategy=BackoffStrategy.JITTER,
            retry_on=[ValueError, ConnectionError]
        )

        # Custom condition function
        def should_retry(error: Exception) -> bool:
            return "rate limit" in str(error).lower()

        config = RetryConfig(
            max_retries=10,
            retry_condition=should_retry
        )
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True

    # Retry conditions
    retry_on: Optional[List[Union[Type[Exception], str]]] = None
    retry_condition: Optional[Callable[[Exception], bool]] = None
    stop_on: Optional[List[Union[Type[Exception], str]]] = None

    # Callbacks
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
    on_failure: Optional[Callable[[Exception, int], None]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.initial_delay <= 0:
            raise ValueError("initial_delay must be > 0")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if self.backoff_multiplier <= 1.0 and self.backoff_strategy != BackoffStrategy.FIXED:
            raise ValueError("backoff_multiplier must be > 1.0 for non-fixed strategies")


@dataclass
class FallbackConfig:
    """Configuration for fallback LLMs.

    Examples:
        # Simple fallback to different LLM instances
        backup_llm = OpenAILLM(model="gpt-4o-mini", temperature=0.1)
        cheap_llm = OpenAILLM(model="gpt-3.5-turbo", temperature=0.1)

        config = FallbackConfig(
            fallback_llms=[backup_llm, cheap_llm]
        )

        # Conditional fallback
        def should_fallback(error: Exception) -> bool:
            return "context length" in str(error).lower()

        config = FallbackConfig(
            fallback_llms=[backup_llm],
            fallback_condition=should_fallback
        )
    """

    fallback_llms: List[Any]  # List[BaseLLM]
    fallback_condition: Optional[Callable[[Exception], bool]] = None

    # Callbacks
    on_fallback: Optional[Callable[[Exception, Any], None]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.fallback_llms:
            raise ValueError("fallback_llms must contain at least one LLM instance")

        # Validate that all items are LLM instances
        from .base import BaseLLM

        for llm in self.fallback_llms:
            if not isinstance(llm, BaseLLM):
                raise ValueError(f"All items in fallback_llms must be BaseLLM instances, got {type(llm)}")


class RetryError(Exception):
    """Exception raised when all retry attempts fail."""

    def __init__(self, message: str, attempts: int, last_error: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class FallbackError(Exception):
    """Exception raised when all fallback attempts fail."""

    def __init__(self, message: str, errors: List[Exception]):
        super().__init__(message)
        self.errors = errors


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt based on backoff strategy."""
    if config.backoff_strategy == BackoffStrategy.FIXED:
        delay = config.initial_delay
    elif config.backoff_strategy == BackoffStrategy.LINEAR:
        delay = config.initial_delay * attempt
    elif config.backoff_strategy in (BackoffStrategy.EXPONENTIAL, BackoffStrategy.JITTER):
        delay = config.initial_delay * (config.backoff_multiplier ** (attempt - 1))
    else:
        delay = config.initial_delay

    # Add jitter if enabled
    if config.jitter or config.backoff_strategy == BackoffStrategy.JITTER:
        jitter_amount = delay * 0.1 * random.random()
        delay += jitter_amount

    # Respect max_delay
    return min(delay, config.max_delay)


def should_retry_error(error: Exception, config: RetryConfig) -> bool:
    """Determine if an error should trigger a retry."""
    # Check stop conditions first
    if config.stop_on:
        for stop_condition in config.stop_on:
            if isinstance(stop_condition, type) and isinstance(error, stop_condition):
                return False
            elif isinstance(stop_condition, str) and stop_condition.lower() in str(error).lower():
                return False

    # Check custom condition
    if config.retry_condition:
        return config.retry_condition(error)

    # Check retry_on conditions
    if config.retry_on:
        for retry_condition in config.retry_on:
            if isinstance(retry_condition, type) and isinstance(error, retry_condition):
                return True
            elif isinstance(retry_condition, str) and retry_condition.lower() in str(error).lower():
                return True
        return False

    # Default: retry on common transient errors
    transient_errors = (
        ConnectionError,
        TimeoutError,
        OSError,  # Network errors
    )

    # Check for rate limit or server errors in message
    error_message = str(error).lower()
    transient_keywords = [
        "rate limit",
        "too many requests",
        "quota exceeded",
        "server error",
        "internal error",
        "service unavailable",
        "timeout",
        "connection",
        "network",
    ]

    return isinstance(error, transient_errors) or any(keyword in error_message for keyword in transient_keywords)


def should_fallback_error(error: Exception, config: FallbackConfig) -> bool:
    """Determine if an error should trigger a fallback."""
    if config.fallback_condition:
        return config.fallback_condition(error)

    # Always fallback on RetryError (all retries exhausted)
    if isinstance(error, RetryError):
        return True

    # Default: fallback on model-specific errors
    fallback_keywords = [
        "context length",
        "token limit",
        "model not found",
        "model unavailable",
        "unsupported",
        "not supported",
    ]

    error_message = str(error).lower()
    return any(keyword in error_message for keyword in fallback_keywords)


class RetryWrapper:
    """Wrapper class that adds retry functionality to LLM instances."""

    def __init__(self, llm: Any, config: RetryConfig):  # llm: BaseLLM
        self.llm = llm
        self.config = config
        # Copy important attributes
        self.model = llm.model

    def __getattr__(self, name):
        """Delegate attribute access to wrapped LLM."""
        return getattr(self.llm, name)

    def with_retry(self, **kwargs):
        """Apply retry to this already-wrapped LLM."""
        from .base import BaseLLM

        # Get the with_retry method from BaseLLM
        return BaseLLM.with_retry(self, **kwargs)

    def with_fallbacks(self, fallback_llms, **kwargs):
        """Apply fallback to this retry-wrapped LLM."""
        from .base import BaseLLM

        # Get the with_fallbacks method from BaseLLM
        return BaseLLM.with_fallbacks(self, fallback_llms, **kwargs)

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
                        f"All {self.config.max_retries + 1} attempts failed. Last error: {error}", attempt + 1, error
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
                        f"All {self.config.max_retries + 1} attempts failed. Last error: {error}", attempt + 1, error
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

    # Override key methods with retry logic
    def ask(self, *args, **kwargs):
        """Ask with retry logic."""
        # Always raise errors for retry logic to work
        kwargs["raise_errors"] = True
        return self._retry_sync_call(self.llm.ask, *args, **kwargs)

    async def ask_async(self, *args, **kwargs):
        """Ask async with retry logic."""
        # Always raise errors for retry logic to work
        kwargs["raise_errors"] = True
        return await self._retry_async_call(self.llm.ask_async, *args, **kwargs)

    def embed(self, *args, **kwargs):
        """Embed with retry logic."""
        return self._retry_sync_call(self.llm.embed, *args, **kwargs)

    async def embed_async(self, *args, **kwargs):
        """Embed async with retry logic."""
        return await self._retry_async_call(self.llm.embed_async, *args, **kwargs)

    def generate_image(self, *args, **kwargs):
        """Generate image with retry logic."""
        return self._retry_sync_call(self.llm.generate_image, *args, **kwargs)

    async def generate_image_async(self, *args, **kwargs):
        """Generate image async with retry logic."""
        return await self._retry_async_call(self.llm.generate_image_async, *args, **kwargs)


class FallbackWrapper:
    """Wrapper class that adds fallback functionality to LLM instances."""

    def __init__(self, llm: Any, config: FallbackConfig):  # llm: BaseLLM
        self.llm = llm
        self.config = config
        # Copy important attributes
        self.model = llm.model

    def __getattr__(self, name):
        """Delegate attribute access to wrapped LLM."""
        return getattr(self.llm, name)

    def with_retry(self, **kwargs):
        """Apply retry to this fallback-wrapped LLM."""
        from .base import BaseLLM

        # Get the with_retry method from BaseLLM
        return BaseLLM.with_retry(self, **kwargs)

    def with_fallbacks(self, fallback_llms, **kwargs):
        """Apply additional fallbacks to this already-wrapped LLM."""
        from .base import BaseLLM

        # Get the with_fallbacks method from BaseLLM
        return BaseLLM.with_fallbacks(self, fallback_llms, **kwargs)

    async def _fallback_async_call(self, method_name: str, *args, **kwargs):
        """Execute async method with fallback logic."""
        errors = []

        # Try primary LLM first
        try:
            method = getattr(self.llm, method_name)
            # Ensure raise_errors is True for ask_async method
            if method_name == "ask_async" and "raise_errors" not in kwargs:
                kwargs["raise_errors"] = True
            return await method(*args, **kwargs)
        except Exception as error:
            errors.append(error)

            if not should_fallback_error(error, self.config):
                raise error

            if self.config.on_fallback:
                self.config.on_fallback(error, self.llm)

        # Try fallback LLMs
        for i, fallback_llm in enumerate(self.config.fallback_llms):
            try:
                method = getattr(fallback_llm, method_name)
                logger.warning(
                    f"Primary LLM ({self.llm.model}) failed, trying fallback {i+1}/{len(self.config.fallback_llms)}: {fallback_llm.model}"
                )
                # Ensure raise_errors is True for ask_async method
                if method_name == "ask_async" and "raise_errors" not in kwargs:
                    kwargs["raise_errors"] = True
                return await method(*args, **kwargs)
            except Exception as error:
                errors.append(error)

                if self.config.on_fallback:
                    self.config.on_fallback(error, fallback_llm)

                logger.warning(f"Fallback {i+1} ({fallback_llm.model}) failed: {error}")

        # All fallbacks failed
        raise FallbackError(f"Primary LLM and all {len(self.config.fallback_llms)} fallbacks failed", errors)

    def _fallback_sync_call(self, method_name: str, *args, **kwargs):
        """Execute sync method with fallback logic."""
        errors = []

        # Try primary LLM first
        try:
            method = getattr(self.llm, method_name)
            # Ensure raise_errors is True for ask method
            if method_name == "ask" and "raise_errors" not in kwargs:
                kwargs["raise_errors"] = True
            return method(*args, **kwargs)
        except Exception as error:
            errors.append(error)

            if not should_fallback_error(error, self.config):
                raise error

            if self.config.on_fallback:
                self.config.on_fallback(error, self.llm)

        # Try fallback LLMs
        for i, fallback_llm in enumerate(self.config.fallback_llms):
            try:
                method = getattr(fallback_llm, method_name)
                logger.warning(
                    f"Primary LLM ({self.llm.model}) failed, trying fallback {i+1}/{len(self.config.fallback_llms)}: {fallback_llm.model}"
                )
                # Ensure raise_errors is True for ask method
                if method_name == "ask" and "raise_errors" not in kwargs:
                    kwargs["raise_errors"] = True
                return method(*args, **kwargs)
            except Exception as error:
                errors.append(error)

                if self.config.on_fallback:
                    self.config.on_fallback(error, fallback_llm)

                logger.warning(f"Fallback {i+1} ({fallback_llm.model}) failed: {error}")

        # All fallbacks failed
        raise FallbackError(f"Primary LLM and all {len(self.config.fallback_llms)} fallbacks failed", errors)

    # Override key methods with fallback logic
    def ask(self, *args, **kwargs):
        """Ask with fallback logic."""
        # Always raise errors for fallback logic to work
        kwargs["raise_errors"] = True
        return self._fallback_sync_call("ask", *args, **kwargs)

    async def ask_async(self, *args, **kwargs):
        """Ask async with fallback logic."""
        # Always raise errors for fallback logic to work
        kwargs["raise_errors"] = True
        return await self._fallback_async_call("ask_async", *args, **kwargs)

    def embed(self, *args, **kwargs):
        """Embed with fallback logic."""
        return self._fallback_sync_call("embed", *args, **kwargs)

    async def embed_async(self, *args, **kwargs):
        """Embed async with fallback logic."""
        return await self._fallback_async_call("embed_async", *args, **kwargs)

    def generate_image(self, *args, **kwargs):
        """Generate image with fallback logic."""
        return self._fallback_sync_call("generate_image", *args, **kwargs)

    async def generate_image_async(self, *args, **kwargs):
        """Generate image async with fallback logic."""
        return await self._fallback_async_call("generate_image_async", *args, **kwargs)


class RetryFallbackWrapper:
    """Wrapper that combines both retry and fallback functionality."""

    def __init__(self, llm: Any, retry_config: RetryConfig, fallback_config: FallbackConfig):
        # First wrap with retry, then with fallback
        self.retry_llm = RetryWrapper(llm, retry_config)
        self.fallback_llm = FallbackWrapper(self.retry_llm, fallback_config)

    def __getattr__(self, name):
        """Delegate attribute access to wrapped LLM."""
        return getattr(self.fallback_llm, name)
