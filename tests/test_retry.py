"""Tests for retry and fallback functionality."""

import os
import random
import sys
from unittest.mock import Mock, patch

import pytest

from pyhub.llm import OpenAILLM
from pyhub.llm.base import BaseLLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pyhub.llm.retry import (
    BackoffStrategy,
    FallbackConfig,
    FallbackError,
    FallbackWrapper,
    RetryConfig,
    RetryError,
    RetryWrapper,
    calculate_delay,
    should_fallback_error,
    should_retry_error,
)
from pyhub.llm.types import Reply


class MockLLM(BaseLLM):
    """Mock LLM for testing."""

    def __init__(self, model="mock-model", **kwargs):
        super().__init__(model=model, **kwargs)
        self.call_count = 0
        self.should_fail = False
        self.failure_exception = ConnectionError("Mock connection error")

    def _make_request_params(self, *args, **kwargs):
        return {}

    def _make_ask(self, *args, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise self.failure_exception
        return Reply(text=f"Response from {self.model}")

    async def _make_ask_async(self, *args, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise self.failure_exception
        return Reply(text=f"Response from {self.model}")

    def _make_ask_stream(self, *args, **kwargs):
        yield Reply(text="Stream")

    async def _make_ask_stream_async(self, *args, **kwargs):
        yield Reply(text="Stream")

    def embed(self, *args, **kwargs):
        if self.should_fail:
            raise self.failure_exception
        return Mock()

    async def embed_async(self, *args, **kwargs):
        if self.should_fail:
            raise self.failure_exception
        return Mock()

    def generate_image(self, *args, **kwargs):
        if self.should_fail:
            raise self.failure_exception
        return Mock()

    async def generate_image_async(self, *args, **kwargs):
        if self.should_fail:
            raise self.failure_exception
        return Mock()


class TestRetryConfig:
    """Test RetryConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert config.jitter is False

    def test_validation(self):
        """Test configuration validation."""
        # Negative retries
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RetryConfig(max_retries=-1)

        # Zero or negative delay
        with pytest.raises(ValueError, match="initial_delay must be positive"):
            RetryConfig(initial_delay=0)

        # max_delay less than initial_delay
        with pytest.raises(ValueError, match="max_delay must be positive"):
            RetryConfig(max_delay=0)

        # Invalid backoff multiplier
        with pytest.raises(ValueError, match="backoff_factor must be positive"):
            RetryConfig(backoff_factor=0.0, backoff_strategy=BackoffStrategy.EXPONENTIAL)


class TestFallbackConfig:
    """Test FallbackConfig class."""

    def test_validation(self):
        """Test configuration validation."""
        # Empty fallback list
        with pytest.raises(ValueError, match="At least one fallback LLM must be provided"):
            FallbackConfig(fallback_llms=[])

    def test_valid_config(self):
        """Test valid configuration."""
        llm1 = MockLLM(model="backup1")
        llm2 = MockLLM(model="backup2")
        config = FallbackConfig(fallback_llms=[llm1, llm2])
        assert len(config.fallback_llms) == 2


class TestDelayCalculation:
    """Test delay calculation functions."""

    def test_fixed_delay(self):
        """Test fixed delay strategy."""
        config = RetryConfig(initial_delay=5.0, backoff_strategy=BackoffStrategy.FIXED, jitter=False)
        assert calculate_delay(1, config) == 5.0
        assert calculate_delay(3, config) == 5.0
        assert calculate_delay(10, config) == 5.0

    def test_linear_delay(self):
        """Test linear delay strategy."""
        config = RetryConfig(initial_delay=2.0, backoff_strategy=BackoffStrategy.LINEAR, jitter=False)
        assert calculate_delay(1, config) == 2.0
        assert calculate_delay(2, config) == 4.0
        assert calculate_delay(3, config) == 6.0

    def test_exponential_delay(self):
        """Test exponential delay strategy."""
        config = RetryConfig(
            initial_delay=1.0, backoff_factor=2.0, backoff_strategy=BackoffStrategy.EXPONENTIAL, jitter=False
        )
        assert calculate_delay(1, config) == 1.0
        assert calculate_delay(2, config) == 2.0
        assert calculate_delay(3, config) == 4.0
        assert calculate_delay(4, config) == 8.0

    def test_max_delay_limit(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            initial_delay=10.0,
            max_delay=20.0,
            backoff_factor=3.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            jitter=False,
        )
        assert calculate_delay(1, config) == 10.0
        assert calculate_delay(2, config) == 20.0  # Would be 30, but capped
        assert calculate_delay(3, config) == 20.0  # Would be 90, but capped

    def test_jitter(self):
        """Test that jitter adds randomness."""
        # Set seed to ensure deterministic test behavior
        random.seed(42)
        config = RetryConfig(initial_delay=10.0, jitter=True)
        # Run multiple times to ensure we get different values
        delays = [calculate_delay(1, config) for _ in range(10)]
        assert len(set(delays)) > 1  # Should have different values
        assert all(9.0 <= d <= 11.0 for d in delays)  # Within ±10% jitter
        # Reset seed to avoid affecting other tests
        random.seed()


class TestRetryConditions:
    """Test retry condition functions."""

    def test_default_retry_conditions(self):
        """Test default retry behavior."""
        config = RetryConfig()

        # Should retry on common transient errors
        assert should_retry_error(ConnectionError("Network error"), config)
        assert should_retry_error(TimeoutError("Timeout"), config)
        assert should_retry_error(Exception("rate limit exceeded"), config)
        assert should_retry_error(Exception("Too many requests"), config)

        # Should not retry on other errors by default
        assert not should_retry_error(ValueError("Invalid value"), config)
        assert not should_retry_error(TypeError("Type error"), config)

    def test_retry_on_specific_exceptions(self):
        """Test retry_on configuration."""
        config = RetryConfig(retry_on=[ValueError, "specific error"])

        assert should_retry_error(ValueError("Test"), config)
        assert should_retry_error(Exception("specific error occurred"), config)
        assert not should_retry_error(TypeError("Other error"), config)

    def test_stop_on_exceptions(self):
        """Test stop_on configuration."""
        config = RetryConfig(retry_on=[Exception], stop_on=[ValueError, "fatal"])  # Retry all exceptions

        assert not should_retry_error(ValueError("Stop this"), config)
        assert not should_retry_error(Exception("Fatal error"), config)
        assert should_retry_error(TypeError("Retry this"), config)

    def test_custom_retry_condition(self):
        """Test custom retry condition function."""

        def custom_condition(error):
            return "retry me" in str(error).lower()

        config = RetryConfig(retry_condition=custom_condition)

        assert should_retry_error(Exception("Please retry me"), config)
        assert not should_retry_error(Exception("Don't retry"), config)


class TestFallbackConditions:
    """Test fallback condition functions."""

    def test_default_fallback_conditions(self):
        """Test default fallback behavior."""
        config = FallbackConfig(fallback_llms=[MockLLM()])

        # Should fallback on model-specific errors
        assert should_fallback_error(Exception("context length exceeded"), config)
        assert should_fallback_error(Exception("Token limit reached"), config)
        assert should_fallback_error(Exception("Model not found"), config)

        # Should not fallback on other errors
        assert not should_fallback_error(ConnectionError("Network error"), config)
        assert not should_fallback_error(ValueError("Invalid value"), config)

    def test_custom_fallback_condition(self):
        """Test custom fallback condition function."""

        def custom_condition(error):
            return isinstance(error, ValueError)

        config = FallbackConfig(fallback_llms=[MockLLM()], fallback_condition=custom_condition)

        assert should_fallback_error(ValueError("Test"), config)
        assert not should_fallback_error(TypeError("Test"), config)


class TestRetryWrapper:
    """Test RetryWrapper functionality."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retries."""
        llm = MockLLM()
        config = RetryConfig(max_retries=3)
        wrapper = RetryWrapper(llm, config)

        result = wrapper.ask("Test")
        assert result.text == "Response from mock-model"
        assert llm.call_count == 1

    def test_retry_on_failure(self):
        """Test retry on failure."""
        llm = MockLLM()
        llm.should_fail = True

        config = RetryConfig(max_retries=2, initial_delay=0.01, retry_on=[ConnectionError])  # Short delay for testing
        wrapper = RetryWrapper(llm, config)

        with pytest.raises(RetryError) as exc_info:
            wrapper.ask("Test")

        assert llm.call_count == 3  # Initial + 2 retries
        assert "All 3 attempts failed" in str(exc_info.value)

    def test_retry_with_recovery(self):
        """Test successful retry after failures."""
        llm = MockLLM()

        # Fail first 2 attempts, succeed on 3rd
        call_count = 0
        original_ask = llm._make_ask

        def mock_ask(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Mock failure")
            return original_ask(*args, **kwargs)

        llm._make_ask = mock_ask

        config = RetryConfig(max_retries=3, initial_delay=0.01, retry_on=[ConnectionError])
        wrapper = RetryWrapper(llm, config)

        result = wrapper.ask("Test")
        assert result.text == "Response from mock-model"
        assert call_count == 3

    def test_retry_callbacks(self):
        """Test retry callback functions."""
        retry_calls = []
        failure_calls = []

        def on_retry(error, attempt, delay):
            retry_calls.append((str(error), attempt, delay))

        def on_failure(error, attempts):
            failure_calls.append((str(error), attempts))

        llm = MockLLM()
        llm.should_fail = True

        config = RetryConfig(max_retries=2, initial_delay=0.01, on_retry=on_retry, on_failure=on_failure)
        wrapper = RetryWrapper(llm, config)

        with pytest.raises(RetryError):
            wrapper.ask("Test")

        assert len(retry_calls) == 2
        assert retry_calls[0][1] == 1  # First retry attempt
        assert retry_calls[1][1] == 2  # Second retry attempt

        assert len(failure_calls) == 1
        assert failure_calls[0][1] == 3  # Total attempts

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test async retry functionality."""
        llm = MockLLM()
        llm.should_fail = True

        config = RetryConfig(max_retries=2, initial_delay=0.01)
        wrapper = RetryWrapper(llm, config)

        with pytest.raises(RetryError):
            await wrapper.ask_async("Test")

        assert llm.call_count == 3

    def test_other_methods_wrapped(self):
        """Test that other methods are also wrapped with retry."""
        llm = MockLLM()
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        wrapper = RetryWrapper(llm, config)

        # Test embed
        llm.should_fail = False
        wrapper.embed("test")

        # Test generate_image
        wrapper.generate_image("test prompt")

        # Verify methods were called
        assert hasattr(wrapper, "embed")
        assert hasattr(wrapper, "generate_image")


class TestFallbackWrapper:
    """Test FallbackWrapper functionality."""

    def test_primary_success_no_fallback(self):
        """Test that fallbacks aren't used when primary succeeds."""
        primary = MockLLM(model="primary")
        backup1 = MockLLM(model="backup1")
        backup2 = MockLLM(model="backup2")

        config = FallbackConfig(fallback_llms=[backup1, backup2])
        wrapper = FallbackWrapper(primary, config)

        result = wrapper.ask("Test")
        assert result.text == "Response from primary"
        assert primary.call_count == 1
        assert backup1.call_count == 0
        assert backup2.call_count == 0

    def test_fallback_on_failure(self):
        """Test fallback to backup LLMs."""
        primary = MockLLM(model="primary")
        primary.should_fail = True
        primary.failure_exception = Exception("context length exceeded")

        backup1 = MockLLM(model="backup1")
        backup2 = MockLLM(model="backup2")

        config = FallbackConfig(fallback_llms=[backup1, backup2])
        wrapper = FallbackWrapper(primary, config)

        result = wrapper.ask("Test")
        assert result.text == "Response from backup1"
        assert primary.call_count == 1
        assert backup1.call_count == 1
        assert backup2.call_count == 0

    def test_multiple_fallbacks(self):
        """Test fallback through multiple backup LLMs."""
        primary = MockLLM(model="primary")
        primary.should_fail = True
        primary.failure_exception = Exception("model not found")

        backup1 = MockLLM(model="backup1")
        backup1.should_fail = True
        backup1.failure_exception = Exception("model not found")

        backup2 = MockLLM(model="backup2")

        config = FallbackConfig(fallback_llms=[backup1, backup2])
        wrapper = FallbackWrapper(primary, config)

        result = wrapper.ask("Test")
        assert result.text == "Response from backup2"
        assert primary.call_count == 1
        assert backup1.call_count == 1
        assert backup2.call_count == 1

    def test_all_fallbacks_fail(self):
        """Test when all fallbacks fail."""
        primary = MockLLM(model="primary")
        primary.should_fail = True
        primary.failure_exception = Exception("model not found")

        backup1 = MockLLM(model="backup1")
        backup1.should_fail = True
        backup1.failure_exception = Exception("model not found")

        backup2 = MockLLM(model="backup2")
        backup2.should_fail = True
        backup2.failure_exception = Exception("model not found")

        config = FallbackConfig(fallback_llms=[backup1, backup2])
        wrapper = FallbackWrapper(primary, config)

        with pytest.raises(FallbackError) as exc_info:
            wrapper.ask("Test")

        assert "All 3 LLMs failed" in str(exc_info.value)
        assert len(exc_info.value.errors) == 3

    def test_fallback_callback(self):
        """Test fallback callback function."""
        fallback_calls = []

        def on_fallback(error, llm):
            fallback_calls.append((str(error), llm.model))

        primary = MockLLM(model="primary")
        primary.should_fail = True
        primary.failure_exception = Exception("model not found")

        backup1 = MockLLM(model="backup1")

        config = FallbackConfig(fallback_llms=[backup1], on_fallback=on_fallback)
        wrapper = FallbackWrapper(primary, config)

        result = wrapper.ask("Test")
        assert result.text == "Response from backup1"

        assert len(fallback_calls) == 1
        assert "model not found" in fallback_calls[0][0]
        assert fallback_calls[0][1] == "backup1"

    def test_conditional_fallback(self):
        """Test conditional fallback logic."""

        def should_fallback(error):
            return "special" in str(error).lower()

        primary = MockLLM(model="primary")
        backup = MockLLM(model="backup")

        config = FallbackConfig(fallback_llms=[backup], fallback_condition=should_fallback)
        wrapper = FallbackWrapper(primary, config)

        # Error that shouldn't trigger fallback
        primary.should_fail = True
        primary.failure_exception = ValueError("Regular error")

        with pytest.raises(ValueError):
            wrapper.ask("Test")

        assert primary.call_count == 1
        assert backup.call_count == 0

        # Error that should trigger fallback
        primary.call_count = 0  # Reset call count
        primary.failure_exception = Exception("Special error case")

        result = wrapper.ask("Test")
        assert result.text == "Response from backup"
        assert backup.call_count == 1

    @pytest.mark.asyncio
    async def test_async_fallback(self):
        """Test async fallback functionality."""
        primary = MockLLM(model="primary")
        primary.should_fail = True
        primary.failure_exception = Exception("model not found")

        backup = MockLLM(model="backup")

        config = FallbackConfig(fallback_llms=[backup])
        wrapper = FallbackWrapper(primary, config)

        result = await wrapper.ask_async("Test")
        assert result.text == "Response from backup"


class TestBaseLLMIntegration:
    """Test integration with BaseLLM class."""

    def test_with_retry_method(self):
        """Test BaseLLM.with_retry() method."""
        llm = MockLLM()

        retry_llm = llm.with_retry(max_retries=5, initial_delay=0.5, backoff_strategy="linear")

        assert isinstance(retry_llm, RetryWrapper)
        assert retry_llm.config.max_retries == 5
        assert retry_llm.config.initial_delay == 0.5
        assert retry_llm.config.backoff_strategy == BackoffStrategy.LINEAR

    def test_with_fallbacks_method(self):
        """Test BaseLLM.with_fallbacks() method."""
        primary = MockLLM(model="primary")
        backup1 = MockLLM(model="backup1")
        backup2 = MockLLM(model="backup2")

        fallback_llm = primary.with_fallbacks([backup1, backup2])

        assert isinstance(fallback_llm, FallbackWrapper)
        assert len(fallback_llm.config.fallback_llms) == 2

    def test_chaining_retry_and_fallback(self):
        """Test chaining with_retry() and with_fallbacks()."""
        primary = MockLLM(model="primary")
        backup = MockLLM(model="backup")

        # Chain retry and fallback
        llm = primary.with_retry(max_retries=2).with_fallbacks([backup])

        # Should be FallbackWrapper wrapping RetryWrapper
        assert isinstance(llm, FallbackWrapper)
        # The llm attribute should be the RetryWrapper
        assert hasattr(llm, "llm")
        # Check that it's the retry wrapper by checking for config attribute
        assert hasattr(llm.llm, "config") and isinstance(llm.llm.config, RetryConfig)

    def test_real_openai_integration(self):
        """Test with real OpenAI LLM class (structure only, no API calls)."""
        # This test verifies the interface works with real LLM classes
        llm = OpenAILLM(model="gpt-4o-mini", api_key="test-key")

        # Should be able to add retry
        retry_llm = llm.with_retry(max_retries=3)
        assert isinstance(retry_llm, RetryWrapper)

        # Should be able to add fallbacks
        backup = OpenAILLM(model="gpt-3.5-turbo", api_key="test-key")
        fallback_llm = llm.with_fallbacks([backup])
        assert isinstance(fallback_llm, FallbackWrapper)

        # Should be able to chain
        combined = llm.with_retry(max_retries=2).with_fallbacks([backup])
        assert isinstance(combined, FallbackWrapper)


class TestErrorHandling:
    """Test error handling edge cases."""

    def test_non_retryable_error(self):
        """Test that non-retryable errors fail immediately."""
        llm = MockLLM()
        llm.should_fail = True
        llm.failure_exception = ValueError("Bad value")

        config = RetryConfig(max_retries=3, retry_on=[ConnectionError])  # Only retry connection errors
        wrapper = RetryWrapper(llm, config)

        with pytest.raises(ValueError):
            wrapper.ask("Test")

        # Should only try once
        assert llm.call_count == 1

    def test_stop_on_specific_error(self):
        """Test stop_on configuration."""
        llm = MockLLM()
        llm.should_fail = True
        llm.failure_exception = Exception("FATAL ERROR")

        config = RetryConfig(
            max_retries=3, retry_on=[Exception], stop_on=["FATAL"]  # Retry all exceptions  # But stop on fatal errors
        )
        wrapper = RetryWrapper(llm, config)

        with pytest.raises(Exception) as exc_info:
            wrapper.ask("Test")

        assert "FATAL ERROR" in str(exc_info.value)
        assert llm.call_count == 1  # No retries


class TestLogging:
    """Test logging functionality."""

    @patch("pyhub.llm.retry.logger")
    def test_retry_logging(self, mock_logger):
        """Test that retries are logged."""
        llm = MockLLM()
        llm.should_fail = True

        config = RetryConfig(max_retries=1, initial_delay=0.01)
        wrapper = RetryWrapper(llm, config)

        with pytest.raises(RetryError):
            wrapper.ask("Test")

        # Check that warning was logged
        assert mock_logger.warning.called
        warning_calls = mock_logger.warning.call_args_list
        assert any("Attempt 1 failed" in str(call) for call in warning_calls)

    @patch("pyhub.llm.retry.logger")
    def test_fallback_logging(self, mock_logger):
        """Test that fallbacks are logged."""
        primary = MockLLM(model="primary")
        primary.should_fail = True
        primary.failure_exception = Exception("model not found")

        backup = MockLLM(model="backup")

        config = FallbackConfig(fallback_llms=[backup])
        wrapper = FallbackWrapper(primary, config)

        wrapper.ask("Test")

        # Check that warning was logged
        assert mock_logger.warning.called
        warning_calls = mock_logger.warning.call_args_list
        assert any("LLM 1/2 failed" in str(call) for call in warning_calls)
        assert any("Falling back to next LLM" in str(call) for call in warning_calls)

    def test_isinstance_check(self):
        """Test that wrapped instances have proper attributes."""
        llm = MockLLM(model="test")

        # Test RetryWrapper
        from pyhub.llm.retry import RetryWrapper

        retry_llm = llm.with_retry()
        assert isinstance(retry_llm, RetryWrapper)
        assert hasattr(retry_llm, "llm")
        assert hasattr(retry_llm, "config")
        assert retry_llm.model == "test"

        # Test FallbackWrapper
        from pyhub.llm.retry import FallbackWrapper

        fallback_llm = llm.with_fallbacks([MockLLM(model="backup")])
        assert isinstance(fallback_llm, FallbackWrapper)
        assert hasattr(fallback_llm, "llm")
        assert hasattr(fallback_llm, "config")
        assert fallback_llm.model == "test"

        # Test chained wrappers
        chained = llm.with_retry().with_fallbacks([MockLLM(model="backup")])
        assert isinstance(chained, FallbackWrapper)
        assert hasattr(chained, "llm")
        assert hasattr(chained, "config")
