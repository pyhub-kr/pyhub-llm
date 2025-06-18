"""
Examples of retry and fallback functionality in pyhub-llm.

This module demonstrates various ways to use retry and fallback
strategies to make your LLM API calls more robust.
"""

import asyncio
import logging
from typing import List

from pyhub.llm import AnthropicLLM, LLM, OpenAILLM

# Set up logging to see retry/fallback behavior
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_retry():
    """Example 1: Basic retry with exponential backoff."""
    print("\n=== Example 1: Basic Retry ===")
    
    # Create an LLM with retry logic
    llm = OpenAILLM(model="gpt-4o-mini").with_retry(
        max_retries=3,
        initial_delay=1.0,
        backoff_strategy="exponential"
    )
    
    # This will automatically retry on transient errors
    response = llm.ask("What is the capital of France?")
    print(f"Response: {response.text}")


def example_custom_retry_conditions():
    """Example 2: Custom retry conditions."""
    print("\n=== Example 2: Custom Retry Conditions ===")
    
    # Define custom retry logic
    def should_retry(error: Exception) -> bool:
        """Retry only on specific errors."""
        error_msg = str(error).lower()
        return any(keyword in error_msg for keyword in [
            "rate limit",
            "timeout",
            "connection",
            "quota exceeded"
        ])
    
    llm = OpenAILLM(model="gpt-4o-mini").with_retry(
        max_retries=5,
        initial_delay=0.5,
        max_delay=30.0,
        backoff_strategy="jitter",  # Adds randomness to avoid thundering herd
        retry_condition=should_retry,
        on_retry=lambda e, attempt, delay: logger.info(
            f"Retry attempt {attempt} after {delay:.2f}s due to: {e}"
        )
    )
    
    response = llm.ask("Explain exponential backoff")
    print(f"Response: {response.text[:100]}...")


def example_simple_fallback():
    """Example 3: Simple fallback chain."""
    print("\n=== Example 3: Simple Fallback ===")
    
    # Create a chain of LLMs with different models
    primary = OpenAILLM(model="gpt-4o", temperature=0.7)
    backup1 = OpenAILLM(model="gpt-4o-mini", temperature=0.5)
    backup2 = AnthropicLLM(model="claude-3-haiku", temperature=0.3)
    
    # Set up fallback chain
    llm = primary.with_fallbacks([backup1, backup2])
    
    # If primary fails (e.g., context too long), it will try backups
    response = llm.ask("Write a haiku about AI")
    print(f"Response: {response.text}")


def example_conditional_fallback():
    """Example 4: Conditional fallback based on error type."""
    print("\n=== Example 4: Conditional Fallback ===")
    
    # Use different models for different error scenarios
    primary = OpenAILLM(model="gpt-4o")
    
    # Use smaller model for context length errors
    small_model = OpenAILLM(model="gpt-4o-mini")
    
    # Use different provider for API errors
    alternative = AnthropicLLM(model="claude-3-sonnet")
    
    def should_use_smaller_model(error: Exception) -> bool:
        """Use smaller model for context length issues."""
        return "context length" in str(error).lower()
    
    # First fallback for context issues
    llm_with_small_fallback = primary.with_fallbacks(
        [small_model],
        fallback_condition=should_use_smaller_model,
        on_fallback=lambda e, llm: logger.info(
            f"Switching to {llm.model} due to: {e}"
        )
    )
    
    # Then add general fallback for other issues
    llm = llm_with_small_fallback.with_fallbacks([alternative])
    
    response = llm.ask("Summarize this text: " + "Lorem ipsum " * 1000)
    print(f"Response: {response.text[:100]}...")


def example_retry_with_fallback():
    """Example 5: Combine retry and fallback strategies."""
    print("\n=== Example 5: Retry + Fallback ===")
    
    # Each LLM gets its own retry configuration
    primary = OpenAILLM(model="gpt-4o").with_retry(
        max_retries=2,
        initial_delay=1.0
    )
    
    backup = AnthropicLLM(model="claude-3-sonnet").with_retry(
        max_retries=3,
        initial_delay=0.5
    )
    
    # Combine with fallback
    llm = primary.with_fallbacks([backup])
    
    # This will:
    # 1. Try primary up to 3 times (1 initial + 2 retries)
    # 2. If all fail, try backup up to 4 times (1 initial + 3 retries)
    response = llm.ask("What are the benefits of retry strategies?")
    print(f"Response: {response.text[:100]}...")


def example_production_setup():
    """Example 6: Production-ready setup with comprehensive error handling."""
    print("\n=== Example 6: Production Setup ===")
    
    # Define callback functions for monitoring
    def on_retry(error: Exception, attempt: int, delay: float):
        """Log retry attempts for monitoring."""
        logger.warning(f"Retry {attempt} after {delay:.2f}s: {type(error).__name__}: {error}")
        # In production, you might send metrics here
        # metrics.increment("llm.retry", tags={"attempt": attempt, "error": type(error).__name__})
    
    def on_fallback(error: Exception, llm):
        """Log fallback events."""
        logger.error(f"Fallback to {llm.model}: {type(error).__name__}: {error}")
        # metrics.increment("llm.fallback", tags={"model": llm.model})
    
    def on_failure(error: Exception, attempts: int):
        """Log final failures."""
        logger.error(f"All {attempts} attempts failed: {error}")
        # metrics.increment("llm.failure", tags={"attempts": attempts})
    
    # Create robust LLM setup
    primary = OpenAILLM(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=2000
    ).with_retry(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_strategy="exponential",
        retry_on=[ConnectionError, TimeoutError, "rate limit"],
        stop_on=["invalid api key", "insufficient quota"],
        on_retry=on_retry,
        on_failure=on_failure
    )
    
    backup1 = OpenAILLM(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1500
    ).with_retry(
        max_retries=2,
        initial_delay=0.5
    )
    
    backup2 = AnthropicLLM(
        model="claude-3-haiku",
        temperature=0.3,
        max_tokens=1000
    ).with_retry(
        max_retries=2,
        initial_delay=0.5
    )
    
    # Create fallback chain
    llm = primary.with_fallbacks(
        [backup1, backup2],
        on_fallback=on_fallback
    )
    
    try:
        response = llm.ask(
            "Explain the importance of fault tolerance in distributed systems",
            raise_errors=True
        )
        print(f"Success! Response: {response.text[:100]}...")
    except Exception as e:
        print(f"Failed after all attempts: {e}")


async def example_async_retry_fallback():
    """Example 7: Async operations with retry and fallback."""
    print("\n=== Example 7: Async Operations ===")
    
    # Async operations work the same way
    llm = OpenAILLM(model="gpt-4o-mini").with_retry(
        max_retries=3
    ).with_fallbacks([
        AnthropicLLM(model="claude-3-haiku")
    ])
    
    # Async call with automatic retry/fallback
    response = await llm.ask_async("What is async/await in Python?")
    print(f"Response: {response.text[:100]}...")


def example_retry_with_different_models():
    """Example 8: Using LLM instances directly in fallbacks."""
    print("\n=== Example 8: LLM Instance Fallbacks ===")
    
    # Create different LLM configurations
    fast_llm = OpenAILLM(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=500
    )
    
    balanced_llm = OpenAILLM(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=1000
    )
    
    powerful_llm = AnthropicLLM(
        model="claude-3-opus",
        temperature=0.5,
        max_tokens=2000
    )
    
    # Start with fast model, fallback to more powerful ones
    llm = fast_llm.with_fallbacks([balanced_llm, powerful_llm])
    
    # For simple queries, fast model will suffice
    simple_response = llm.ask("What is 2+2?")
    print(f"Simple query response: {simple_response.text}")
    
    # For complex queries that might fail on simple models,
    # it will automatically fallback to more powerful ones
    complex_response = llm.ask(
        "Write a detailed analysis of the philosophical implications "
        "of artificial general intelligence on human society"
    )
    print(f"Complex query response: {complex_response.text[:100]}...")


def main():
    """Run all examples."""
    # Synchronous examples
    example_basic_retry()
    example_custom_retry_conditions()
    example_simple_fallback()
    example_conditional_fallback()
    example_retry_with_fallback()
    example_production_setup()
    example_retry_with_different_models()
    
    # Async example
    print("\nRunning async example...")
    asyncio.run(example_async_retry_fallback())
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()