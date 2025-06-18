# Retry and Fallback Strategies

pyhub-llm provides robust retry and fallback mechanisms to handle transient failures and ensure reliable LLM API calls.

## Overview

The retry and fallback system is inspired by LangChain's approach but designed to be more Pythonic and intuitive. It provides:

- **Retry Logic**: Automatically retry failed requests with configurable backoff strategies
- **Fallback Chains**: Switch to alternative LLMs when primary ones fail
- **LLM Instance-based Fallbacks**: Use actual LLM instances instead of model names for clarity
- **Flexible Configuration**: Customize retry conditions, delays, and callbacks
- **Async Support**: Full support for async operations

## Basic Usage

### Simple Retry

Add retry logic to any LLM with the `with_retry()` method:

```python
from pyhub.llm import OpenAILLM

# Create LLM with retry
llm = OpenAILLM(model="gpt-4o-mini").with_retry(max_retries=3)

# Automatically retries on transient errors
response = llm.ask("Hello, world!")
```

### Simple Fallback

Create a fallback chain with alternative LLMs:

```python
from pyhub.llm import OpenAILLM, AnthropicLLM

# Create primary and backup LLMs
primary = OpenAILLM(model="gpt-4o")
backup1 = OpenAILLM(model="gpt-4o-mini")
backup2 = AnthropicLLM(model="claude-3-haiku")

# Set up fallback chain
llm = primary.with_fallbacks([backup1, backup2])

# Automatically falls back on errors
response = llm.ask("Write a poem")
```

### Combining Retry and Fallback

You can chain retry and fallback for maximum reliability:

```python
# Each LLM can have its own retry configuration
llm = OpenAILLM(model="gpt-4o").with_retry(
    max_retries=2
).with_fallbacks([
    OpenAILLM(model="gpt-4o-mini").with_retry(max_retries=3),
    AnthropicLLM(model="claude-3-sonnet")
])
```

## Retry Configuration

### Backoff Strategies

Configure how delays increase between retries:

```python
# Exponential backoff (default)
llm.with_retry(
    max_retries=3,
    initial_delay=1.0,
    backoff_multiplier=2.0,
    backoff_strategy="exponential"
)

# Linear backoff
llm.with_retry(
    initial_delay=2.0,
    backoff_strategy="linear"
)

# Fixed delay
llm.with_retry(
    initial_delay=5.0,
    backoff_strategy="fixed"
)

# Exponential with jitter (recommended for production)
llm.with_retry(
    backoff_strategy="jitter"
)
```

### Retry Conditions

Control which errors trigger retries:

```python
# Retry on specific exceptions
llm.with_retry(
    retry_on=[ConnectionError, TimeoutError, "rate limit"]
)

# Custom retry condition
def should_retry(error: Exception) -> bool:
    return "temporary" in str(error).lower()

llm.with_retry(
    retry_condition=should_retry
)

# Stop on specific errors (don't retry)
llm.with_retry(
    retry_on=[Exception],  # Retry all errors
    stop_on=["invalid api key", "insufficient quota"]  # Except these
)
```

### Callbacks

Monitor retry behavior with callbacks:

```python
def on_retry(error: Exception, attempt: int, delay: float):
    print(f"Retry {attempt} after {delay}s: {error}")

def on_failure(error: Exception, attempts: int):
    print(f"Failed after {attempts} attempts: {error}")

llm.with_retry(
    on_retry=on_retry,
    on_failure=on_failure
)
```

## Fallback Configuration

### Conditional Fallbacks

Control when fallbacks are triggered:

```python
def should_fallback(error: Exception) -> bool:
    # Only fallback on context length errors
    return "context length" in str(error).lower()

llm.with_fallbacks(
    [smaller_model],
    fallback_condition=should_fallback
)
```

### Fallback Callbacks

Monitor fallback events:

```python
def on_fallback(error: Exception, llm):
    print(f"Switching to {llm.model} due to: {error}")

llm.with_fallbacks(
    [backup1, backup2],
    on_fallback=on_fallback
)
```

## Advanced Examples

### Production-Ready Setup

A comprehensive setup for production environments:

```python
import logging
from pyhub.llm import OpenAILLM, AnthropicLLM

logger = logging.getLogger(__name__)

# Primary LLM with aggressive retry
primary = OpenAILLM(
    model="gpt-4o",
    temperature=0.3
).with_retry(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    backoff_strategy="jitter",
    retry_on=[ConnectionError, TimeoutError, "rate limit"],
    stop_on=["invalid api key"],
    on_retry=lambda e, attempt, delay: 
        logger.warning(f"Retry {attempt}: {e}"),
    on_failure=lambda e, attempts: 
        logger.error(f"Failed after {attempts} attempts: {e}")
)

# Backup LLMs with less aggressive retry
backup1 = OpenAILLM(
    model="gpt-4o-mini"
).with_retry(max_retries=2)

backup2 = AnthropicLLM(
    model="claude-3-haiku"
).with_retry(max_retries=1)

# Create robust LLM chain
llm = primary.with_fallbacks(
    [backup1, backup2],
    on_fallback=lambda e, llm: 
        logger.info(f"Fallback to {llm.model}: {e}")
)
```

### Different Models for Different Errors

Use different fallback strategies based on error type:

```python
# Primary high-performance model
primary = OpenAILLM(model="gpt-4o")

# Smaller model for context length issues
small_model = OpenAILLM(model="gpt-4o-mini", max_tokens=1000)

# Alternative provider for API issues
alternative = AnthropicLLM(model="claude-3-sonnet")

# First, handle context length errors
llm = primary.with_fallbacks(
    [small_model],
    fallback_condition=lambda e: "context length" in str(e).lower()
)

# Then, handle general API errors
llm = llm.with_fallbacks([alternative])
```

### Async Operations

Retry and fallback work seamlessly with async operations:

```python
import asyncio

# Configure retry and fallback
llm = OpenAILLM(model="gpt-4o").with_retry(
    max_retries=3
).with_fallbacks([
    AnthropicLLM(model="claude-3-sonnet")
])

# Use with async
async def main():
    response = await llm.ask_async("Hello!")
    print(response.text)

asyncio.run(main())
```

## Best Practices

### 1. Use Jitter in Production

Jitter helps prevent thundering herd problems:

```python
llm.with_retry(backoff_strategy="jitter")
```

### 2. Set Reasonable Limits

Don't retry forever:

```python
llm.with_retry(
    max_retries=5,
    max_delay=60.0  # Cap at 1 minute
)
```

### 3. Log Important Events

Use callbacks for observability:

```python
llm.with_retry(
    on_retry=lambda e, attempt, delay: 
        logger.warning(f"Retry {attempt}: {e}"),
    on_failure=lambda e, attempts: 
        logger.error(f"Failed: {e}")
).with_fallbacks(
    [backup],
    on_fallback=lambda e, llm: 
        logger.info(f"Using fallback: {llm.model}")
)
```

### 4. Handle Non-Retryable Errors

Some errors shouldn't be retried:

```python
llm.with_retry(
    stop_on=["invalid api key", "insufficient quota", "unauthorized"]
)
```

### 5. Use Different Strategies for Different LLMs

Each LLM in your fallback chain can have different retry settings:

```python
primary = OpenAILLM("gpt-4o").with_retry(max_retries=3)
backup = AnthropicLLM("claude-3-sonnet").with_retry(max_retries=1)
llm = primary.with_fallbacks([backup])
```

## Error Types

### Retryable Errors (Default)

These errors are retried by default:
- `ConnectionError`
- `TimeoutError`
- `OSError` (network-related)
- Errors containing: "rate limit", "too many requests", "quota exceeded", "server error", "internal error", "service unavailable", "timeout", "connection", "network"

### Fallback Errors (Default)

These errors trigger fallback by default:
- Errors containing: "context length", "token limit", "model not found", "model unavailable", "unsupported", "not supported"

### Custom Error Handling

You can customize error handling for your specific needs:

```python
# Custom retry logic
def my_retry_condition(error):
    # Your logic here
    return should_retry

# Custom fallback logic
def my_fallback_condition(error):
    # Your logic here
    return should_fallback

llm.with_retry(retry_condition=my_retry_condition)
llm.with_fallbacks([backup], fallback_condition=my_fallback_condition)
```

## API Reference

### RetryConfig

```python
from pyhub.llm.retry import RetryConfig, BackoffStrategy

config = RetryConfig(
    max_retries=3,                    # Maximum retry attempts
    initial_delay=1.0,                # Initial delay in seconds
    max_delay=60.0,                   # Maximum delay in seconds
    backoff_multiplier=2.0,           # Multiplier for exponential backoff
    backoff_strategy=BackoffStrategy.EXPONENTIAL,  # Strategy enum
    jitter=True,                      # Add randomness to delays
    retry_on=[Exception],             # Exceptions to retry
    retry_condition=None,             # Custom retry function
    stop_on=[],                       # Exceptions to not retry
    on_retry=None,                    # Callback on retry
    on_failure=None                   # Callback on final failure
)
```

### FallbackConfig

```python
from pyhub.llm.retry import FallbackConfig

config = FallbackConfig(
    fallback_llms=[llm1, llm2],      # List of fallback LLM instances
    fallback_condition=None,          # Custom fallback function
    on_fallback=None                  # Callback on fallback
)
```

### BackoffStrategy Enum

```python
from pyhub.llm.retry import BackoffStrategy

BackoffStrategy.EXPONENTIAL  # Default: delay * multiplier^attempt
BackoffStrategy.LINEAR       # delay * attempt
BackoffStrategy.FIXED        # Always use initial_delay
BackoffStrategy.JITTER       # Exponential with random jitter
```