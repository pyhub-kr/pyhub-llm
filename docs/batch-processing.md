# Batch Processing

pyhub-llm supports efficient batch processing of multiple prompts using the `batch()` method. This feature allows you to process multiple queries in parallel or sequentially with configurable history management.

## Features

- **Parallel Processing**: Independent prompts can be processed simultaneously
- **Sequential Mode**: Build conversations where each response provides context for the next
- **Shared Context**: Use the same initial history for all prompts
- **Error Handling**: Continue processing even if some prompts fail
- **Concurrency Control**: Limit the number of parallel requests
- **Synchronous Option**: Use `batch_sync()` for synchronous execution

## Usage

### Basic Batch Processing

```python
import asyncio
from pyhub.llm import LLM

# Create LLM instance
llm = LLM.create("gpt-4o-mini")

# Process multiple prompts in parallel
prompts = [
    "What is Python?",
    "What is JavaScript?", 
    "What is Go?"
]

replies = await llm.batch(prompts)

for i, reply in enumerate(replies):
    print(f"Q: {prompts[i]}")
    print(f"A: {reply.text}")
```

### History Modes

#### 1. Independent Mode (Default)

Each prompt runs independently without affecting others. Allows for parallel processing.

```python
# Independent processing (parallel execution)
replies = await llm.batch([
    "What is machine learning?",
    "What is deep learning?",
    "What is neural networks?"
], history_mode="independent", max_parallel=3)
```

#### 2. Sequential Mode

Each response becomes context for the next prompt. Processes sequentially.

```python
# Sequential conversation building
replies = await llm.batch([
    "Explain the Fibonacci sequence",
    "Write a Python function to calculate it",
    "What's the time complexity?",
    "How can we optimize it?"
], history_mode="sequential", use_history=True)
```

#### 3. Shared Context Mode

All prompts share the same initial history but don't affect each other.

```python
# Set up context
llm.ask("Our company has products A, B, and C", use_history=True)

# Use shared context for multiple queries
replies = await llm.batch([
    "What are the benefits of product A?",
    "Who is the target audience for product B?",
    "What are the use cases for product C?"
], history_mode="shared", use_history=True)
```

### Advanced Options

#### Concurrency Control

```python
# Limit parallel requests to avoid rate limits
replies = await llm.batch(
    prompts,
    max_parallel=2,  # Only 2 requests at once
    history_mode="independent"
)
```

#### Error Handling

```python
# Continue processing even if some prompts fail
replies = await llm.batch(
    prompts,
    fail_fast=False  # Default: continue on errors
)

# Stop on first error
replies = await llm.batch(
    prompts,
    fail_fast=True
)
```

#### Custom Parameters

```python
# Use custom system prompt and parameters
replies = await llm.batch(
    prompts,
    system_prompt="You are a helpful coding assistant",
    temperature=0.7,
    max_tokens=150
)
```

### Synchronous Usage

For non-async code, use `batch_sync()`:

```python
# Synchronous batch processing
replies = llm.batch_sync([
    "What is 2 + 2?",
    "What is 10 * 5?",
    "What is 100 / 4?"
])
```

## Method Signature

```python
async def batch(
    self,
    prompts: List[str],
    *,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_parallel: int = 5,
    use_history: bool = False,
    history_mode: Literal["independent", "sequential", "shared"] = "independent",
    fail_fast: bool = False,
    **kwargs
) -> List[Reply]:
```

### Parameters

- `prompts`: List of prompts to process
- `system_prompt`: System prompt to use (optional)
- `temperature`: Generation temperature (optional)
- `max_tokens`: Maximum tokens per response (optional)
- `max_parallel`: Maximum number of parallel requests (only for "independent" mode)
- `use_history`: Whether to use conversation history
- `history_mode`: How to handle conversation history:
  - `"independent"`: Each prompt runs independently (default, allows parallel)
  - `"sequential"`: Each response becomes context for next prompt
  - `"shared"`: All prompts share the same initial history
- `fail_fast`: If True, stop on first error. If False, continue processing
- `**kwargs`: Additional parameters passed to `ask()`

## Performance Considerations

1. **Independent Mode**: Fastest for unrelated queries due to parallel processing
2. **Sequential Mode**: Slowest but builds conversational context
3. **Shared Mode**: Balance between context sharing and parallel processing
4. **Rate Limits**: Use `max_parallel` to avoid hitting API rate limits
5. **Memory**: Large batch sizes may consume significant memory

## Error Handling

When `fail_fast=False` (default), failed prompts return Reply objects with error messages:

```python
replies = await llm.batch(prompts, fail_fast=False)

for i, reply in enumerate(replies):
    if "Error processing prompt" in reply.text:
        print(f"Prompt {i+1} failed: {reply.text}")
    else:
        print(f"Prompt {i+1} succeeded: {reply.text}")
```

## Examples

See `examples/basic/07_batch_processing.py` for comprehensive examples of all batch processing modes and features.