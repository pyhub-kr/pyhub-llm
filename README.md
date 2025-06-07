# pyhub-llm

Standalone LLM library with support for multiple providers.

## Features

- Multiple LLM provider support (OpenAI, Anthropic, Google, Ollama)
- Async/streaming support
- Built-in caching system
- Agent framework with tool support
- Model Context Protocol (MCP) integration
- Type-safe interfaces
- Provider-agnostic design

## Installation

```bash
pip install pyhub-llm

# With specific providers
pip install pyhub-llm[openai]
pip install pyhub-llm[anthropic]
pip install pyhub-llm[all]  # All providers
```

## Quick Start

```python
from pyhub.llm import LLM

# Create LLM instance
llm = LLM.create("gpt-4", api_key="your-api-key")

# Simple completion
response = llm.ask("What is the capital of France?")
print(response)

# Streaming
for chunk in llm.stream("Tell me a story"):
    print(chunk, end="", flush=True)

# Async usage
import asyncio

async def main():
    response = await llm.ask_async("Hello, world!")
    print(response)

asyncio.run(main())
```

## Caching

```python
from pyhub.llm import LLM
from pyhub.llm.cache import FileCache

# Use file-based cache
llm = LLM.create(
    "gpt-4",
    cache=FileCache("~/.pyhub/cache")
)

# Cached response (second call will be instant)
response1 = llm.ask("What is 2+2?")
response2 = llm.ask("What is 2+2?")  # From cache
```

## Agents

```python
from pyhub.llm import LLM
from pyhub.llm.agents import create_react_agent
from pyhub.llm.agents.tools import Calculator, WebSearch

# Create agent with tools
llm = LLM.create("gpt-4")
agent = create_react_agent(
    llm=llm,
    tools=[Calculator(), WebSearch()]
)

# Run agent
result = await agent.run("What is the weather in Paris today?")
print(result)
```

## CLI Usage

```bash
# Ask a question
pyhub-llm ask "What is the capital of France?"

# Interactive chat
pyhub-llm chat

# Compare models
pyhub-llm compare "Explain quantum computing" --models gpt-4 --models claude-3

# Run agent
pyhub-llm agent run "Calculate 25 * 4 + 10"
```

## License

MIT