# pyhub-llm

A powerful, standalone LLM library with unified interface for multiple providers including OpenAI, Anthropic, Google, and Ollama. Built with modern Python practices, type safety, and extensibility in mind.

## ✨ Key Features

- **🔌 Multiple Provider Support**: Seamlessly switch between OpenAI, Anthropic, Google, and Ollama
- **⚡ Async & Streaming**: First-class support for asynchronous operations and streaming responses
- **💾 Smart Caching**: Built-in caching system to reduce API calls and costs
- **🤖 Agent Framework**: Build complex AI agents with tool support and ReAct pattern
- **🔧 MCP Integration**: Full Model Context Protocol support for advanced tool usage
- **📝 Type Safety**: Comprehensive type hints for better IDE support and fewer runtime errors
- **🎯 Provider Agnostic**: Write once, run with any supported LLM provider
- **🔗 Method Chaining**: Elegant API with pipe operator support for sequential operations

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

# Auto-detect provider from model name
llm = LLM("gpt-4o-mini")

# Simple completion
response = llm.ask("What is the capital of France?")
print(response.text)

# Streaming
for chunk in llm.ask("Tell me a story", stream=True):
    print(chunk.text, end="", flush=True)

# Async usage
import asyncio

async def main():
    llm = LLM("claude-3-5-haiku-latest")
    response = await llm.ask_async("Hello, world!")
    print(response.text)

asyncio.run(main())
```

## Caching

```python
from pyhub.llm import LLM

# Enable caching per request
llm = LLM("gpt-4o-mini")

# First call - makes API request
response1 = llm.ask("What is 2+2?", enable_cache=True)

# Second call - returns from cache instantly
response2 = llm.ask("What is 2+2?", enable_cache=True)
```

## Agents

```python
from pyhub.llm import LLM
from pyhub.llm.agents import ReActAgent
from pyhub.llm.agents.tools import Calculator

# Create agent with tools
llm = LLM("gpt-4o-mini")
agent = ReActAgent(llm=llm, tools=[Calculator()])

# Run agent
result = agent.run("What is 25 * 4 + 10?")
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

## Advanced Usage

### Method Chaining

```python
from pyhub.llm import LLM

# Chain multiple LLMs together
llm1 = LLM("gpt-4o-mini", prompt="Translate to French: {input}")
llm2 = LLM("claude-3-5-haiku-latest", prompt="Make it more formal: {text}")

chain = llm1 | llm2
result = chain.ask({"input": "Hello, how are you?"})
print(result.text)
```

### Custom Prompts with Templates

```python
from pyhub.llm import LLM

llm = LLM(
    "gpt-4o-mini",
    system_prompt="You are a helpful coding assistant.",
    prompt="Language: {language}\nTask: {task}\nCode:"
)

response = llm.ask({
    "language": "Python", 
    "task": "Binary search implementation"
})
print(response.text)
```

## Environment Variables

You can set API keys via environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

Or use a `.env` file in your project root.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT