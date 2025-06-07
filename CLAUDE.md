# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation
```bash
# Install development environment with all providers
make install

# Or manually:
pip install -e ".[dev,all]"
```

### Testing
```bash
# Run all tests
make test

# Run specific test
pytest tests/test_file.py::test_function -v

# Run with coverage
pytest tests/ --cov=src/pyhub/llm --cov-report=html
```

### Code Quality
```bash
# Format code
make format

# Check code style and types
make lint

# Or individually:
black src/ tests/
ruff check src/ tests/
mypy src/
```

### Building
```bash
# Clean and build package
make build

# Build documentation
make docs
```

## High-Level Architecture

### Core Design Pattern: Provider Abstraction
The library uses an abstract factory pattern where all LLM providers (OpenAI, Anthropic, Google, Ollama) inherit from `BaseLLM` in `src/pyhub/llm/base.py`. New providers are registered with `LLMFactory` which handles automatic provider detection based on model names.

### Key Architectural Components

1. **Factory Pattern (`factory.py`)**: Central entry point for creating LLM instances. Auto-detects providers from model names and integrates with settings system for API keys.

2. **Type System (`types.py`)**: Comprehensive type definitions using dataclasses and TypeAlias. All vendor-specific model types are defined here with extensibility via Union types.

3. **Caching Layer (`cache/`)**: Pluggable cache system with `BaseCache` interface. Implementations include in-memory and file-based caches with TTL support.

4. **Template Engine (`templates/`)**: Jinja2-based prompt templating system for managing complex prompts across providers.

5. **Settings Management (`settings.py`)**: Hierarchical configuration system that reads from environment variables, .env files, and pyproject.toml.

### Provider Implementation Pattern
When adding a new provider:
1. Create a new file in `src/pyhub/llm/providers/`
2. Inherit from `BaseLLM` and implement required abstract methods: `ask()`, `messages()`, `embed()`
3. Register the provider in `factory.py` using `LLMFactory.register_provider()`
4. Add provider-specific model types to `types.py`

### Method Chaining and Streaming
The library supports elegant method chaining using the pipe operator (`|`) through `SequentialChain`. All providers support both sync and async operations with streaming capabilities.

### Agent Framework
While the agents directory is currently being developed, the base architecture supports tool/function calling through `ask_with_tools()` methods with provider-specific tool adaptation.