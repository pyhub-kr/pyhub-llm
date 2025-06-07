# Migration Guide: django-pyhub-rag to pyhub-llm

This guide helps you migrate from using the integrated LLM module in django-pyhub-rag to the standalone pyhub-llm library.

## Overview

The LLM functionality has been extracted from django-pyhub-rag into a standalone library called pyhub-llm. This allows:
- Using LLM features without Django
- Better modularity and maintainability
- Reduced dependencies for LLM-only use cases

## Installation

### Option 1: With django-pyhub-rag (Recommended)
If you're already using django-pyhub-rag, it will automatically use pyhub-llm when available:

```bash
pip install pyhub-llm
pip install django-pyhub-rag
```

### Option 2: Standalone
For LLM features only:

```bash
pip install pyhub-llm
```

## Import Changes

### Django Projects
If pyhub-llm is installed, django-pyhub-rag will automatically use it. Your existing imports will continue to work:

```python
# These imports still work with django-pyhub-rag
from pyhub.llm_adapter import LLM
from pyhub.llm_adapter.types import Reply, Message
```

### Standalone Usage
For non-Django projects, import directly from pyhub-llm:

```python
# Direct imports from pyhub-llm
from pyhub.llm import LLM
from pyhub.llm.types import Reply, Message
```

## API Compatibility

The API remains largely the same:

```python
# Django project (with automatic Django integration)
from pyhub.llm_adapter import LLM

llm = LLM.create("gpt-4")  # Uses Django cache and settings automatically
response = llm.ask("Hello!")

# Standalone project
from pyhub.llm import LLM
from pyhub.llm.cache import FileCache

llm = LLM.create("gpt-4", cache=FileCache())
response = llm.ask("Hello!")
```

## Key Differences

### 1. Caching
- **Django**: Automatically uses Django's cache backend
- **Standalone**: Uses file-based or memory cache

### 2. Templates
- **Django**: Uses Django's template engine
- **Standalone**: Uses Jinja2

### 3. Settings
- **Django**: Reads from Django settings (e.g., `OPENAI_API_KEY`)
- **Standalone**: Uses environment variables or config files

### 4. Models
- **Django**: `ImageDescriptorPrompt` model still in django-pyhub-rag
- **Standalone**: No database models

## Configuration

### Django Projects
Continue using Django settings:

```python
# settings.py
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

### Standalone Projects
Use environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or configuration file (`~/.pyhub/config.toml`):

```toml
openai_api_key = "sk-..."
anthropic_api_key = "sk-ant-..."

[cache]
dir = "~/.pyhub/cache"
```

## Feature Availability

| Feature | django-pyhub-rag | pyhub-llm |
|---------|------------------|-----------|
| LLM Providers | ✓ | ✓ |
| Streaming | ✓ | ✓ |
| Embeddings | ✓ | ✓ |
| Function Calling | ✓ | ✓ |
| Agents | ✓ | ✓ |
| MCP Support | ✓ | ✓ |
| Django Models | ✓ | ✗ |
| Django Cache | ✓ | ✗ (File/Memory cache) |
| Django Templates | ✓ | ✗ (Jinja2) |

## Troubleshooting

### Import Errors
If you see `ImportError: No module named 'pyhub.llm'`:
1. Install pyhub-llm: `pip install pyhub-llm`
2. Or continue using the legacy implementation in django-pyhub-rag

### Cache Issues
If caching doesn't work as expected:
- Django: Check Django cache settings
- Standalone: Ensure cache directory is writable

### API Key Issues
- Django: Check Django settings
- Standalone: Check environment variables or config file