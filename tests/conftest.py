import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from pyhub.llm import LLM
from pyhub.llm.cache import FileCache, MemoryCache
from pyhub.llm.types import Message


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for cache testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def memory_cache() -> MemoryCache:
    """Create a memory cache instance."""
    return MemoryCache()


@pytest.fixture
def file_cache(temp_cache_dir: Path) -> FileCache:
    """Create a file cache instance."""
    return FileCache(str(temp_cache_dir))


@pytest.fixture
def mock_openai_client() -> Mock:
    """Create a mock OpenAI client."""
    mock = Mock()

    # Mock completion response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    mock.chat.completions.create.return_value = mock_response

    # Mock streaming response
    def mock_stream():
        chunks = [
            Mock(choices=[Mock(delta=Mock(content="Test"))]),
            Mock(choices=[Mock(delta=Mock(content=" streaming"))]),
            Mock(choices=[Mock(delta=Mock(content=" response"))]),
        ]
        for chunk in chunks:
            yield chunk

    mock.chat.completions.create.return_value = mock_stream()

    return mock


@pytest.fixture
def mock_anthropic_client() -> Mock:
    """Create a mock Anthropic client."""
    mock = Mock()

    # Mock completion response
    mock_response = Mock()
    mock_response.content = [Mock(text="Test response")]
    mock_response.usage = Mock(input_tokens=10, output_tokens=20)

    mock.messages.create.return_value = mock_response

    return mock


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create sample messages for testing."""
    return [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
        Message(role="user", content="How are you?"),
    ]


@pytest.fixture
def mock_llm_factory(monkeypatch):
    """Mock LLMFactory to avoid real API calls."""

    def _mock_factory(model: str, **kwargs):
        mock_llm = Mock(spec=LLM)
        mock_llm.ask.return_value = "Mocked response"
        mock_llm.ask_async = AsyncMock(return_value="Mocked async response")
        mock_llm.messages.return_value = "Mocked messages response"
        mock_llm.embed.return_value = Mock(embeddings=[0.1, 0.2, 0.3])
        return mock_llm

    monkeypatch.setattr("pyhub.llm.LLM.create", _mock_factory)


# Skip markers for provider tests
def pytest_configure(config):
    config.addinivalue_line("markers", "openai: mark test as requiring OpenAI API key")
    config.addinivalue_line("markers", "anthropic: mark test as requiring Anthropic API key")
    config.addinivalue_line("markers", "google: mark test as requiring Google API key")
    config.addinivalue_line("markers", "ollama: mark test as requiring Ollama running locally")


# Auto-skip tests based on API key availability
def pytest_collection_modifyitems(config, items):
    skip_openai = pytest.mark.skip(reason="OPENAI_API_KEY not set")
    skip_anthropic = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    skip_google = pytest.mark.skip(reason="GOOGLE_API_KEY not set")
    skip_ollama = pytest.mark.skip(reason="Ollama not running")

    for item in items:
        if "openai" in item.keywords and not os.getenv("OPENAI_API_KEY"):
            item.add_marker(skip_openai)
        if "anthropic" in item.keywords and not os.getenv("ANTHROPIC_API_KEY"):
            item.add_marker(skip_anthropic)
        if "google" in item.keywords and not os.getenv("GOOGLE_API_KEY"):
            item.add_marker(skip_google)
        if "ollama" in item.keywords:
            # Check if Ollama is running
            try:
                import httpx

                response = httpx.get("http://localhost:11434/api/tags")
                if response.status_code != 200:
                    item.add_marker(skip_ollama)
            except Exception:
                item.add_marker(skip_ollama)
