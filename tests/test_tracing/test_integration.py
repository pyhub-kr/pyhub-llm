"""
Integration tests for tracing with LLM classes.
"""

from unittest.mock import Mock

import pytest

from pyhub.llm.base import BaseLLM
from pyhub.llm.tracing.base import (
    SpanData,
    SpanKind,
    TracingProvider,
    init_tracer,
)
from pyhub.llm.types import Reply


class MockProvider(TracingProvider):
    """Mock provider that captures all tracing calls."""

    def __init__(self):
        self.spans = []
        self.events = []
        self.flush_called = False

    def start_span(self, span_data: SpanData) -> None:
        self.spans.append(("start", span_data))

    def end_span(self, span_data: SpanData) -> None:
        self.spans.append(("end", span_data))

    def add_event(self, span_id: str, name: str, attributes: dict) -> None:
        self.events.append((span_id, name, attributes))

    def flush(self) -> None:
        self.flush_called = True


class MockLLM(BaseLLM):
    """Mock LLM for testing tracing integration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0
        self.mock_response = Reply(text="Mock response")

    def _make_request_params(self, input_context, human_message, messages, model):
        return {}

    def _make_ask(self, input_context, human_message, messages, model):
        self.call_count += 1
        self._last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return self.mock_response

    async def _make_ask_async(self, input_context, human_message, messages, model):
        return self._make_ask(input_context, human_message, messages, model)

    def embed(self, input, model=None):
        return Mock()

    async def embed_async(self, input, model=None):
        return Mock()

    def _make_ask_stream(self, input_context, human_message, messages, model):
        """Generate a streaming response using the specific LLM provider"""
        yield Reply(text="Mock stream response")

    async def _make_ask_stream_async(self, input_context, human_message, messages, model):
        """Generate a streaming response asynchronously using the specific LLM provider"""
        yield Reply(text="Mock async stream response")

    def generate_image(self, prompt, **kwargs):
        return Mock()

    async def generate_image_async(self, prompt, **kwargs):
        return Mock()


class TestLLMTracingIntegration:
    """Test tracing integration with LLM classes."""

    def test_llm_tracing_disabled_by_default(self):
        """Test that tracing is disabled by default."""
        provider = MockProvider()
        init_tracer(provider)

        llm = MockLLM(enable_tracing=False)
        response = llm.ask("Test prompt")

        assert response.text == "Mock response"
        assert llm.call_count == 1
        # No spans should be created
        assert len(provider.spans) == 0

    def test_llm_tracing_enabled(self):
        """Test LLM tracing when enabled."""
        provider = MockProvider()
        init_tracer(provider)

        llm = MockLLM(enable_tracing=True, model="test-model")
        response = llm.ask("Test prompt")

        assert response.text == "Mock response"
        assert llm.call_count == 1

        # Should have start and end spans
        assert len(provider.spans) == 2
        start_action, start_span = provider.spans[0]
        end_action, end_span = provider.spans[1]

        assert start_action == "start"
        assert end_action == "end"
        assert start_span.name == "MockLLM.ask"
        assert start_span.kind == SpanKind.LLM
        assert start_span.model == "test-model"
        assert "prompt" in start_span.inputs
        assert "temperature" in start_span.inputs
        assert "max_tokens" in start_span.inputs

        # End span should have outputs and token usage
        assert end_span.outputs["text"] == "Mock response"
        assert end_span.prompt_tokens == 10
        assert end_span.completion_tokens == 5
        assert end_span.total_tokens == 15

    @pytest.mark.asyncio
    async def test_llm_async_tracing(self):
        """Test async LLM tracing."""
        provider = MockProvider()
        init_tracer(provider)

        llm = MockLLM(enable_tracing=True, model="async-model")
        response = await llm.ask_async("Async test prompt")

        assert response.text == "Mock response"
        assert llm.call_count == 1

        # Should have start and end spans
        assert len(provider.spans) == 2
        start_action, start_span = provider.spans[0]
        end_action, end_span = provider.spans[1]

        assert start_span.name == "MockLLM.ask"
        assert start_span.model == "async-model"
        assert end_span.outputs["text"] == "Mock response"

    def test_llm_tracing_with_error(self):
        """Test LLM tracing when an error occurs."""
        provider = MockProvider()
        init_tracer(provider)

        llm = MockLLM(enable_tracing=True)

        # Mock an error in _make_ask
        def error_make_ask(*args, **kwargs):
            raise ValueError("Test error")

        llm._make_ask = error_make_ask

        response = llm.ask("Test prompt", raise_errors=False)

        # Should return error response
        assert "Error:" in response.text

        # Should have start and end spans with error
        assert len(provider.spans) == 2
        start_action, start_span = provider.spans[0]
        end_action, end_span = provider.spans[1]

        assert start_span.name == "MockLLM.ask"
        assert end_span.error is not None
        assert isinstance(end_span.error, ValueError)

    def test_llm_tracing_with_choices(self):
        """Test LLM tracing with choices parameter."""
        provider = MockProvider()
        init_tracer(provider)

        llm = MockLLM(enable_tracing=True)
        llm.mock_response = Reply(text="Option A")

        response = llm.ask("Choose an option", choices=["Option A", "Option B", "Option C"])

        assert response.text == "Option A"

        # Check that choices are included in inputs
        assert len(provider.spans) == 2
        start_action, start_span = provider.spans[0]

        assert "prompt" in start_span.inputs
        # Note: choices are processed and added to input_context

    def test_llm_tracing_metadata(self):
        """Test that tracing captures metadata correctly."""
        provider = MockProvider()
        init_tracer(provider)

        llm = MockLLM(enable_tracing=True, model="gpt-4", temperature=0.7, max_tokens=100)

        # Mock raw response
        llm.mock_response = Reply(text="Response text", raw_response={"id": "test-123", "model": "gpt-4"})

        llm.ask("Test prompt")

        assert len(provider.spans) == 2
        start_action, start_span = provider.spans[0]
        end_action, end_span = provider.spans[1]

        # Check inputs
        assert start_span.inputs["temperature"] == 0.7
        assert start_span.inputs["max_tokens"] == 100
        assert start_span.model == "gpt-4"

        # Check outputs and metadata
        assert end_span.outputs["text"] == "Response text"
        assert "raw_response" in end_span.metadata

    def test_tracing_with_context_manager(self):
        """Test manual tracing with context manager."""
        provider = MockProvider()
        tracer = init_tracer(provider)

        llm = MockLLM(enable_tracing=False)  # Disable automatic tracing

        with tracer.trace("custom_workflow", kind=SpanKind.CHAIN) as span:
            # Add custom metadata
            span.metadata["workflow_version"] = "1.0"
            span.tags.append("custom")

            # Manual LLM call
            response1 = llm.ask("First question")
            tracer.add_event("first_call_complete", response_length=len(response1.text))

            # Second call
            response2 = llm.ask("Second question")
            tracer.add_event("second_call_complete", response_length=len(response2.text))

            # Set outputs
            span.outputs["first_response"] = response1.text
            span.outputs["second_response"] = response2.text

        # Should have one span for the workflow
        assert len(provider.spans) == 2  # start and end
        start_action, start_span = provider.spans[0]
        end_action, end_span = provider.spans[1]

        assert start_span.name == "custom_workflow"
        assert start_span.kind == SpanKind.CHAIN
        assert start_span.metadata["workflow_version"] == "1.0"
        assert "custom" in start_span.tags

        assert end_span.outputs["first_response"] == "Mock response"
        assert end_span.outputs["second_response"] == "Mock response"

        # Should have events
        assert len(provider.events) == 2
        span_id1, event_name1, attrs1 = provider.events[0]
        span_id2, event_name2, attrs2 = provider.events[1]

        assert event_name1 == "first_call_complete"
        assert event_name2 == "second_call_complete"
        assert attrs1["response_length"] > 0
        assert attrs2["response_length"] > 0

    def test_nested_tracing(self):
        """Test nested tracing contexts."""
        provider = MockProvider()
        tracer = init_tracer(provider)

        llm = MockLLM(enable_tracing=True)

        with tracer.trace("outer_workflow", kind=SpanKind.CHAIN) as outer_span:
            # This will create nested spans due to LLM tracing
            response = llm.ask("Test prompt")

            with tracer.trace("inner_task", kind=SpanKind.TOOL) as inner_span:
                inner_span.outputs["task_result"] = "completed"

            outer_span.outputs["workflow_result"] = response.text

        # Should have: outer_start, llm_start, llm_end, inner_start, inner_end, outer_end
        assert len(provider.spans) == 6

        # Check nesting
        outer_start = provider.spans[0][1]
        llm_start = provider.spans[1][1]
        inner_start = provider.spans[3][1]

        # LLM span should be child of outer span
        assert llm_start.context.parent_span_id == outer_start.context.span_id
        assert llm_start.context.trace_id == outer_start.context.trace_id

        # Inner span should also be child of outer span
        assert inner_start.context.parent_span_id == outer_start.context.span_id
        assert inner_start.context.trace_id == outer_start.context.trace_id


@pytest.fixture(autouse=True)
def reset_tracer():
    """Reset tracer after each test."""
    yield
    from pyhub.llm.tracing.base import NoOpProvider, init_tracer

    init_tracer(NoOpProvider())
