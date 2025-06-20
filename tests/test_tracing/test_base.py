"""
Tests for the base tracing interface.
"""

from unittest.mock import Mock

import pytest

from pyhub.llm.tracing.base import (
    CompositeProvider,
    NoOpProvider,
    SpanContext,
    SpanData,
    SpanKind,
    Tracer,
    TracingProvider,
    get_tracer,
    init_tracer,
    is_tracing_enabled,
    set_tracer,
)


class TestSpanContext:
    def test_create_root(self):
        """Test creating root span context."""
        context = SpanContext.create_root()

        assert context.trace_id
        assert context.span_id
        assert context.parent_span_id is None
        assert len(context.trace_id) == 32  # UUID hex string
        assert len(context.span_id) == 16

    def test_create_child(self):
        """Test creating child span context."""
        parent = SpanContext.create_root()
        child = parent.create_child()

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.parent_span_id == parent.span_id
        assert len(child.span_id) == 16


class TestSpanData:
    def test_span_data_creation(self):
        """Test SpanData creation and defaults."""
        context = SpanContext.create_root()
        span = SpanData(name="test_span", kind=SpanKind.LLM, context=context)

        assert span.name == "test_span"
        assert span.kind == SpanKind.LLM
        assert span.context == context
        assert span.start_time
        assert span.end_time is None
        assert span.inputs == {}
        assert span.outputs == {}
        assert span.metadata == {}
        assert span.tags == []
        assert span.error is None
        assert span.status == "success"

    def test_finish_method(self):
        """Test SpanData finish method."""
        context = SpanContext.create_root()
        span = SpanData("test", SpanKind.LLM, context)

        # Test successful finish
        outputs = {"result": "success"}
        span.finish(outputs=outputs)

        assert span.end_time is not None
        assert span.outputs["result"] == "success"
        assert span.status == "success"

    def test_finish_with_error(self):
        """Test SpanData finish with error."""
        context = SpanContext.create_root()
        span = SpanData("test", SpanKind.LLM, context)

        error = ValueError("test error")
        span.finish(error=error)

        assert span.end_time is not None
        assert span.error == error
        assert span.status == "error"

    def test_duration_ms(self):
        """Test duration calculation."""
        context = SpanContext.create_root()
        span = SpanData("test", SpanKind.LLM, context)

        # Before finish
        assert span.duration_ms is None

        # After finish
        span.finish()
        assert span.duration_ms is not None
        assert span.duration_ms >= 0


class MockProvider(TracingProvider):
    """Mock provider for testing."""

    def __init__(self):
        self.started_spans = []
        self.ended_spans = []
        self.events = []
        self.flush_called = False

    def start_span(self, span_data: SpanData) -> None:
        self.started_spans.append(span_data)

    def end_span(self, span_data: SpanData) -> None:
        self.ended_spans.append(span_data)

    def add_event(self, span_id: str, name: str, attributes: dict) -> None:
        self.events.append((span_id, name, attributes))

    def flush(self) -> None:
        self.flush_called = True


class TestNoOpProvider:
    def test_noop_provider(self):
        """Test that NoOpProvider does nothing."""
        provider = NoOpProvider()
        context = SpanContext.create_root()
        span = SpanData("test", SpanKind.LLM, context)

        # Should not raise any exceptions
        provider.start_span(span)
        provider.end_span(span)
        provider.add_event("test_id", "test_event", {"key": "value"})
        provider.flush()


class TestCompositeProvider:
    def test_composite_provider(self):
        """Test CompositeProvider delegates to all providers."""
        provider1 = MockProvider()
        provider2 = MockProvider()
        composite = CompositeProvider([provider1, provider2])

        context = SpanContext.create_root()
        span = SpanData("test", SpanKind.LLM, context)

        # Test start_span
        composite.start_span(span)
        assert len(provider1.started_spans) == 1
        assert len(provider2.started_spans) == 1

        # Test end_span
        composite.end_span(span)
        assert len(provider1.ended_spans) == 1
        assert len(provider2.ended_spans) == 1

        # Test add_event
        composite.add_event("test_id", "test_event", {"key": "value"})
        assert len(provider1.events) == 1
        assert len(provider2.events) == 1

        # Test flush
        composite.flush()
        assert provider1.flush_called
        assert provider2.flush_called

    def test_composite_provider_error_handling(self):
        """Test CompositeProvider handles errors gracefully."""
        good_provider = MockProvider()

        # Create a provider that raises errors
        bad_provider = Mock(spec=TracingProvider)
        bad_provider.start_span.side_effect = Exception("provider error")
        bad_provider.end_span.side_effect = Exception("provider error")
        bad_provider.add_event.side_effect = Exception("provider error")
        bad_provider.flush.side_effect = Exception("provider error")

        composite = CompositeProvider([good_provider, bad_provider])

        context = SpanContext.create_root()
        span = SpanData("test", SpanKind.LLM, context)

        # Should not raise exceptions, but good provider should still work
        composite.start_span(span)
        assert len(good_provider.started_spans) == 1

        composite.end_span(span)
        assert len(good_provider.ended_spans) == 1

        composite.add_event("test_id", "test_event", {"key": "value"})
        assert len(good_provider.events) == 1

        composite.flush()
        assert good_provider.flush_called


class TestTracer:
    def test_tracer_with_noop_provider(self):
        """Test tracer with NoOpProvider."""
        tracer = Tracer(NoOpProvider())

        assert tracer.current_context is None

        with tracer.trace("test_span") as span:
            assert span.name == "test_span"
            assert span.kind == SpanKind.LLM
            assert tracer.current_context is not None

        # Context should be cleared after exiting
        assert tracer.current_context is None

    def test_tracer_context_management(self):
        """Test tracer context stack management."""
        provider = MockProvider()
        tracer = Tracer(provider)

        with tracer.trace("parent_span") as parent_span:
            assert tracer.current_context.span_id == parent_span.context.span_id

            with tracer.trace("child_span") as child_span:
                assert tracer.current_context.span_id == child_span.context.span_id
                assert child_span.context.parent_span_id == parent_span.context.span_id
                assert child_span.context.trace_id == parent_span.context.trace_id

            # Should return to parent context
            assert tracer.current_context.span_id == parent_span.context.span_id

        # Should be no active context
        assert tracer.current_context is None

        # Check provider was called correctly
        assert len(provider.started_spans) == 2
        assert len(provider.ended_spans) == 2

    def test_tracer_error_handling(self):
        """Test tracer handles errors in context."""
        provider = MockProvider()
        tracer = Tracer(provider)

        with pytest.raises(ValueError):
            with tracer.trace("error_span"):
                raise ValueError("test error")

        # Span should be finished with error
        assert len(provider.ended_spans) == 1
        ended_span = provider.ended_spans[0]
        assert ended_span.error is not None
        assert ended_span.status == "error"
        assert tracer.current_context is None

    def test_trace_function_decorator(self):
        """Test trace_function decorator."""
        provider = MockProvider()
        tracer = Tracer(provider)

        @tracer.trace_function(name="decorated_function", kind=SpanKind.TOOL)
        def test_function(x, y, keyword_arg="default"):
            return x + y

        result = test_function(1, 2, keyword_arg="test")

        assert result == 3
        assert len(provider.started_spans) == 1
        assert len(provider.ended_spans) == 1

        span = provider.ended_spans[0]
        assert span.name == "decorated_function"
        assert span.kind == SpanKind.TOOL
        assert "args" in span.inputs
        assert "kwargs" in span.inputs
        assert "result" in span.outputs

    def test_add_event(self):
        """Test adding events to spans."""
        provider = MockProvider()
        tracer = Tracer(provider)

        with tracer.trace("test_span") as span:
            tracer.add_event("test_event", key1="value1", key2="value2")

        assert len(provider.events) == 1
        span_id, event_name, attributes = provider.events[0]
        assert span_id == span.context.span_id
        assert event_name == "test_event"
        assert attributes["key1"] == "value1"
        assert attributes["key2"] == "value2"

    def test_add_event_without_active_span(self):
        """Test adding event when no span is active."""
        provider = MockProvider()
        tracer = Tracer(provider)

        # Should not raise error, but should not add event
        tracer.add_event("test_event", key="value")
        assert len(provider.events) == 0


class TestGlobalTracer:
    def test_get_tracer_default(self):
        """Test get_tracer returns default tracer."""
        tracer = get_tracer()
        assert isinstance(tracer, Tracer)
        assert isinstance(tracer.provider, NoOpProvider)

    def test_init_tracer(self):
        """Test init_tracer sets global tracer."""
        provider = MockProvider()
        tracer = init_tracer(provider)

        assert isinstance(tracer, Tracer)
        assert tracer.provider == provider

        # get_tracer should return the same instance
        assert get_tracer() is tracer

    def test_set_tracer(self):
        """Test set_tracer directly sets tracer."""
        provider = MockProvider()
        tracer = Tracer(provider)

        set_tracer(tracer)
        assert get_tracer() is tracer

    def test_is_tracing_enabled(self):
        """Test is_tracing_enabled function."""
        # With NoOpProvider
        init_tracer(NoOpProvider())
        assert not is_tracing_enabled()

        # With real provider
        init_tracer(MockProvider())
        assert is_tracing_enabled()


class TestSpanKind:
    def test_span_kind_values(self):
        """Test SpanKind enum values."""
        assert SpanKind.LLM.value == "llm"
        assert SpanKind.CHAIN.value == "chain"
        assert SpanKind.TOOL.value == "tool"
        assert SpanKind.RETRIEVER.value == "retriever"
        assert SpanKind.EMBEDDING.value == "embedding"
        assert SpanKind.AGENT.value == "agent"


@pytest.fixture(autouse=True)
def reset_global_tracer():
    """Reset global tracer after each test."""
    yield
    # Reset to default NoOpProvider tracer
    init_tracer(NoOpProvider())
