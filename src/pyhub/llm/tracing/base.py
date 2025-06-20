"""
Core tracing interface with zero dependencies.

This module provides the foundation for observability in pyhub-llm with:
- Provider-agnostic tracing interface
- Context management for nested spans
- Decorator pattern for function tracing
- Zero overhead when disabled
"""

import logging
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SpanKind(Enum):
    """Type of span/run for categorizing different operations."""

    LLM = "llm"
    CHAIN = "chain"
    TOOL = "tool"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"
    AGENT = "agent"


@dataclass
class SpanContext:
    """Context information for a span/run."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None

    @classmethod
    def create_root(cls) -> "SpanContext":
        """Create a new root context."""
        trace_id = uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        return cls(trace_id=trace_id, span_id=span_id)

    def create_child(self) -> "SpanContext":
        """Create a child context from this context."""
        span_id = uuid.uuid4().hex[:16]
        return SpanContext(trace_id=self.trace_id, span_id=span_id, parent_span_id=self.span_id)


@dataclass
class SpanData:
    """Data collected for a span during its lifetime."""

    name: str
    kind: SpanKind
    context: SpanContext
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    # Input/Output data
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Metadata and categorization
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Error tracking
    error: Optional[Exception] = None
    status: str = "success"

    # LLM-specific metrics
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def finish(self, outputs: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None) -> None:
        """Mark span as finished and update final data."""
        self.end_time = datetime.utcnow()
        if outputs:
            self.outputs.update(outputs)
        if error:
            self.error = error
            self.status = "error"

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate duration in milliseconds."""
        if not self.end_time:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


class TracingProvider(ABC):
    """Abstract base class for tracing providers."""

    @abstractmethod
    def start_span(self, span_data: SpanData) -> None:
        """Called when a span starts."""
        pass

    @abstractmethod
    def end_span(self, span_data: SpanData) -> None:
        """Called when a span ends."""
        pass

    @abstractmethod
    def add_event(self, span_id: str, name: str, attributes: Dict[str, Any]) -> None:
        """Add an event to a span."""
        pass

    def flush(self) -> None:
        """Flush any pending data. Override if needed."""
        pass


class NoOpProvider(TracingProvider):
    """No-operation provider for when tracing is disabled."""

    def start_span(self, span_data: SpanData) -> None:
        pass

    def end_span(self, span_data: SpanData) -> None:
        pass

    def add_event(self, span_id: str, name: str, attributes: Dict[str, Any]) -> None:
        pass


class CompositeProvider(TracingProvider):
    """Provider that delegates to multiple providers simultaneously."""

    def __init__(self, providers: List[TracingProvider]):
        self.providers = providers

    def start_span(self, span_data: SpanData) -> None:
        for provider in self.providers:
            try:
                provider.start_span(span_data)
            except Exception as e:
                logger.error(f"Error starting span in {provider.__class__.__name__}: {e}")

    def end_span(self, span_data: SpanData) -> None:
        for provider in self.providers:
            try:
                provider.end_span(span_data)
            except Exception as e:
                logger.error(f"Error ending span in {provider.__class__.__name__}: {e}")

    def add_event(self, span_id: str, name: str, attributes: Dict[str, Any]) -> None:
        for provider in self.providers:
            try:
                provider.add_event(span_id, name, attributes)
            except Exception as e:
                logger.error(f"Error adding event in {provider.__class__.__name__}: {e}")

    def flush(self) -> None:
        for provider in self.providers:
            try:
                provider.flush()
            except Exception as e:
                logger.error(f"Error flushing {provider.__class__.__name__}: {e}")


class Tracer:
    """Main tracer interface for managing spans and contexts."""

    def __init__(self, provider: Optional[TracingProvider] = None):
        self.provider = provider or NoOpProvider()
        self._context_stack: List[SpanContext] = []

    @property
    def current_context(self) -> Optional[SpanContext]:
        """Get the current active span context."""
        return self._context_stack[-1] if self._context_stack else None

    @contextmanager
    def trace(self, name: str, kind: SpanKind = SpanKind.LLM, **kwargs):
        """Context manager for tracing a block of code."""
        # Create appropriate context
        if self.current_context:
            context = self.current_context.create_child()
        else:
            context = SpanContext.create_root()

        # Create span data
        span_data = SpanData(
            name=name,
            kind=kind,
            context=context,
            metadata=kwargs.get("metadata", {}),
            tags=kwargs.get("tags", []),
            inputs=kwargs.get("inputs", {}),
            model=kwargs.get("model"),
        )

        # Manage context stack
        self._context_stack.append(context)
        self.provider.start_span(span_data)

        try:
            yield span_data
        except Exception as e:
            span_data.finish(error=e)
            raise
        finally:
            # Ensure span is finished
            if not span_data.end_time:
                span_data.finish()
            self.provider.end_span(span_data)
            self._context_stack.pop()

    def trace_function(
        self,
        name: Optional[str] = None,
        kind: SpanKind = SpanKind.LLM,
        capture_input: bool = True,
        capture_output: bool = True,
    ):
        """Decorator for automatically tracing function calls."""

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            trace_name = name or f"{func.__module__}.{func.__name__}"

            def wrapper(*args, **kwargs):
                inputs = {}
                if capture_input:
                    # Capture inputs with truncation for safety
                    inputs = {"args": str(args)[:1000] if args else "", "kwargs": str(kwargs)[:1000] if kwargs else ""}

                with self.trace(trace_name, kind=kind, inputs=inputs) as span:
                    try:
                        result = func(*args, **kwargs)
                        if capture_output and result is not None:
                            span.outputs["result"] = str(result)[:1000]
                        return result
                    except Exception as e:
                        span.error = e
                        raise

            return wrapper

        return decorator

    def add_event(self, name: str, **attributes) -> None:
        """Add an event to the current active span."""
        if self.current_context:
            self.provider.add_event(self.current_context.span_id, name, attributes)
        else:
            logger.debug(f"No active span to add event '{name}' to")

    def flush(self) -> None:
        """Flush any pending trace data."""
        self.provider.flush()


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(NoOpProvider())
    return _tracer


def init_tracer(provider: TracingProvider) -> Tracer:
    """Initialize the global tracer with a specific provider."""
    global _tracer
    _tracer = Tracer(provider)
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    """Set the global tracer instance directly."""
    global _tracer
    _tracer = tracer


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    tracer = get_tracer()
    return not isinstance(tracer.provider, NoOpProvider)
