"""
OpenTelemetry provider for tracing with minimal dependencies.

This module provides OpenTelemetry integration using only the opentelemetry-api
package for maximum compatibility and minimal dependency footprint.
"""

import logging
from typing import Any, Dict, Optional

from .base import SpanData, SpanKind, TracingProvider

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard dependencies
_otel_available = False
_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind as OTelSpanKind
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.util.types import Attributes

    _otel_available = True
except ImportError:
    trace = None
    Status = None
    StatusCode = None
    OTelSpanKind = None
    Attributes = None


class OpenTelemetryProvider(TracingProvider):
    """
    OpenTelemetry provider with minimal dependencies.

    Features:
    - Uses only opentelemetry-api (no SDK required)
    - Automatic span kind mapping
    - LLM-specific attribute conventions
    - Robust error handling
    """

    def __init__(
        self,
        service_name: str = "pyhub-llm",
        tracer_name: str = "pyhub.llm",
        version: Optional[str] = None,
    ):
        if not _otel_available:
            raise ImportError(
                "opentelemetry-api is required for OpenTelemetry provider. "
                "Install with: pip install opentelemetry-api"
            )

        self.service_name = service_name
        self.tracer_name = tracer_name
        self.version = version

        # Get tracer from global provider
        global _tracer
        if _tracer is None:
            _tracer = trace.get_tracer(
                instrumenting_module_name=self.tracer_name,
                instrumenting_library_version=self.version,
                schema_url=None,
            )
        self.tracer = _tracer

        # Active spans mapping: span_id -> otel_span
        self._active_spans: Dict[str, Any] = {}

    def start_span(self, span_data: SpanData) -> None:
        """Start an OpenTelemetry span."""
        try:
            # Map span kind
            otel_kind = self._map_span_kind(span_data.kind)

            # Create attributes
            attributes = self._create_attributes(span_data)

            # Create span
            span = self.tracer.start_span(
                name=span_data.name,
                kind=otel_kind,
                attributes=attributes,
                start_time=int(span_data.start_time.timestamp() * 1_000_000_000),  # nanoseconds
            )

            # Store span reference
            self._active_spans[span_data.context.span_id] = span

            logger.debug(f"Started OpenTelemetry span: {span_data.name} ({span_data.context.span_id})")

        except Exception as e:
            logger.error(f"Failed to start OpenTelemetry span: {e}")

    def end_span(self, span_data: SpanData) -> None:
        """End an OpenTelemetry span."""
        span_id = span_data.context.span_id

        if span_id not in self._active_spans:
            logger.warning(f"Span {span_id} not found in active spans")
            return

        try:
            span = self._active_spans[span_id]

            # Update attributes with outputs
            self._add_output_attributes(span, span_data)

            # Add LLM-specific attributes
            self._add_llm_attributes(span, span_data)

            # Set span status
            if span_data.error:
                span.set_status(Status(StatusCode.ERROR, str(span_data.error)))
                span.record_exception(span_data.error)
            else:
                span.set_status(Status(StatusCode.OK))

            # End span with proper timing
            end_time = None
            if span_data.end_time:
                end_time = int(span_data.end_time.timestamp() * 1_000_000_000)  # nanoseconds

            span.end(end_time=end_time)

            logger.debug(f"Ended OpenTelemetry span: {span_data.name} ({span_id})")

        except Exception as e:
            logger.error(f"Failed to end OpenTelemetry span: {e}")
        finally:
            # Clean up reference
            self._active_spans.pop(span_id, None)

    def add_event(self, span_id: str, name: str, attributes: Dict[str, Any]) -> None:
        """Add an event to an OpenTelemetry span."""
        if span_id not in self._active_spans:
            logger.debug(f"Span {span_id} not found for event {name}")
            return

        try:
            span = self._active_spans[span_id]

            # Convert attributes to proper format
            otel_attributes = self._convert_attributes(attributes)

            span.add_event(name, attributes=otel_attributes)

            logger.debug(f"Added event '{name}' to span {span_id}")

        except Exception as e:
            logger.error(f"Failed to add event to OpenTelemetry span: {e}")

    def _map_span_kind(self, kind: SpanKind) -> Any:
        """Map our SpanKind to OpenTelemetry SpanKind."""
        mapping = {
            SpanKind.LLM: OTelSpanKind.CLIENT,
            SpanKind.CHAIN: OTelSpanKind.INTERNAL,
            SpanKind.TOOL: OTelSpanKind.CLIENT,
            SpanKind.RETRIEVER: OTelSpanKind.CLIENT,
            SpanKind.EMBEDDING: OTelSpanKind.CLIENT,
            SpanKind.AGENT: OTelSpanKind.INTERNAL,
        }
        return mapping.get(kind, OTelSpanKind.INTERNAL)

    def _create_attributes(self, span_data: SpanData) -> Attributes:
        """Create OpenTelemetry attributes from span data."""
        attributes = {
            # Standard attributes
            "service.name": self.service_name,
            "span.type": span_data.kind.value,
            # Trace context
            "trace.id": span_data.context.trace_id,
            "span.id": span_data.context.span_id,
        }

        # Parent span ID
        if span_data.context.parent_span_id:
            attributes["parent.span.id"] = span_data.context.parent_span_id

        # Model information
        if span_data.model:
            attributes["llm.model"] = span_data.model

        # Tags as comma-separated string
        if span_data.tags:
            attributes["tags"] = ",".join(span_data.tags)

        # Input attributes (with truncation)
        for key, value in span_data.inputs.items():
            safe_key = f"input.{self._sanitize_key(key)}"
            attributes[safe_key] = self._sanitize_value(value)

        # Metadata attributes
        for key, value in span_data.metadata.items():
            safe_key = f"meta.{self._sanitize_key(key)}"
            attributes[safe_key] = self._sanitize_value(value)

        return self._convert_attributes(attributes)

    def _add_output_attributes(self, span: Any, span_data: SpanData) -> None:
        """Add output attributes to the span."""
        for key, value in span_data.outputs.items():
            safe_key = f"output.{self._sanitize_key(key)}"
            safe_value = self._sanitize_value(value)
            span.set_attribute(safe_key, safe_value)

    def _add_llm_attributes(self, span: Any, span_data: SpanData) -> None:
        """Add LLM-specific attributes following semantic conventions."""
        if span_data.prompt_tokens is not None:
            span.set_attribute("llm.usage.prompt_tokens", span_data.prompt_tokens)

        if span_data.completion_tokens is not None:
            span.set_attribute("llm.usage.completion_tokens", span_data.completion_tokens)

        if span_data.total_tokens is not None:
            span.set_attribute("llm.usage.total_tokens", span_data.total_tokens)

        # Duration in milliseconds
        if span_data.duration_ms is not None:
            span.set_attribute("duration.ms", span_data.duration_ms)

    def _sanitize_key(self, key: str) -> str:
        """Sanitize attribute key for OpenTelemetry."""
        # Replace invalid characters
        sanitized = key.replace(" ", "_").replace("-", "_").replace(".", "_")
        # Ensure it's not too long
        return sanitized[:64]

    def _sanitize_value(self, value: Any) -> str:
        """Sanitize attribute value for OpenTelemetry."""
        if value is None:
            return ""

        str_value = str(value)

        # Truncate long values
        if len(str_value) > 1000:
            return str_value[:1000] + "..."

        return str_value

    def _convert_attributes(self, attributes: Dict[str, Any]) -> Attributes:
        """Convert attributes to OpenTelemetry format."""
        converted = {}

        for key, value in attributes.items():
            if value is None:
                continue

            # OpenTelemetry supports specific types
            if isinstance(value, (str, int, float, bool)):
                converted[key] = value
            elif isinstance(value, list):
                # Convert list to comma-separated string
                converted[key] = ",".join(str(item) for item in value)
            else:
                # Convert everything else to string
                converted[key] = str(value)

        return converted

    def flush(self) -> None:
        """Flush OpenTelemetry data."""
        try:
            # Force flush all active spans (emergency cleanup)
            if self._active_spans:
                logger.warning(f"Force-ending {len(self._active_spans)} active spans during flush")
                for span_id, span in list(self._active_spans.items()):
                    try:
                        span.set_status(Status(StatusCode.ERROR, "Force ended during flush"))
                        span.end()
                    except Exception as e:
                        logger.error(f"Error force-ending span {span_id}: {e}")

                self._active_spans.clear()

            # Try to get tracer provider and force flush
            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, "force_flush"):
                tracer_provider.force_flush(timeout_millis=5000)

        except Exception as e:
            logger.error(f"Error during OpenTelemetry flush: {e}")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.flush()
        except Exception as e:
            logger.error(f"Error during OpenTelemetry provider cleanup: {e}")


def create_opentelemetry_provider(**kwargs) -> OpenTelemetryProvider:
    """
    Factory function to create OpenTelemetry provider with validation.

    Args:
        **kwargs: Arguments passed to OpenTelemetryProvider constructor

    Returns:
        OpenTelemetryProvider instance

    Raises:
        ImportError: If opentelemetry-api is not available
    """
    return OpenTelemetryProvider(**kwargs)
