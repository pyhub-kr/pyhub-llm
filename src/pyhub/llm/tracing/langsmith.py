"""
LangSmith provider for tracing with minimal dependencies.

This module provides LangSmith integration using only httpx and orjson
for optimal performance and minimal dependency footprint.
"""

import asyncio
import logging
import os
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Optional

from .base import SpanData, TracingProvider

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard dependencies
_httpx_available = False
_orjson_available = False

try:
    import httpx

    _httpx_available = True
except ImportError:
    httpx = None

try:
    import orjson

    _orjson_available = True
except ImportError:
    orjson = None


class LangSmithProvider(TracingProvider):
    """
    LangSmith provider with minimal dependencies and efficient batching.

    Features:
    - Automatic batching and background flushing
    - Minimal API surface (httpx + orjson only)
    - Robust error handling
    - Async-friendly design
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: Optional[str] = None,
        api_url: Optional[str] = None,
        batch_size: int = 20,
        flush_interval: float = 2.0,
        timeout: float = 30.0,
    ):
        if not _httpx_available:
            raise ImportError("httpx is required for LangSmith provider. " "Install with: pip install httpx")

        if not _orjson_available:
            raise ImportError("orjson is required for LangSmith provider. " "Install with: pip install orjson")

        # Configuration
        self.api_key = api_key or os.getenv("LANGCHAIN_API_KEY")
        self.project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "default")
        self.api_url = api_url or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

        if not self.api_key:
            raise ValueError(
                "LangSmith API key is required. "
                "Set LANGCHAIN_API_KEY environment variable or pass api_key parameter."
            )

        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.timeout = timeout

        # Internal state
        self._queue: Deque[Dict[str, Any]] = deque()
        self._runs_in_progress: Dict[str, Dict[str, Any]] = {}
        self._client: Optional[httpx.Client] = None
        self._flush_task: Optional[asyncio.Task] = None

        # Initialize HTTP client
        self._init_client()

        # Start background flusher if possible
        self._start_background_flusher()

    def _init_client(self) -> None:
        """Initialize HTTP client with proper headers."""
        self._client = httpx.Client(
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "pyhub-llm-tracer/1.0",
            },
            timeout=self.timeout,
            follow_redirects=True,
        )

    def _start_background_flusher(self) -> None:
        """Start background task to periodically flush the queue."""

        async def flush_loop():
            while True:
                try:
                    await asyncio.sleep(self.flush_interval)
                    self.flush()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in background flush: {e}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._flush_task = loop.create_task(flush_loop())
        except RuntimeError:
            # No event loop running, flush will be manual only
            logger.debug("No event loop for background flushing, using manual flush only")

    def start_span(self, span_data: SpanData) -> None:
        """Queue run creation for LangSmith."""
        run_data = self._convert_to_langsmith_run(span_data, "start")

        # Store for later updates
        self._runs_in_progress[span_data.context.span_id] = run_data

        # Queue for sending
        self._queue.append(run_data)

        # Auto-flush if queue is getting large
        if len(self._queue) >= self.batch_size:
            self.flush()

    def end_span(self, span_data: SpanData) -> None:
        """Queue run update for LangSmith."""
        run_id = span_data.context.span_id

        if run_id in self._runs_in_progress:
            # Update existing run data
            run_data = self._runs_in_progress[run_id]
            run_data.update(self._get_end_data(span_data))

            # Queue updated run
            self._queue.append(run_data.copy())

            # Clean up
            del self._runs_in_progress[run_id]
        else:
            # Create new end run if start wasn't captured
            run_data = self._convert_to_langsmith_run(span_data, "end")
            self._queue.append(run_data)

        # Auto-flush if needed
        if len(self._queue) >= self.batch_size:
            self.flush()

    def add_event(self, span_id: str, name: str, attributes: Dict[str, Any]) -> None:
        """Add event to run as part of extra metadata."""
        if span_id in self._runs_in_progress:
            run_data = self._runs_in_progress[span_id]

            # Add to events list in extra metadata
            if "events" not in run_data.get("extra", {}):
                run_data.setdefault("extra", {})["events"] = []

            event = {"name": name, "time": datetime.utcnow().isoformat(), "attributes": attributes}
            run_data["extra"]["events"].append(event)

    def _convert_to_langsmith_run(self, span_data: SpanData, event_type: str) -> Dict[str, Any]:
        """Convert SpanData to LangSmith run format."""
        run_data = {
            "id": span_data.context.span_id,
            "trace_id": span_data.context.trace_id,
            "parent_run_id": span_data.context.parent_span_id,
            "session_name": self.project_name,
            "name": span_data.name,
            "run_type": span_data.kind.value,
            "start_time": span_data.start_time.isoformat(),
            "inputs": self._sanitize_data(span_data.inputs),
            "tags": span_data.tags,
            "extra": {
                "metadata": span_data.metadata.copy(),
            },
        }

        # Add model information if available
        if span_data.model:
            run_data["extra"]["metadata"]["model"] = span_data.model

        if event_type == "end":
            run_data.update(self._get_end_data(span_data))

        return run_data

    def _get_end_data(self, span_data: SpanData) -> Dict[str, Any]:
        """Get data specific to run completion."""
        end_data = {
            "end_time": span_data.end_time.isoformat() if span_data.end_time else None,
            "outputs": self._sanitize_data(span_data.outputs),
        }

        # Add error information
        if span_data.error:
            end_data["error"] = str(span_data.error)
            end_data["extra"] = end_data.get("extra", {})
            end_data["extra"]["error_type"] = type(span_data.error).__name__

        # Add token usage if available
        if span_data.prompt_tokens is not None:
            token_usage = {
                "prompt_tokens": span_data.prompt_tokens,
                "completion_tokens": span_data.completion_tokens or 0,
                "total_tokens": span_data.total_tokens or span_data.prompt_tokens,
            }
            end_data.setdefault("extra", {}).setdefault("metadata", {})["token_usage"] = token_usage

        return end_data

    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for JSON serialization and size limits."""
        if not data:
            return {}

        sanitized = {}
        for key, value in data.items():
            # Truncate large strings
            if isinstance(value, str) and len(value) > 10000:
                sanitized[key] = value[:10000] + "... [truncated]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            elif isinstance(value, list) and len(value) > 100:
                # Truncate large lists
                sanitized[key] = value[:100] + ["... [truncated]"]
            else:
                sanitized[key] = value

        return sanitized

    def flush(self) -> None:
        """Send queued runs to LangSmith."""
        if not self._queue or not self._client:
            return

        # Collect batch
        batch = []
        while self._queue and len(batch) < self.batch_size:
            batch.append(self._queue.popleft())

        if not batch:
            return

        try:
            # Send to LangSmith
            self._send_batch(batch)
            logger.debug(f"Successfully sent batch of {len(batch)} runs to LangSmith")

        except Exception as e:
            logger.error(f"Failed to send batch to LangSmith: {e}")

            # Re-queue failed items (with limit to prevent infinite growth)
            if len(self._queue) < 1000:  # Prevent memory issues
                self._queue.extendleft(reversed(batch))

    def _send_batch(self, batch: list) -> None:
        """Send a batch of runs to LangSmith API."""
        if not self._client:
            raise RuntimeError("HTTP client not initialized")

        # Prepare payload
        payload = {"runs": batch}

        # Serialize with orjson for performance
        content = orjson.dumps(payload)

        # Send request
        response = self._client.post(
            f"{self.api_url}/runs/batch",
            content=content,
        )

        # Check response
        response.raise_for_status()

        # Log response for debugging
        if response.status_code == 200:
            logger.debug("Batch successfully processed by LangSmith")
        else:
            logger.warning(f"Unexpected response status: {response.status_code}")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            # Cancel background task
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()

            # Final flush
            self.flush()

            # Close HTTP client
            if hasattr(self, "_client") and self._client:
                self._client.close()

        except Exception as e:
            logger.error(f"Error during LangSmith provider cleanup: {e}")


def create_langsmith_provider(**kwargs) -> LangSmithProvider:
    """
    Factory function to create LangSmith provider with validation.

    Args:
        **kwargs: Arguments passed to LangSmithProvider constructor

    Returns:
        LangSmithProvider instance

    Raises:
        ImportError: If required dependencies are not available
        ValueError: If configuration is invalid
    """
    return LangSmithProvider(**kwargs)
