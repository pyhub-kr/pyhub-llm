"""
Observability and tracing module for pyhub-llm.

This module provides a minimal, extensible interface for observability that supports
multiple backends including LangSmith and OpenTelemetry.

Features:
- Zero dependencies for core interface
- Automatic LLM call tracing
- Context managers and decorators
- Multi-provider support
- Environment-based auto-configuration
"""

import logging
import os
from typing import List, Optional

from .base import (
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

logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"


def auto_configure_tracing() -> Optional[Tracer]:
    """
    Auto-configure tracing based on environment variables.

    Environment Variables:
        PYHUB_LLM_TRACE: Enable tracing ("true", "1", "yes")
        LANGCHAIN_API_KEY: LangSmith API key
        LANGCHAIN_PROJECT: LangSmith project name
        LANGCHAIN_ENDPOINT: LangSmith API endpoint
        OTEL_EXPORTER_OTLP_ENDPOINT: OpenTelemetry collector endpoint
        OTEL_SERVICE_NAME: Service name for OpenTelemetry

    Returns:
        Tracer instance or None if tracing is disabled
    """
    # Check if tracing is enabled
    trace_enabled = os.getenv("PYHUB_LLM_TRACE", "").lower() in ("true", "1", "yes")

    if not trace_enabled:
        logger.debug("Tracing disabled via PYHUB_LLM_TRACE")
        return init_tracer(NoOpProvider())

    providers: List[TracingProvider] = []

    # Configure LangSmith if available
    if os.getenv("LANGCHAIN_API_KEY"):
        try:
            from .langsmith import LangSmithProvider

            provider = LangSmithProvider(
                api_key=os.getenv("LANGCHAIN_API_KEY"),
                project_name=os.getenv("LANGCHAIN_PROJECT"),
                api_url=os.getenv("LANGCHAIN_ENDPOINT"),
            )
            providers.append(provider)
            logger.info("LangSmith tracing provider configured")

        except ImportError as e:
            logger.warning(f"LangSmith provider not available: {e}")
        except Exception as e:
            logger.error(f"Failed to configure LangSmith provider: {e}")

    # Configure OpenTelemetry if available
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("OTEL_SERVICE_NAME"):
        try:
            from .opentelemetry import OpenTelemetryProvider

            provider = OpenTelemetryProvider(
                service_name=os.getenv("OTEL_SERVICE_NAME", "pyhub-llm"),
            )
            providers.append(provider)
            logger.info("OpenTelemetry tracing provider configured")

        except ImportError as e:
            logger.warning(f"OpenTelemetry provider not available: {e}")
        except Exception as e:
            logger.error(f"Failed to configure OpenTelemetry provider: {e}")

    # Initialize tracer based on available providers
    if not providers:
        logger.warning("No tracing providers configured, using no-op provider")
        return init_tracer(NoOpProvider())
    elif len(providers) == 1:
        logger.info(f"Initialized tracing with {providers[0].__class__.__name__}")
        return init_tracer(providers[0])
    else:
        logger.info(f"Initialized tracing with {len(providers)} providers: {[p.__class__.__name__ for p in providers]}")
        return init_tracer(CompositeProvider(providers))


def create_langsmith_provider(
    api_key: Optional[str] = None, project_name: Optional[str] = None, **kwargs
) -> TracingProvider:
    """
    Create a LangSmith tracing provider.

    Args:
        api_key: LangSmith API key (defaults to LANGCHAIN_API_KEY env var)
        project_name: Project name (defaults to LANGCHAIN_PROJECT env var)
        **kwargs: Additional arguments for LangSmithProvider

    Returns:
        LangSmithProvider instance

    Raises:
        ImportError: If required dependencies are not available
        ValueError: If configuration is invalid
    """
    try:
        from .langsmith import LangSmithProvider

        return LangSmithProvider(api_key=api_key, project_name=project_name, **kwargs)
    except ImportError:
        raise ImportError(
            "LangSmith provider requires additional dependencies. " "Install with: pip install httpx orjson"
        )


def create_opentelemetry_provider(service_name: str = "pyhub-llm", **kwargs) -> TracingProvider:
    """
    Create an OpenTelemetry tracing provider.

    Args:
        service_name: Service name for traces
        **kwargs: Additional arguments for OpenTelemetryProvider

    Returns:
        OpenTelemetryProvider instance

    Raises:
        ImportError: If opentelemetry-api is not available
    """
    try:
        from .opentelemetry import OpenTelemetryProvider

        return OpenTelemetryProvider(service_name=service_name, **kwargs)
    except ImportError:
        raise ImportError(
            "OpenTelemetry provider requires opentelemetry-api. " "Install with: pip install opentelemetry-api"
        )


def setup_tracing(
    providers: Optional[List[str]] = None,
    langsmith_config: Optional[dict] = None,
    opentelemetry_config: Optional[dict] = None,
) -> Tracer:
    """
    Setup tracing with specific providers and configurations.

    Args:
        providers: List of provider names ("langsmith", "opentelemetry")
        langsmith_config: Configuration for LangSmith provider
        opentelemetry_config: Configuration for OpenTelemetry provider

    Returns:
        Configured Tracer instance

    Example:
        >>> tracer = setup_tracing(
        ...     providers=["langsmith", "opentelemetry"],
        ...     langsmith_config={"project_name": "my-project"},
        ...     opentelemetry_config={"service_name": "my-service"}
        ... )
    """
    if not providers:
        providers = ["langsmith", "opentelemetry"]

    configured_providers: List[TracingProvider] = []

    # Configure requested providers
    for provider_name in providers:
        if provider_name.lower() == "langsmith":
            try:
                config = langsmith_config or {}
                provider = create_langsmith_provider(**config)
                configured_providers.append(provider)
            except Exception as e:
                logger.error(f"Failed to configure LangSmith: {e}")

        elif provider_name.lower() == "opentelemetry":
            try:
                config = opentelemetry_config or {}
                provider = create_opentelemetry_provider(**config)
                configured_providers.append(provider)
            except Exception as e:
                logger.error(f"Failed to configure OpenTelemetry: {e}")

        else:
            logger.warning(f"Unknown provider: {provider_name}")

    # Initialize tracer
    if not configured_providers:
        logger.warning("No providers configured, using no-op tracer")
        return init_tracer(NoOpProvider())
    elif len(configured_providers) == 1:
        return init_tracer(configured_providers[0])
    else:
        return init_tracer(CompositeProvider(configured_providers))


# Export main interface
__all__ = [
    # Core classes
    "Tracer",
    "TracingProvider",
    "SpanKind",
    "SpanData",
    "SpanContext",
    "NoOpProvider",
    "CompositeProvider",
    # Factory functions
    "get_tracer",
    "init_tracer",
    "set_tracer",
    "is_tracing_enabled",
    "auto_configure_tracing",
    "setup_tracing",
    "create_langsmith_provider",
    "create_opentelemetry_provider",
    # Version
    "__version__",
]
