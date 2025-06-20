"""
Tests for auto-configuration functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from pyhub.llm.tracing import auto_configure_tracing
from pyhub.llm.tracing.base import CompositeProvider, NoOpProvider


class TestAutoConfiguration:
    """Test auto-configuration based on environment variables."""

    def test_auto_config_disabled(self):
        """Test auto-configuration when tracing is disabled."""
        with patch.dict(os.environ, {}, clear=True):
            tracer = auto_configure_tracing()

            assert tracer is not None
            assert isinstance(tracer.provider, NoOpProvider)

    def test_auto_config_enabled_no_providers(self):
        """Test auto-configuration when enabled but no providers available."""
        with patch.dict(os.environ, {"PYHUB_LLM_TRACE": "true"}, clear=True):
            tracer = auto_configure_tracing()

            assert tracer is not None
            assert isinstance(tracer.provider, NoOpProvider)

    @patch("pyhub.llm.tracing.langsmith.LangSmithProvider")
    def test_auto_config_langsmith_only(self, mock_langsmith_class):
        """Test auto-configuration with only LangSmith."""
        mock_provider = Mock()
        mock_langsmith_class.return_value = mock_provider

        env_vars = {"PYHUB_LLM_TRACE": "true", "LANGCHAIN_API_KEY": "test-key", "LANGCHAIN_PROJECT": "test-project"}

        with patch.dict(os.environ, env_vars, clear=True):
            tracer = auto_configure_tracing()

            assert tracer is not None
            assert tracer.provider == mock_provider

            # Verify LangSmith provider was created with correct args
            mock_langsmith_class.assert_called_once_with(
                api_key="test-key",
                project_name="test-project",
                api_url=None,
            )

    @patch("pyhub.llm.tracing.opentelemetry.OpenTelemetryProvider")
    def test_auto_config_opentelemetry_only(self, mock_otel_class):
        """Test auto-configuration with only OpenTelemetry."""
        mock_provider = Mock()
        mock_otel_class.return_value = mock_provider

        env_vars = {
            "PYHUB_LLM_TRACE": "true",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
            "OTEL_SERVICE_NAME": "test-service",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            tracer = auto_configure_tracing()

            assert tracer is not None
            assert tracer.provider == mock_provider

            # Verify OpenTelemetry provider was created with correct args
            mock_otel_class.assert_called_once_with(
                service_name="test-service",
            )

    @patch("pyhub.llm.tracing.opentelemetry.OpenTelemetryProvider")
    @patch("pyhub.llm.tracing.langsmith.LangSmithProvider")
    def test_auto_config_multiple_providers(self, mock_langsmith_class, mock_otel_class):
        """Test auto-configuration with multiple providers."""
        mock_langsmith = Mock()
        mock_otel = Mock()
        mock_langsmith_class.return_value = mock_langsmith
        mock_otel_class.return_value = mock_otel

        env_vars = {
            "PYHUB_LLM_TRACE": "true",
            "LANGCHAIN_API_KEY": "test-key",
            "LANGCHAIN_PROJECT": "test-project",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
            "OTEL_SERVICE_NAME": "test-service",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            tracer = auto_configure_tracing()

            assert tracer is not None
            assert isinstance(tracer.provider, CompositeProvider)
            assert len(tracer.provider.providers) == 2
            assert mock_langsmith in tracer.provider.providers
            assert mock_otel in tracer.provider.providers

    def test_auto_config_trace_values(self):
        """Test different values for PYHUB_LLM_TRACE."""
        test_cases = [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("TRUE", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("", False),
            ("invalid", False),
        ]

        for env_value, should_enable in test_cases:
            with patch.dict(os.environ, {"PYHUB_LLM_TRACE": env_value}, clear=True):
                tracer = auto_configure_tracing()

                if should_enable:
                    # Even if enabled, should be NoOp without providers
                    assert isinstance(tracer.provider, NoOpProvider)
                else:
                    assert isinstance(tracer.provider, NoOpProvider)

    @patch("pyhub.llm.tracing.langsmith.LangSmithProvider")
    def test_auto_config_langsmith_import_error(self, mock_langsmith_class):
        """Test auto-configuration handles LangSmith import errors."""
        mock_langsmith_class.side_effect = ImportError("Missing dependency")

        env_vars = {"PYHUB_LLM_TRACE": "true", "LANGCHAIN_API_KEY": "test-key"}

        with patch.dict(os.environ, env_vars, clear=True):
            tracer = auto_configure_tracing()

            # Should fallback to NoOp provider
            assert isinstance(tracer.provider, NoOpProvider)

    @patch("pyhub.llm.tracing.langsmith.LangSmithProvider")
    def test_auto_config_langsmith_config_error(self, mock_langsmith_class):
        """Test auto-configuration handles LangSmith configuration errors."""
        mock_langsmith_class.side_effect = ValueError("Invalid API key")

        env_vars = {"PYHUB_LLM_TRACE": "true", "LANGCHAIN_API_KEY": "invalid-key"}

        with patch.dict(os.environ, env_vars, clear=True):
            tracer = auto_configure_tracing()

            # Should fallback to NoOp provider
            assert isinstance(tracer.provider, NoOpProvider)

    @patch("pyhub.llm.tracing.opentelemetry.OpenTelemetryProvider")
    def test_auto_config_opentelemetry_import_error(self, mock_otel_class):
        """Test auto-configuration handles OpenTelemetry import errors."""
        mock_otel_class.side_effect = ImportError("Missing opentelemetry-api")

        env_vars = {"PYHUB_LLM_TRACE": "true", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"}

        with patch.dict(os.environ, env_vars, clear=True):
            tracer = auto_configure_tracing()

            # Should fallback to NoOp provider
            assert isinstance(tracer.provider, NoOpProvider)

    def test_auto_config_partial_langsmith_config(self):
        """Test auto-configuration with partial LangSmith configuration."""
        # Missing LANGCHAIN_PROJECT (should use default)
        env_vars = {"PYHUB_LLM_TRACE": "true", "LANGCHAIN_API_KEY": "test-key"}

        with patch("pyhub.llm.tracing.langsmith.LangSmithProvider") as mock_langsmith_class:
            mock_provider = Mock()
            mock_langsmith_class.return_value = mock_provider

            with patch.dict(os.environ, env_vars, clear=True):
                tracer = auto_configure_tracing()

                assert tracer.provider == mock_provider
                mock_langsmith_class.assert_called_once_with(
                    api_key="test-key",
                    project_name=None,  # Will use default from environment or provider
                    api_url=None,
                )

    def test_auto_config_opentelemetry_service_name_only(self):
        """Test auto-configuration with only OTEL_SERVICE_NAME."""
        env_vars = {"PYHUB_LLM_TRACE": "true", "OTEL_SERVICE_NAME": "my-service"}

        with patch("pyhub.llm.tracing.opentelemetry.OpenTelemetryProvider") as mock_otel_class:
            mock_provider = Mock()
            mock_otel_class.return_value = mock_provider

            with patch.dict(os.environ, env_vars, clear=True):
                tracer = auto_configure_tracing()

                assert tracer.provider == mock_provider
                mock_otel_class.assert_called_once_with(
                    service_name="my-service",
                )


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment and tracer after each test."""
    yield
    # Reset tracer to default
    from pyhub.llm.tracing.base import NoOpProvider, init_tracer

    init_tracer(NoOpProvider())
