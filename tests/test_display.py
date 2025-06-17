"""Tests for display utilities."""
import sys
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from pyhub.llm.display import display, print_stream
from pyhub.llm.types import Reply, ChainReply, Usage


class TestDisplay:
    """Test display function."""

    def test_display_reply_plain(self, capsys):
        """Test displaying Reply object as plain text."""
        reply = Reply(text="Hello, world!")
        result = display(reply, markdown=False)
        
        captured = capsys.readouterr()
        assert captured.out == "Hello, world!\n"
        assert result == "Hello, world!"

    def test_display_reply_with_usage(self, capsys):
        """Test displaying Reply with usage information."""
        reply = Reply(
            text="Test response",
            usage=Usage(input=10, output=5)
        )
        result = display(reply, markdown=False)
        
        captured = capsys.readouterr()
        assert captured.out == "Test response\n"
        assert result == "Test response"

    def test_display_chain_reply(self, capsys):
        """Test displaying ChainReply object."""
        chain_reply = ChainReply(
            reply_list=[
                Reply(text="First"),
                Reply(text="Second"),
                Reply(text="Final response")
            ]
        )
        result = display(chain_reply, markdown=False)
        
        captured = capsys.readouterr()
        assert captured.out == "Final response\n"
        assert result == "Final response"

    def test_display_generator_plain(self, capsys):
        """Test displaying streaming generator as plain text."""
        def generator():
            yield "Hello"
            yield " "
            yield "world!"
        
        result = display(generator(), markdown=False)
        
        captured = capsys.readouterr()
        assert captured.out == "Hello world!\n"
        assert result == "Hello world!"

    def test_display_string(self, capsys):
        """Test displaying plain string."""
        result = display("Plain text", markdown=False)
        
        captured = capsys.readouterr()
        assert captured.out == "Plain text\n"
        assert result == "Plain text"

    def test_print_stream(self, capsys):
        """Test print_stream convenience function."""
        def generator():
            yield "Streaming"
            yield " text"
        
        result = print_stream(generator(), markdown=False)
        
        captured = capsys.readouterr()
        assert captured.out == "Streaming text\n"
        assert result == "Streaming text"

    @patch('pyhub.llm.display.HAS_RICH', True)
    @patch('pyhub.llm.display.Console')
    @patch('pyhub.llm.display.Markdown')
    def test_display_with_markdown(self, mock_markdown, mock_console):
        """Test display with markdown rendering."""
        # Setup mocks
        console_instance = Mock()
        mock_console.return_value = console_instance
        
        reply = Reply(text="# Hello\n**world**")
        result = display(reply, markdown=True)
        
        # Verify markdown was used
        mock_console.assert_called_once()
        mock_markdown.assert_called_once_with("# Hello\n**world**", code_theme="monokai", style="none")
        console_instance.print.assert_called_once()
        assert result == "# Hello\n**world**"

    @patch('pyhub.llm.display.HAS_RICH', False)
    def test_display_markdown_without_rich(self, capsys):
        """Test markdown fallback when Rich is not installed."""
        reply = Reply(text="# Hello")
        
        # Capture stderr for warning
        result = display(reply, markdown=True)
        
        captured = capsys.readouterr()
        assert "Warning: Rich library not installed" in captured.err
        assert captured.out == "# Hello\n"
        assert result == "# Hello"

    def test_reply_print_method(self, capsys):
        """Test Reply.print() method."""
        reply = Reply(text="Test print method")
        result = reply.print(markdown=False)
        
        captured = capsys.readouterr()
        assert captured.out == "Test print method\n"
        assert result == "Test print method"

    def test_chain_reply_print_method(self, capsys):
        """Test ChainReply.print() method."""
        chain = ChainReply(
            reply_list=[Reply(text="Final text")]
        )
        result = chain.print(markdown=False)
        
        captured = capsys.readouterr()
        assert captured.out == "Final text\n"
        assert result == "Final text"


class TestStreamingDisplay:
    """Test streaming display functionality."""

    @patch('pyhub.llm.display.HAS_RICH', True)
    @patch('pyhub.llm.display.Console')
    @patch('pyhub.llm.display.Live')
    @patch('pyhub.llm.display.Markdown')
    def test_stream_markdown_display(self, mock_markdown, mock_live, mock_console):
        """Test streaming with markdown rendering."""
        def generator():
            yield "# Title\n"
            yield "Some "
            yield "content"
        
        # Setup mocks
        console_instance = Mock()
        mock_console.return_value = console_instance
        live_instance = Mock()
        mock_live.return_value.__enter__ = Mock(return_value=live_instance)
        mock_live.return_value.__exit__ = Mock(return_value=None)
        
        result = display(generator(), markdown=True)
        
        # Verify Live was used for streaming
        mock_console.assert_called_once()
        mock_live.assert_called_once()
        assert result == "# Title\nSome content"

    def test_generator_exhaustion(self, capsys):
        """Test that generator is properly exhausted."""
        exhausted = False
        
        def generator():
            nonlocal exhausted
            yield "Test"
            exhausted = True
        
        display(generator(), markdown=False)
        assert exhausted, "Generator should be exhausted"


class TestJupyterSupport:
    """Test Jupyter notebook support."""

    @patch('pyhub.llm.display._is_jupyter', return_value=True)
    @patch('IPython.display.display')
    @patch('IPython.display.Markdown')
    @patch('IPython.display.clear_output')
    def test_display_jupyter_markdown(self, mock_clear, mock_markdown, mock_display, mock_is_jupyter):
        """Test display in Jupyter with markdown."""
        from pyhub.llm.display import display_jupyter
        
        reply = Reply(text="# Jupyter Test")
        result = display_jupyter(reply, markdown=True)
        
        mock_markdown.assert_called_once_with("# Jupyter Test")
        mock_display.assert_called_once()
        assert result == "# Jupyter Test"

    @patch('pyhub.llm.display._is_jupyter', return_value=True)
    @patch('IPython.display.display')
    @patch('IPython.display.Markdown')
    @patch('IPython.display.clear_output')
    def test_display_jupyter_streaming(self, mock_clear, mock_markdown, mock_display, mock_is_jupyter):
        """Test streaming display in Jupyter."""
        from pyhub.llm.display import display_jupyter
        
        def generator():
            yield "Part 1"
            yield " Part 2"
        
        result = display_jupyter(generator(), markdown=True)
        
        # Should clear output and update for each chunk
        assert mock_clear.call_count == 2
        assert mock_display.call_count == 2
        assert result == "Part 1 Part 2"