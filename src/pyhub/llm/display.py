"""Display utilities for pretty output of LLM responses."""
import sys
from typing import Union, Generator, Optional, Any
from dataclasses import dataclass

from .types import Reply, ChainReply

# Rich 라이브러리 임포트 (optional dependency)
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.syntax import Syntax
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def display(
    response: Union[Reply, ChainReply, Generator[str, None, None], Any],
    markdown: bool = True,
    style: Optional[str] = "none",
    code_theme: str = "monokai",
    console: Optional[Any] = None,
    **kwargs
) -> str:
    """
    Display LLM response with optional markdown rendering.
    
    This function provides a unified way to display both streaming and non-streaming
    responses with optional markdown formatting using Rich library.
    
    Args:
        response: LLM response object or generator
        markdown: Whether to render as markdown (requires Rich)
        style: Markdown style (if applicable)
        code_theme: Theme for code blocks in markdown
        console: Rich Console instance (optional)
        **kwargs: Additional arguments passed to rendering
        
    Returns:
        str: The complete text content
        
    Examples:
        >>> # Display streaming response
        >>> response = llm.ask("Hello", stream=True)
        >>> text = display(response)
        
        >>> # Display with plain text
        >>> response = llm.ask("Hello")
        >>> text = display(response, markdown=False)
        
        >>> # Custom console
        >>> from rich.console import Console
        >>> console = Console(width=120)
        >>> display(response, console=console)
    """
    # Check if it's a generator (streaming response)
    is_generator = hasattr(response, '__iter__') and not hasattr(response, 'text')
    
    if markdown and not HAS_RICH:
        print("Warning: Rich library not installed. Falling back to plain text output.", file=sys.stderr)
        print("Install with: pip install pyhub-llm[rich]", file=sys.stderr)
        markdown = False
    
    if is_generator:
        # Handle streaming response
        if markdown and HAS_RICH:
            return _display_stream_markdown(response, console, style, code_theme, **kwargs)
        else:
            return _display_stream_plain(response)
    else:
        # Handle completion response
        if hasattr(response, 'text'):
            text = response.text
        else:
            text = str(response)
            
        if markdown and HAS_RICH:
            return _display_markdown(text, console, style, code_theme, **kwargs)
        else:
            print(text)
            return text


def _display_markdown(
    text: str,
    console: Optional[Any],
    style: Optional[str],
    code_theme: str,
    **kwargs
) -> str:
    """Display text as markdown."""
    if console is None:
        console = Console()
    
    md = Markdown(text, code_theme=code_theme, style=style)
    console.print(md)
    return text


def _display_stream_markdown(
    generator: Generator[str, None, None],
    console: Optional[Any],
    style: Optional[str],
    code_theme: str,
    **kwargs
) -> str:
    """Display streaming text as markdown with live updates."""
    if console is None:
        console = Console()
    
    text = ""
    with Live(console=console, refresh_per_second=10, **kwargs) as live:
        for chunk in generator:
            # Handle both string chunks and Reply objects
            if hasattr(chunk, 'text'):
                text += chunk.text
            else:
                text += chunk
            md = Markdown(text, code_theme=code_theme, style=style)
            live.update(md)
    
    return text


def _display_stream_plain(generator: Generator[str, None, None]) -> str:
    """Display streaming text as plain text."""
    text = ""
    for chunk in generator:
        # Handle both string chunks and Reply objects
        if hasattr(chunk, 'text'):
            print(chunk.text, end="", flush=True)
            text += chunk.text
        else:
            print(chunk, end="", flush=True)
            text += chunk
    print()  # Final newline
    return text


def print_stream(
    generator: Generator[str, None, None],
    markdown: bool = False,
    **kwargs
) -> str:
    """
    Convenience function to print streaming response.
    
    Args:
        generator: Streaming response generator
        markdown: Whether to render as markdown
        **kwargs: Additional arguments passed to display()
        
    Returns:
        str: The complete text content
        
    Examples:
        >>> response = llm.ask("Hello", stream=True)
        >>> text = print_stream(response)
        
        >>> # With markdown
        >>> text = print_stream(response, markdown=True)
    """
    return display(generator, markdown=markdown, **kwargs)


# Jupyter notebook support
def _is_jupyter() -> bool:
    """Check if running in Jupyter notebook."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


def display_jupyter(
    response: Union[Reply, ChainReply, Generator[str, None, None], Any],
    markdown: bool = True
) -> str:
    """
    Display response in Jupyter notebook.
    
    This function uses IPython.display for better rendering in notebooks.
    
    Args:
        response: LLM response object or generator
        markdown: Whether to render as markdown
        
    Returns:
        str: The complete text content
    """
    try:
        from IPython.display import display as ipython_display, Markdown as IPMarkdown, clear_output
        
        is_generator = hasattr(response, '__iter__') and not hasattr(response, 'text')
        
        if is_generator:
            # Handle streaming in Jupyter
            text = ""
            for chunk in response:
                text += chunk
                clear_output(wait=True)
                if markdown:
                    ipython_display(IPMarkdown(text))
                else:
                    print(text)
            return text
        else:
            # Handle completion response
            text = response.text if hasattr(response, 'text') else str(response)
            if markdown:
                ipython_display(IPMarkdown(text))
            else:
                print(text)
            return text
            
    except ImportError:
        # Fallback to regular display
        return display(response, markdown=markdown)