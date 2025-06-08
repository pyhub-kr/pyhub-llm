"""Template engine using Jinja2."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from jinja2 import DictLoader, Environment, FileSystemLoader, Template


class TemplateEngine:
    """Template engine wrapper for Jinja2."""

    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        """Initialize template engine.

        Args:
            template_dir: Directory containing templates
        """
        self.template_dir = Path(template_dir) if template_dir else None

        if self.template_dir and self.template_dir.exists():
            loader = FileSystemLoader(str(self.template_dir))
        else:
            loader = DictLoader({})

        self.env = Environment(loader=loader, autoescape=True)

    def render_string(self, template_string: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Render a template string with context.

        Args:
            template_string: Template string to render
            context: Template context variables

        Returns:
            Rendered template string
        """
        context = context or {}
        template = self.env.from_string(template_string)
        return template.render(**context)

    def render_template(self, template_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Render a template file with context.

        Args:
            template_name: Name of template file
            context: Template context variables

        Returns:
            Rendered template string
        """
        context = context or {}
        template = self.env.get_template(template_name)
        return template.render(**context)

    def add_filter(self, name: str, func: callable) -> None:
        """Add a custom filter to the template engine.

        Args:
            name: Filter name
            func: Filter function
        """
        self.env.filters[name] = func

    def add_global(self, name: str, value: Any) -> None:
        """Add a global variable to the template engine.

        Args:
            name: Variable name
            value: Variable value
        """
        self.env.globals[name] = value

    def get_template(self, name: str) -> Template:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            Jinja2 Template object
        """
        return self.env.get_template(name)

    def from_string(self, source: str) -> Template:
        """Create a template from a string.

        Args:
            source: Template source string

        Returns:
            Jinja2 Template object
        """
        return self.env.from_string(source)
