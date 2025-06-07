"""Template engine for PyHub LLM."""

from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape


class TemplateEngine:
    """Jinja2-based template engine for prompts."""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = Path(template_dir) if template_dir else None
        
        # Create Jinja2 environment
        if self.template_dir:
            loader = FileSystemLoader(str(self.template_dir))
        else:
            loader = None
        
        self.env = Environment(
            loader=loader,
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def render(self, template_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Render a template file with the given context.
        
        Args:
            template_name: Name of the template file
            context: Template context variables
            
        Returns:
            Rendered template string
        """
        if not self.env.loader:
            raise ValueError("No template directory configured")
        
        template = self.env.get_template(template_name)
        return template.render(context or {})
    
    def render_string(self, template_string: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Render a template string with the given context.
        
        Args:
            template_string: Template string
            context: Template context variables
            
        Returns:
            Rendered template string
        """
        template = self.env.from_string(template_string)
        return template.render(context or {})
    
    def has_template(self, template_name: str) -> bool:
        """Check if a template file exists.
        
        Args:
            template_name: Name of the template file
            
        Returns:
            True if the template exists
        """
        if not self.env.loader:
            return False
        
        try:
            self.env.get_template(template_name)
            return True
        except:
            return False
    
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
    
    def add_test(self, name: str, func: callable) -> None:
        """Add a custom test to the template engine.
        
        Args:
            name: Test name
            func: Test function
        """
        self.env.tests[name] = func