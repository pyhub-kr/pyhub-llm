"""Simple template utilities to replace Django templates"""

from typing import Any, Dict


class Template:
    """Simple template wrapper that mimics Django Template interface"""

    def __init__(self, template_string: str):
        self.template_string = template_string
        self._has_django_syntax = "{{" in template_string or "{%" in template_string

    def render(self, context: Dict[str, Any]) -> str:
        """Render the template with given context"""
        if self._has_django_syntax:
            # For now, just do simple variable replacement
            result = self.template_string
            for key, value in context.items():
                result = result.replace(f"{{{{{key}}}}}", str(value))
                result = result.replace(f"{{{{ {key} }}}}", str(value))
            return result
        else:
            # Use Python string formatting
            return self.template_string.format_map(context)


class Context(dict):
    """Simple context wrapper that mimics Django Context"""

    pass


class TemplateDoesNotExist(Exception):
    """Exception raised when template is not found"""

    pass


def get_template(template_name: str) -> Template:
    """Load a template from file (simplified version)"""
    # For now, just raise exception
    raise TemplateDoesNotExist(f"Template '{template_name}' not found")


def async_to_sync(async_func):
    """Convert async function to sync (simplified version)"""
    import asyncio

    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()

    return wrapper
