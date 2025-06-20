"""Prompt template implementation compatible with LangChain format."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class PromptTemplate:
    """A prompt template for generating formatted prompts with variable substitution.

    Compatible with LangChain prompt format for easy migration.
    """

    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
        template_format: str = "f-string",
        partial_variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a prompt template.

        Args:
            template: The template string with variables
            input_variables: List of variable names in the template
            template_format: Format type - "f-string" or "jinja2"
            partial_variables: Pre-filled variables
            metadata: Additional metadata (description, tags, etc.)
        """
        self.template = template
        self.template_format = template_format
        self.partial_variables = partial_variables or {}
        self.metadata = metadata or {}

        # Auto-detect input variables if not provided
        if input_variables is None:
            self.input_variables = self._extract_variables()
        else:
            self.input_variables = input_variables

    def _extract_variables(self) -> List[str]:
        """Extract variable names from the template string."""
        if self.template_format == "f-string":
            # Simple approach: find all {identifier} patterns
            pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
            matches = re.findall(pattern, self.template)
            # Remove duplicates and filter out partial variables
            unique_vars = list(dict.fromkeys(matches))
            return [m for m in unique_vars if m not in self.partial_variables]
        elif self.template_format == "jinja2":
            # Match {{ variable_name }} patterns
            pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
            matches = re.findall(pattern, self.template)
            # Remove duplicates and filter out partial variables
            unique_vars = list(dict.fromkeys(matches))
            return [m for m in unique_vars if m not in self.partial_variables]
        else:
            raise ValueError(f"Unsupported template format: {self.template_format}")

    def format(self, **kwargs) -> str:
        """Format the template with provided variables.

        Args:
            **kwargs: Variable values to substitute

        Returns:
            Formatted string with variables substituted
        """
        # Combine partial variables with provided kwargs
        all_vars = {**self.partial_variables, **kwargs}

        # Check for missing variables
        missing = set(self.input_variables) - set(all_vars.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        if self.template_format == "f-string":
            return self.template.format(**all_vars)
        elif self.template_format == "jinja2":
            try:
                from jinja2 import Template

                jinja_template = Template(self.template)
                return jinja_template.render(**all_vars)
            except ImportError:
                raise ImportError("jinja2 is required for jinja2 template format")
        else:
            raise ValueError(f"Unsupported template format: {self.template_format}")

    def partial(self, **kwargs) -> "PromptTemplate":
        """Create a new template with some variables pre-filled.

        Args:
            **kwargs: Variables to pre-fill

        Returns:
            New PromptTemplate with partial variables
        """
        new_partial = {**self.partial_variables, **kwargs}
        new_input_vars = [v for v in self.input_variables if v not in kwargs]

        return PromptTemplate(
            template=self.template,
            input_variables=new_input_vars,
            template_format=self.template_format,
            partial_variables=new_partial,
            metadata=self.metadata.copy(),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the prompt template to a JSON file.

        Args:
            path: File path to save to
        """
        path = Path(path)
        data = {
            "template": self.template,
            "input_variables": self.input_variables,
            "template_format": self.template_format,
            "partial_variables": self.partial_variables,
            "metadata": self.metadata,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PromptTemplate":
        """Load a prompt template from a JSON file.

        Args:
            path: File path to load from

        Returns:
            PromptTemplate instance
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            template=data.get("template"),
            input_variables=data.get("input_variables"),
            template_format=data.get("template_format", "f-string"),
            partial_variables=data.get("partial_variables"),
            metadata=data.get("metadata"),
        )

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "template": self.template,
            "input_variables": self.input_variables,
            "template_format": self.template_format,
            "partial_variables": self.partial_variables,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"PromptTemplate(input_variables={self.input_variables}, " f"template_format='{self.template_format}')"
