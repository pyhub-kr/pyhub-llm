from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pyhub.llm.templates.engine import TemplateEngine


class TestTemplateEngine:
    """Test TemplateEngine functionality."""

    def test_template_engine_initialization(self, tmp_path):
        """Test TemplateEngine initialization."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        engine = TemplateEngine(str(template_dir))
        assert engine.template_dir == template_dir

    def test_render_string_template(self):
        """Test rendering a string template."""
        engine = TemplateEngine()

        template = "Hello {{ name }}!"
        result = engine.render_string(template, {"name": "World"})
        assert result == "Hello World!"

    def test_render_string_with_complex_context(self):
        """Test rendering with complex context."""
        engine = TemplateEngine()

        template = """
        User: {{ user.name }}
        Items:
        {% for item in items %}
        - {{ item }}
        {% endfor %}
        """

        context = {"user": {"name": "Alice"}, "items": ["apple", "banana", "orange"]}

        result = engine.render_string(template, context)
        assert "User: Alice" in result
        assert "- apple" in result
        assert "- banana" in result
        assert "- orange" in result

    def test_template_with_filters(self):
        """Test templates with filters."""
        engine = TemplateEngine()

        template = "{{ name|upper }} - {{ price|round(2) }}"
        result = engine.render_string(template, {"name": "product", "price": 12.345})
        assert result == "PRODUCT - 12.35"

    def test_template_with_conditionals(self):
        """Test templates with conditionals."""
        engine = TemplateEngine()

        template = """
        {% if user %}
        Hello {{ user }}!
        {% else %}
        Hello Guest!
        {% endif %}
        """

        result = engine.render_string(template, {"user": "Alice"})
        assert "Hello Alice!" in result

        result = engine.render_string(template, {})
        assert "Hello Guest!" in result

    def test_template_with_macros(self):
        """Test templates with macros."""
        engine = TemplateEngine()

        template = """
        {% macro greeting(name) %}
        Hello {{ name }}!
        {% endmacro %}
        
        {{ greeting("World") }}
        """

        result = engine.render_string(template)
        assert "Hello World!" in result

    def test_template_inheritance(self):
        """Test template inheritance."""
        # This test requires file-based templates
        tmp_path = Path("/tmp/test_templates")
        tmp_path.mkdir(exist_ok=True)

        # Create base template
        base_template = tmp_path / "base.html"
        base_template.write_text(
            """
        <html>
        <head><title>{% block title %}Default{% endblock %}</title></head>
        <body>{% block content %}{% endblock %}</body>
        </html>
        """
        )

        # Create child template
        child_template = tmp_path / "child.html"
        child_template.write_text(
            """
        {% extends "base.html" %}
        {% block title %}Child Page{% endblock %}
        {% block content %}Hello from child!{% endblock %}
        """
        )

        engine = TemplateEngine(str(tmp_path))
        result = engine.render_template("child.html")

        assert "<title>Child Page</title>" in result
        assert "Hello from child!" in result

        # Cleanup
        base_template.unlink()
        child_template.unlink()
        tmp_path.rmdir()

    def test_template_with_includes(self):
        """Test template includes."""
        tmp_path = Path("/tmp/test_includes")
        tmp_path.mkdir(exist_ok=True)

        # Create included template
        header_template = tmp_path / "header.html"
        header_template.write_text("<h1>{{ title }}</h1>")

        # Create main template
        main_template = tmp_path / "main.html"
        main_template.write_text(
            """
        {% include "header.html" %}
        <p>Content here</p>
        """
        )

        engine = TemplateEngine(str(tmp_path))
        result = engine.render_template("main.html", {"title": "My Page"})

        assert "<h1>My Page</h1>" in result
        assert "<p>Content here</p>" in result

        # Cleanup
        header_template.unlink()
        main_template.unlink()
        tmp_path.rmdir()

    def test_template_with_custom_filters(self):
        """Test adding custom filters."""
        engine = TemplateEngine()

        # Add custom filter
        def reverse_string(s):
            return s[::-1]

        engine.add_filter("reverse", reverse_string)

        template = "{{ text|reverse }}"
        result = engine.render_string(template, {"text": "hello"})
        assert result == "olleh"

    def test_template_globals(self):
        """Test adding global variables."""
        engine = TemplateEngine()

        # Add global variable
        engine.add_global("app_name", "MyApp")

        template = "Welcome to {{ app_name }}!"
        result = engine.render_string(template)
        assert result == "Welcome to MyApp!"

    def test_render_file_template(self, tmp_path):
        """Test rendering file templates."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        # Create template file
        template_file = template_dir / "test.html"
        template_file.write_text("Hello {{ name }}!")

        engine = TemplateEngine(str(template_dir))
        result = engine.render_template("test.html", {"name": "World"})
        assert result == "Hello World!"

    def test_autoescape(self):
        """Test autoescaping of HTML."""
        engine = TemplateEngine()

        template = "{{ content }}"
        result = engine.render_string(template, {"content": "<script>alert('xss')</script>"})

        # Should escape HTML
        assert "&lt;script&gt;" in result
        assert "&lt;/script&gt;" in result

    def test_template_not_found(self, tmp_path):
        """Test handling of missing templates."""
        engine = TemplateEngine(str(tmp_path))

        with pytest.raises(Exception):  # Jinja2 raises TemplateNotFound
            engine.render_template("nonexistent.html")

    def test_empty_context(self):
        """Test rendering with empty context."""
        engine = TemplateEngine()

        template = "Static content"
        result = engine.render_string(template)
        assert result == "Static content"

    def test_complex_nested_data(self):
        """Test rendering with complex nested data."""
        engine = TemplateEngine()

        template = """
        {% for user in users %}
        Name: {{ user.name }}
        Tags: {% for tag in user.tags %}{{ tag }}{% if not loop.last %}, {% endif %}{% endfor %}
        {% endfor %}
        """

        context = {"users": [{"name": "Alice", "tags": ["admin", "user"]}, {"name": "Bob", "tags": ["user", "guest"]}]}

        result = engine.render_string(template, context)
        assert "Name: Alice" in result
        assert "Tags: admin, user" in result
        assert "Name: Bob" in result
        assert "Tags: user, guest" in result

    def test_from_string_method(self):
        """Test creating template from string."""
        engine = TemplateEngine()

        template = engine.from_string("Hello {{ name }}!")
        result = template.render(name="World")
        assert result == "Hello World!"

    def test_get_template_method(self, tmp_path):
        """Test getting template object."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        # Create template file
        template_file = template_dir / "test.html"
        template_file.write_text("Hello {{ name }}!")

        engine = TemplateEngine(str(template_dir))
        template = engine.get_template("test.html")
        result = template.render(name="World")
        assert result == "Hello World!"

    def test_template_with_loops(self):
        """Test templates with loops."""
        engine = TemplateEngine()

        template = """
        {% for i in range(3) %}
        Item {{ loop.index }}: {{ items[i] }}
        {% endfor %}
        """

        context = {"items": ["apple", "banana", "orange"]}
        result = engine.render_string(template, context)

        assert "Item 1: apple" in result
        assert "Item 2: banana" in result
        assert "Item 3: orange" in result

    def test_template_with_comments(self):
        """Test templates with comments."""
        engine = TemplateEngine()

        template = """
        {# This is a comment #}
        Hello {{ name }}!
        <!-- This HTML comment will show -->
        """

        result = engine.render_string(template, {"name": "World"})
        assert "This is a comment" not in result  # Jinja2 comment
        assert "<!-- This HTML comment will show -->" in result  # HTML comment

    @patch("pyhub.llm.templates.engine.Environment")
    def test_environment_configuration(self, mock_env_class):
        """Test that environment is properly configured."""
        mock_env = Mock()
        mock_env_class.return_value = mock_env

        _engine = TemplateEngine()

        # Verify Environment was created with autoescape
        mock_env_class.assert_called_once()
        args, kwargs = mock_env_class.call_args
        assert kwargs.get("autoescape") is True


class TestTemplateIntegration:
    """Test template integration with LLM."""

    def test_template_with_llm_prompts(self):
        """Test using templates for LLM prompts."""
        engine = TemplateEngine()

        prompt_template = """
        You are a helpful assistant.
        
        User: {{ question }}
        
        Please provide a {{ style }} response.
        """

        context = {"question": "What is Python?", "style": "concise"}

        result = engine.render_string(prompt_template, context)
        assert "User: What is Python?" in result
        assert "Please provide a concise response." in result

    def test_template_with_system_prompts(self):
        """Test templates for system prompts."""
        engine = TemplateEngine()

        system_template = """
        You are {{ role }}.
        Your expertise includes: {% for skill in skills %}{{ skill }}{% if not loop.last %}, {% endif %}{% endfor %}.
        {% if constraints %}
        Constraints:
        {% for constraint in constraints %}
        - {{ constraint }}
        {% endfor %}
        {% endif %}
        """

        context = {
            "role": "a Python expert",
            "skills": ["Django", "FastAPI", "Data Science"],
            "constraints": ["Keep responses under 100 words", "Use examples"],
        }

        result = engine.render_string(system_template, context)
        assert "You are a Python expert." in result
        assert "Django, FastAPI, Data Science" in result
        assert "- Keep responses under 100 words" in result
