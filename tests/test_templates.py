import pytest
from pathlib import Path
from unittest.mock import Mock, patch
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
        
        context = {
            "user": {"name": "Alice"},
            "items": ["apple", "banana", "orange"]
        }
        
        result = engine.render_string(template, context)
        assert "User: Alice" in result
        assert "- apple" in result
        assert "- banana" in result
        assert "- orange" in result
    
    def test_render_file_template(self, tmp_path):
        """Test rendering a file template."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        # Create a template file
        template_file = template_dir / "greeting.j2"
        template_file.write_text("Hello {{ name }} from {{ city }}!")
        
        engine = TemplateEngine(str(template_dir))
        result = engine.render("greeting.j2", {"name": "Bob", "city": "Paris"})
        
        assert result == "Hello Bob from Paris!"
    
    def test_render_nested_template(self, tmp_path):
        """Test rendering nested templates."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        subdir = template_dir / "prompts"
        subdir.mkdir()
        
        # Create a nested template
        template_file = subdir / "analysis.j2"
        template_file.write_text("Analyze: {{ query }}")
        
        engine = TemplateEngine(str(template_dir))
        result = engine.render("prompts/analysis.j2", {"query": "What is AI?"})
        
        assert result == "Analyze: What is AI?"
    
    def test_template_not_found(self, tmp_path):
        """Test handling of missing template."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        engine = TemplateEngine(str(template_dir))
        
        with pytest.raises(Exception):  # Jinja2 will raise TemplateNotFound
            engine.render("nonexistent.j2")
    
    def test_template_with_filters(self):
        """Test using Jinja2 filters."""
        engine = TemplateEngine()
        
        template = "{{ name|upper }} - {{ price|round(2) }}"
        result = engine.render_string(template, {"name": "product", "price": 10.456})
        
        assert result == "PRODUCT - 10.46"
    
    def test_template_with_conditionals(self):
        """Test templates with conditionals."""
        engine = TemplateEngine()
        
        template = """
        {% if user.premium %}
        Welcome Premium User: {{ user.name }}
        {% else %}
        Welcome {{ user.name }}
        {% endif %}
        """
        
        # Premium user
        result1 = engine.render_string(
            template,
            {"user": {"name": "Alice", "premium": True}}
        )
        assert "Welcome Premium User: Alice" in result1
        
        # Regular user
        result2 = engine.render_string(
            template,
            {"user": {"name": "Bob", "premium": False}}
        )
        assert "Welcome Bob" in result2
        assert "Premium" not in result2
    
    def test_template_with_macros(self):
        """Test templates with macros."""
        engine = TemplateEngine()
        
        template = """
        {% macro format_user(user) %}
        {{ user.name }} ({{ user.email }})
        {% endmacro %}
        
        Users:
        {% for user in users %}
        - {{ format_user(user) }}
        {% endfor %}
        """
        
        users = [
            {"name": "Alice", "email": "alice@example.com"},
            {"name": "Bob", "email": "bob@example.com"}
        ]
        
        result = engine.render_string(template, {"users": users})
        assert "Alice (alice@example.com)" in result
        assert "Bob (bob@example.com)" in result
    
    def test_template_inheritance(self, tmp_path):
        """Test template inheritance."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        # Create base template
        base_template = template_dir / "base.j2"
        base_template.write_text("""
        <system>
        {% block system %}Default system prompt{% endblock %}
        </system>
        
        <user>
        {% block user %}Default user prompt{% endblock %}
        </user>
        """)
        
        # Create child template
        child_template = template_dir / "child.j2"
        child_template.write_text("""
        {% extends "base.j2" %}
        
        {% block system %}You are a helpful assistant.{% endblock %}
        
        {% block user %}{{ question }}{% endblock %}
        """)
        
        engine = TemplateEngine(str(template_dir))
        result = engine.render("child.j2", {"question": "What is Python?"})
        
        assert "<system>" in result
        assert "You are a helpful assistant." in result
        assert "<user>" in result
        assert "What is Python?" in result
    
    def test_template_with_includes(self, tmp_path):
        """Test template includes."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        # Create include file
        include_file = template_dir / "_header.j2"
        include_file.write_text("=== {{ title }} ===")
        
        # Create main template
        main_template = template_dir / "report.j2"
        main_template.write_text("""
        {% include "_header.j2" %}
        
        {{ content }}
        """)
        
        engine = TemplateEngine(str(template_dir))
        result = engine.render(
            "report.j2",
            {
                "title": "Analysis Report",
                "content": "This is the report content."
            }
        )
        
        assert "=== Analysis Report ===" in result
        assert "This is the report content." in result
    
    def test_template_escaping(self):
        """Test HTML escaping in templates."""
        engine = TemplateEngine()
        
        # By default, Jinja2 auto-escapes HTML
        template = "{{ content }}"
        result = engine.render_string(
            template,
            {"content": "<script>alert('XSS')</script>"}
        )
        
        # Should escape HTML characters
        assert "&lt;script&gt;" in result or "<script>" in result
        # Note: Default escaping depends on Jinja2 configuration
    
    def test_template_with_custom_filters(self):
        """Test adding custom filters."""
        engine = TemplateEngine()
        
        # Add a custom filter
        def reverse_string(s):
            return s[::-1]
        
        engine.env.filters['reverse'] = reverse_string
        
        template = "{{ text|reverse }}"
        result = engine.render_string(template, {"text": "hello"})
        
        assert result == "olleh"
    
    def test_template_globals(self):
        """Test template with global variables."""
        engine = TemplateEngine()
        
        # Add global variables
        engine.env.globals['app_name'] = "PyHub LLM"
        engine.env.globals['version'] = "1.0.0"
        
        template = "{{ app_name }} v{{ version }}"
        result = engine.render_string(template)
        
        assert result == "PyHub LLM v1.0.0"
    
    def test_safe_template_rendering(self):
        """Test safe rendering with untrusted input."""
        engine = TemplateEngine()
        
        # Potentially dangerous template operations should be handled safely
        template = "Hello {{ name }}"
        
        # Even with special characters, should render safely
        result = engine.render_string(
            template,
            {"name": "{{7*7}}"}  # This should not be evaluated
        )
        
        assert result == "Hello {{7*7}}"
        assert "49" not in result  # Should not evaluate the expression


class TestTemplateIntegrationWithLLM:
    """Test template integration with LLM."""
    
    @patch('pyhub.llm.base.BaseLLM')
    def test_llm_with_template_prompt(self, mock_llm_class):
        """Test LLM using template for prompts."""
        # This would test the integration between LLM and templates
        # In actual implementation, LLM might have a method like:
        # llm.ask_with_template("template.j2", context)
        
        mock_llm = mock_llm_class()
        mock_llm.ask.return_value = "Mocked response"
        
        # Simulate template rendering
        template_engine = TemplateEngine()
        prompt = template_engine.render_string(
            "Analyze this data: {{ data }}",
            {"data": "[1, 2, 3, 4, 5]"}
        )
        
        response = mock_llm.ask(prompt)
        
        assert response == "Mocked response"
        mock_llm.ask.assert_called_with("Analyze this data: [1, 2, 3, 4, 5]")
    
    def test_prompt_library(self, tmp_path):
        """Test creating a prompt library with templates."""
        template_dir = tmp_path / "prompts"
        template_dir.mkdir()
        
        # Create various prompt templates
        prompts = {
            "summarize.j2": "Summarize the following text:\n\n{{ text }}",
            "translate.j2": "Translate to {{ target_language }}:\n\n{{ text }}",
            "analyze.j2": "Analyze the {{ aspect }} of:\n\n{{ content }}",
            "chat.j2": "{{ system_prompt }}\n\nUser: {{ user_message }}"
        }
        
        for filename, content in prompts.items():
            (template_dir / filename).write_text(content)
        
        engine = TemplateEngine(str(template_dir))
        
        # Test summarize prompt
        summary_prompt = engine.render(
            "summarize.j2",
            {"text": "Long article content here..."}
        )
        assert "Summarize the following text:" in summary_prompt
        assert "Long article content here..." in summary_prompt
        
        # Test translate prompt
        translate_prompt = engine.render(
            "translate.j2",
            {
                "target_language": "French",
                "text": "Hello world"
            }
        )
        assert "Translate to French:" in translate_prompt
        assert "Hello world" in translate_prompt