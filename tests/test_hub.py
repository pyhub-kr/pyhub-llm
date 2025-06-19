"""Tests for the hub functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from pyhub.llm import hub
from pyhub.llm.templates import PromptTemplate


class TestPromptTemplate:
    """Test PromptTemplate functionality."""
    
    def test_create_simple_prompt(self):
        """Test creating a simple prompt template."""
        prompt = PromptTemplate(
            template="Hello {name}!",
            input_variables=["name"]
        )
        assert prompt.input_variables == ["name"]
        assert prompt.template_format == "f-string"
        
    def test_auto_detect_variables(self):
        """Test auto-detection of input variables."""
        prompt = PromptTemplate(template="Hello {name}, you are {age} years old.")
        assert set(prompt.input_variables) == {"name", "age"}
        
    def test_format_prompt(self):
        """Test formatting a prompt."""
        prompt = PromptTemplate(
            template="Hello {name}!",
            input_variables=["name"]
        )
        result = prompt.format(name="Alice")
        assert result == "Hello Alice!"
        
    def test_missing_variable_error(self):
        """Test error when missing required variables."""
        prompt = PromptTemplate(
            template="Hello {name}!",
            input_variables=["name"]
        )
        with pytest.raises(ValueError, match="Missing required variables"):
            prompt.format()
            
    def test_partial_prompt(self):
        """Test creating partial prompts."""
        prompt = PromptTemplate(
            template="Hello {name}, you are {age} years old.",
            input_variables=["name", "age"]
        )
        partial = prompt.partial(name="Bob")
        assert partial.input_variables == ["age"]
        result = partial.format(age=30)
        assert result == "Hello Bob, you are 30 years old."
        
    def test_save_and_load(self):
        """Test saving and loading prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = PromptTemplate(
                template="Test {variable}",
                input_variables=["variable"],
                metadata={"test": True}
            )
            
            # Save
            path = Path(tmpdir) / "test_prompt.json"
            prompt.save(path)
            
            # Load
            loaded = PromptTemplate.load(path)
            assert loaded.template == prompt.template
            assert loaded.input_variables == prompt.input_variables
            assert loaded.metadata == prompt.metadata
            
    def test_jinja2_format(self):
        """Test Jinja2 template format."""
        prompt = PromptTemplate(
            template="Hello {{ name }}!",
            input_variables=["name"],
            template_format="jinja2"
        )
        result = prompt.format(name="Charlie")
        assert result == "Hello Charlie!"
        
    def test_unsupported_format(self):
        """Test unsupported template format."""
        prompt = PromptTemplate(
            template="Hello {{name}}!",
            input_variables=["name"],
            template_format="mustache"
        )
        with pytest.raises(ValueError, match="Unsupported template format"):
            prompt.format(name="David")


class TestPromptHub:
    """Test PromptHub functionality."""
    
    def test_pull_builtin_prompts(self):
        """Test pulling built-in prompts."""
        # Test RAG prompt
        rag_prompt = hub.pull("rlm/rag-prompt")
        assert isinstance(rag_prompt, PromptTemplate)
        assert set(rag_prompt.input_variables) == {"context", "question"}
        assert rag_prompt.metadata["author"] == "rlm"
        
        # Test ReAct prompt
        react_prompt = hub.pull("hwchase17/react")
        assert isinstance(react_prompt, PromptTemplate)
        assert set(react_prompt.input_variables) == {"tools", "tool_names", "input"}
        assert react_prompt.metadata["author"] == "hwchase17"
        
    def test_pull_nonexistent_prompt(self):
        """Test pulling a non-existent prompt."""
        with pytest.raises(ValueError, match="not found in hub"):
            hub.pull("nonexistent/prompt")
            
    def test_push_and_pull_custom_prompt(self):
        """Test pushing and pulling custom prompts."""
        # Create a custom prompt
        custom_prompt = PromptTemplate(
            template="Custom {test} prompt",
            input_variables=["test"],
            metadata={"custom": True}
        )
        
        # Push it
        hub.push("test/custom", custom_prompt)
        
        # Pull it back
        loaded = hub.pull("test/custom")
        assert loaded.template == custom_prompt.template
        assert loaded.input_variables == custom_prompt.input_variables
        assert loaded.metadata == custom_prompt.metadata
        
    def test_list_prompts(self):
        """Test listing available prompts."""
        prompts = hub.list_prompts()
        
        # Check built-in prompts are included
        assert "rlm/rag-prompt" in prompts
        assert "hwchase17/react" in prompts
        assert "hwchase17/openai-functions-agent" in prompts
        assert "hwchase17/structured-chat-agent" in prompts
        
    def test_custom_cache_directory(self):
        """Test using a custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create hub with custom cache dir
            custom_hub = hub.PromptHub(cache_dir=tmpdir)
            
            # Push a prompt
            prompt = PromptTemplate(
                template="Test {var}",
                input_variables=["var"]
            )
            custom_hub.push("test/prompt", prompt)
            
            # Check the file exists in custom location with new structure
            cache_file = Path(tmpdir) / "test" / "prompt.json"
            assert cache_file.exists()
            
            # Pull it back
            loaded = custom_hub.pull("test/prompt")
            assert loaded.template == prompt.template
    
    def test_prompt_name_validation(self):
        """Test prompt name validation for security."""
        # Test path traversal attempts
        with pytest.raises(ValueError, match="Invalid prompt name"):
            hub.pull("../etc/passwd")
        
        with pytest.raises(ValueError, match="Invalid prompt name"):
            hub.push("../../secret", PromptTemplate(template="test"))
        
        # Test invalid characters
        with pytest.raises(ValueError, match="Invalid prompt name format"):
            hub.pull("test@prompt")
        
        # Test too long name
        long_name = "a" * 101
        with pytest.raises(ValueError, match="Prompt name too long"):
            hub.pull(long_name)
    
    def test_escaped_braces_in_template(self):
        """Test that escaped braces are not detected as variables."""
        # Test with properly escaped braces in f-string format
        # {{ becomes { and }} becomes } in the output
        prompt = PromptTemplate(
            template="Show literal braces: {{ and }}. Variable: {value}",
        )
        assert prompt.input_variables == ["value"]
        
        formatted = prompt.format(value="test")
        assert formatted == "Show literal braces: { and }. Variable: test"
        
    def test_complex_json_template(self):
        """Test template with JSON-like structure."""
        # This shows {value} is a variable inside a JSON structure
        prompt = PromptTemplate(
            template='The JSON is: {{"key": "{value}", "number": {number}}}',
        )
        assert set(prompt.input_variables) == {"value", "number"}
        
        formatted = prompt.format(value="hello", number=42)
        assert formatted == 'The JSON is: {"key": "hello", "number": 42}'