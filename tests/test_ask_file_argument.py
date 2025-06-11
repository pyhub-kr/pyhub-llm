"""Tests for ask command --file argument"""
import tempfile
from pathlib import Path
from unittest.mock import patch
import typer.testing
from pyhub.llm.commands import app


def test_ask_with_single_file():
    """Test ask command with single file"""
    runner = typer.testing.CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def hello():\n    return 'world'")
        temp_file = Path(f.name)
    
    try:
        with patch('pyhub.llm.LLM.create') as mock_create:
            # Mock the LLM response
            mock_llm = mock_create.return_value
            mock_response = type('Response', (), {'text': 'Test response', 'usage': None})
            mock_llm.ask.return_value = mock_response
            
            result = runner.invoke(
                app, 
                ["ask", "분석해주세요", "--file", str(temp_file), "--no-stream"]
            )
            
            assert result.exit_code == 0
            assert "Test response" in result.output
            
            # Verify that the file content was passed as context
            call_args = mock_llm.ask.call_args
            assert call_args is not None
    finally:
        temp_file.unlink()


def test_ask_with_multiple_files():
    """Test ask command with multiple files"""
    runner = typer.testing.CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f2:
        f1.write("def add(a, b):\n    return a + b")
        f2.write("def multiply(a, b):\n    return a * b")
        temp_file1 = Path(f1.name)
        temp_file2 = Path(f2.name)
    
    try:
        with patch('pyhub.llm.LLM.create') as mock_create:
            # Mock the LLM response
            mock_llm = mock_create.return_value
            mock_response = type('Response', (), {'text': 'Test response', 'usage': None})
            mock_llm.ask.return_value = mock_response
            
            result = runner.invoke(
                app, 
                ["ask", "분석해주세요", "--file", str(temp_file1), "--file", str(temp_file2), "--no-stream"]
            )
            
            assert result.exit_code == 0
            assert "Test response" in result.output
    finally:
        temp_file1.unlink()
        temp_file2.unlink()


def test_ask_with_nonexistent_file():
    """Test ask command with nonexistent file"""
    runner = typer.testing.CliRunner()
    
    result = runner.invoke(
        app, 
        ["ask", "분석해주세요", "--file", "nonexistent_file.py"]
    )
    
    assert result.exit_code == 1
    assert "오류: 파일을 찾을 수 없습니다" in result.output


def test_ask_with_file_and_context():
    """Test ask command with both file and context options"""
    runner = typer.testing.CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def hello():\n    return 'world'")
        temp_file = Path(f.name)
    
    try:
        with patch('pyhub.llm.LLM.create') as mock_create:
            # Mock the LLM response
            mock_llm = mock_create.return_value
            mock_response = type('Response', (), {'text': 'Test response', 'usage': None})
            mock_llm.ask.return_value = mock_response
            
            result = runner.invoke(
                app, 
                ["ask", "분석해주세요", "--file", str(temp_file), "--context", "추가 컨텍스트", "--no-stream"]
            )
            
            assert result.exit_code == 0
            assert "Test response" in result.output
    finally:
        temp_file.unlink()