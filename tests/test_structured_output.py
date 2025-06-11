"""Tests for structured output using Pydantic BaseModel"""

import pytest
from typing import Optional, List
from pydantic import BaseModel, Field

from pyhub.llm import LLM
from pyhub.llm.mock import MockLLM
from pyhub.llm.types import Reply, Usage


# Test schemas
class SimpleUser(BaseModel):
    name: str
    age: int
    email: str


class Address(BaseModel):
    street: str
    city: str
    country: str
    
    
class ComplexUser(BaseModel):
    name: str
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    addresses: List[Address]
    is_active: bool = True
    metadata: Optional[dict] = None


class TestStructuredOutput:
    """Test structured output functionality"""
    
    def test_reply_has_structured_data_attribute(self):
        """Test that Reply class has structured_data attribute"""
        reply = Reply(text="test")
        assert hasattr(reply, 'structured_data')
        assert reply.structured_data is None
        
    def test_reply_has_validation_errors_attribute(self):
        """Test that Reply class has validation_errors attribute"""
        reply = Reply(text="test")
        assert hasattr(reply, 'validation_errors')
        assert reply.validation_errors is None
        
    def test_reply_has_structured_data_property(self):
        """Test that Reply has has_structured_data property"""
        reply = Reply(text="test")
        assert hasattr(reply, 'has_structured_data')
        assert reply.has_structured_data is False
        
        # With structured data
        user = SimpleUser(name="John", age=30, email="john@example.com")
        reply_with_data = Reply(text='{"name": "John", "age": 30, "email": "john@example.com"}', structured_data=user)
        assert reply_with_data.has_structured_data is True
        
    def test_ask_accepts_schema_parameter(self):
        """Test that ask method accepts schema parameter"""
        llm = MockLLM()
        
        # Should not raise TypeError
        response = llm.ask("Create a user", schema=SimpleUser)
        assert isinstance(response, Reply)
        
    def test_ask_with_simple_schema(self):
        """Test ask with simple Pydantic schema"""
        # Create a mock that returns valid JSON
        llm = MockLLM(response='{"name": "John Doe", "age": 30, "email": "john@example.com"}')
        
        response = llm.ask(
            "Create a user named John Doe, 30 years old, email john@example.com",
            schema=SimpleUser
        )
        
        assert response.has_structured_data
        assert isinstance(response.structured_data, SimpleUser)
        assert response.structured_data.name == "John Doe"
        assert response.structured_data.age == 30
        assert response.structured_data.email == "john@example.com"
        
    def test_ask_with_complex_schema(self):
        """Test ask with complex nested schema"""
        json_response = '''{
            "name": "Jane Smith",
            "age": 25,
            "email": "jane@example.com",
            "addresses": [
                {"street": "123 Main St", "city": "Seoul", "country": "Korea"},
                {"street": "456 Oak Ave", "city": "Busan", "country": "Korea"}
            ],
            "is_active": true,
            "metadata": {"role": "admin", "department": "IT"}
        }'''
        
        llm = MockLLM(response=json_response)
        response = llm.ask("Create a user with multiple addresses", schema=ComplexUser)
        
        assert response.has_structured_data
        assert isinstance(response.structured_data, ComplexUser)
        assert response.structured_data.name == "Jane Smith"
        assert len(response.structured_data.addresses) == 2
        assert response.structured_data.addresses[0].city == "Seoul"
        assert response.structured_data.metadata["role"] == "admin"
        
    def test_ask_with_invalid_json_response(self):
        """Test handling of invalid JSON response"""
        llm = MockLLM(response="This is not JSON")
        
        response = llm.ask("Create a user", schema=SimpleUser)
        
        assert not response.has_structured_data
        assert response.structured_data is None
        assert response.validation_errors is not None
        assert len(response.validation_errors) > 0
        assert "JSON" in response.validation_errors[0] or "parse" in response.validation_errors[0]
        
    def test_ask_with_schema_validation_error(self):
        """Test handling of schema validation errors"""
        # Missing required field
        llm = MockLLM(response='{"name": "John", "age": 30}')  # missing email
        
        response = llm.ask("Create a user", schema=SimpleUser)
        
        assert not response.has_structured_data
        assert response.structured_data is None
        assert response.validation_errors is not None
        assert any("email" in error for error in response.validation_errors)
        
    def test_ask_with_schema_and_choices_raises_error(self):
        """Test that using both schema and choices raises an error"""
        llm = MockLLM()
        
        with pytest.raises(ValueError, match="Cannot use both 'schema' and 'choices'"):
            llm.ask(
                "Create something",
                schema=SimpleUser,
                choices=["option1", "option2"]
            )
            
    def test_ask_async_with_schema(self):
        """Test async ask with schema"""
        import asyncio
        
        async def test():
            llm = MockLLM(response='{"name": "Async User", "age": 25, "email": "async@example.com"}')
            response = await llm.ask_async("Create a user", schema=SimpleUser)
            
            assert response.has_structured_data
            assert response.structured_data.name == "Async User"
            
        asyncio.run(test())
        
    def test_streaming_with_schema(self):
        """Test streaming response with schema (structured data in final chunk)"""
        # For streaming, structured data should be in the last chunk
        llm = MockLLM(
            response='{"name": "Stream User", "age": 28, "email": "stream@example.com"}',
            streaming_response=True
        )
        
        chunks = list(llm.ask("Create a user", schema=SimpleUser, stream=True))
        
        # Last chunk should have structured data
        last_chunk = chunks[-1]
        assert last_chunk.has_structured_data
        assert isinstance(last_chunk.structured_data, SimpleUser)
        assert last_chunk.structured_data.name == "Stream User"
        
    def test_schema_with_field_validation(self):
        """Test schema with Pydantic field validation"""
        # Invalid age (negative)
        llm = MockLLM(response='{"name": "John", "age": -5, "email": "john@example.com"}')
        
        response = llm.ask("Create a user", schema=ComplexUser)
        
        assert not response.has_structured_data
        assert response.validation_errors is not None
        assert any("age" in error for error in response.validation_errors)
        
    def test_schema_with_optional_fields(self):
        """Test schema with optional fields"""
        # Without optional fields
        json_response = '''{
            "name": "Simple User",
            "age": 30,
            "email": "simple@example.com",
            "addresses": []
        }'''
        
        llm = MockLLM(response=json_response)
        response = llm.ask("Create a simple user", schema=ComplexUser)
        
        assert response.has_structured_data
        assert response.structured_data.is_active is True  # default value
        assert response.structured_data.metadata is None  # optional
        
    @pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
    def test_provider_specific_implementation(self, provider):
        """Test that each provider can handle structured output"""
        # This is a placeholder for provider-specific tests
        # Will be implemented when we add provider support
        pass