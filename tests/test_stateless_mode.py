"""Test cases for stateless mode functionality."""

import pytest

from pyhub.llm import LLM
from pyhub.llm.mock import MockLLM
from pyhub.llm.types import Message


class TestStatelessMode:
    """Test stateless mode functionality."""

    def test_stateless_initialization(self):
        """Test stateless mode can be initialized."""
        llm = MockLLM(model="mock-model", stateless=True)
        assert llm.is_stateless is True
        assert len(llm.history) == 0

    def test_stateless_no_history_accumulation(self):
        """Test that stateless mode doesn't accumulate history."""
        llm = MockLLM(model="mock-model", stateless=True)

        # Make multiple requests
        llm.ask("First question")
        assert len(llm.history) == 0

        llm.ask("Second question")
        assert len(llm.history) == 0

        llm.ask("Third question")
        assert len(llm.history) == 0

    def test_stateless_vs_stateful_comparison(self):
        """Compare stateless and stateful behavior."""
        # Stateful (normal) mode
        stateful_llm = MockLLM(model="mock-model", stateless=False)
        stateful_llm.ask("Question 1")
        stateful_llm.ask("Question 2")
        assert len(stateful_llm.history) == 4  # 2 questions + 2 answers

        # Stateless mode
        stateless_llm = MockLLM(model="mock-model", stateless=True)
        stateless_llm.ask("Question 1")
        stateless_llm.ask("Question 2")
        assert len(stateless_llm.history) == 0

    def test_stateless_clear_noop(self):
        """Test that clear() is a no-op in stateless mode."""
        llm = MockLLM(model="mock-model", stateless=True)

        # Even if we somehow add to history
        llm.history.append(Message(role="user", content="test"))

        # Clear should not affect anything
        llm.clear()
        # History remains as is (though it shouldn't have been modified in the first place)
        assert len(llm.history) == 1

    def test_stateless_use_history_ignored(self):
        """Test that use_history is ignored in stateless mode."""
        llm = MockLLM(model="mock-model", stateless=True)

        # Even with use_history=True, no history should be used or saved
        reply = llm.ask("Test question", use_history=True)
        assert reply.text == "Mock response: Test question"
        assert len(llm.history) == 0

    def test_stateless_with_choices(self):
        """Test stateless mode with choices parameter."""
        llm = MockLLM(model="mock-model", stateless=True)

        # Multiple classification tasks
        reply1 = llm.ask("Classify: 'I want a refund'", choices=["refund", "inquiry", "complaint"])
        assert reply1.choice == "refund"
        assert len(llm.history) == 0

        reply2 = llm.ask("Classify: 'When will it arrive?'", choices=["refund", "shipping", "complaint"])
        assert reply2.choice == "refund"  # MockLLM always returns first choice
        assert len(llm.history) == 0

    def test_stateless_with_schema(self):
        """Test stateless mode with schema parameter."""
        from pydantic import BaseModel

        class Sentiment(BaseModel):
            sentiment: str
            confidence: float

        llm = MockLLM(model="mock-model", stateless=True)

        # Multiple extraction tasks
        _reply1 = llm.ask("Extract sentiment: 'This is great!'", schema=Sentiment)
        assert len(llm.history) == 0

        _reply2 = llm.ask("Extract sentiment: 'This is terrible!'", schema=Sentiment)
        assert len(llm.history) == 0

    def test_stateless_through_factory(self):
        """Test stateless mode through LLM.create factory."""
        # Use a real model name that will be recognized
        llm = LLM.create("gpt-4o-mini", stateless=True, api_key="test-key")

        # Check type and stateless property
        from pyhub.llm.openai import OpenAILLM

        assert isinstance(llm, OpenAILLM)
        assert llm.is_stateless is True

    @pytest.mark.asyncio
    async def test_stateless_async(self):
        """Test stateless mode with async methods."""
        llm = MockLLM(model="mock-model", stateless=True)

        await llm.ask_async("Async question 1")
        assert len(llm.history) == 0

        await llm.ask_async("Async question 2")
        assert len(llm.history) == 0

    def test_stateless_streaming(self):
        """Test stateless mode with streaming."""
        llm = MockLLM(model="mock-model", stateless=True)

        # Collect streaming response
        chunks = list(llm.ask("Streaming question", stream=True))
        assert len(chunks) > 0
        assert len(llm.history) == 0

    def test_stateless_property_readonly(self):
        """Test that is_stateless property is read-only."""
        llm = MockLLM(model="mock-model", stateless=True)

        # Try to change stateless property (should not be possible)
        with pytest.raises(AttributeError):
            llm.is_stateless = False

        # Internal stateless flag should remain unchanged
        assert llm.stateless is True
