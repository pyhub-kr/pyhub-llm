"""Mock LLM provider for testing."""

import asyncio
from pathlib import Path
from typing import IO, Any, AsyncGenerator, Generator, List, Optional, Union

from .base import BaseLLM
from .types import Embed, EmbedList, LLMChatModelType, Message, Reply, Usage


class MockLLM(BaseLLM):
    """Mock LLM implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0
        self.last_question = None
        self.last_messages = None
        self.mock_response = "Mock response"
        self.mock_usage = Usage(input=10, output=20)
        self.mock_embedding = [0.1, 0.2, 0.3, 0.4]

    def ask(
        self,
        input: Union[str, dict[str, Any]],
        files: Optional[list[Union[str, Path, IO]]] = None,
        model: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        *,
        choices: Optional[list[str]] = None,
        choices_optional: bool = False,
        stream: bool = False,
        use_history: bool = True,
        raise_errors: bool = False,
        enable_cache: bool = False,
        tools: Optional[list] = None,
        tool_choice: str = "auto",
        max_tool_calls: int = 5,
        **kwargs,
    ) -> Union[Reply, Generator[Reply, None, None]]:
        """Ask a question to the mock LLM."""
        self.call_count += 1

        # Convert input to string if it's a dict
        if isinstance(input, dict):
            question = str(input)
        else:
            question = input

        self.last_question = question

        # Add to history if requested
        if use_history:
            self.history.append(Message(role="user", content=question))

        # Handle choices
        if choices:
            # Return first choice
            response_text = choices[0]
            reply = Reply(text=response_text, usage=self.mock_usage, choice=choices[0], choice_index=0, confidence=0.95)
        else:
            response_text = f"{self.mock_response}: {question}"
            reply = Reply(text=response_text, usage=self.mock_usage)

        # Add assistant response to history if requested
        if use_history:
            self.history.append(Message(role="assistant", content=response_text))

        # Handle streaming
        if stream:

            def _stream():
                for word in response_text.split():
                    yield word + " "

            return _stream()

        return reply

    async def ask_async(
        self,
        input: Union[str, dict[str, Any]],
        files: Optional[list[Union[str, Path, IO]]] = None,
        model: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        *,
        choices: Optional[list[str]] = None,
        choices_optional: bool = False,
        stream: bool = False,
        raise_errors: bool = False,
        use_history: bool = True,
        enable_cache: bool = False,
        tools: Optional[list] = None,
        tool_choice: str = "auto",
        max_tool_calls: int = 5,
    ) -> Union[Reply, AsyncGenerator[Reply, None]]:
        """Ask a question to the mock LLM asynchronously."""
        # Simulate async delay
        await asyncio.sleep(0.01)

        # Convert input to string if it's a dict
        if isinstance(input, dict):
            question = str(input)
        else:
            question = input

        # Use sync implementation
        result = self.ask(
            input=input,
            files=files,
            model=model,
            context=context,
            choices=choices,
            choices_optional=choices_optional,
            stream=stream,
            use_history=use_history,
            enable_cache=enable_cache,
            tools=tools,
            tool_choice=tool_choice,
            max_tool_calls=max_tool_calls,
        )

        # Convert generator to async generator if streaming
        if stream and isinstance(result, Generator):

            async def _async_stream():
                for chunk in result:
                    yield chunk
                    await asyncio.sleep(0.001)

            return _async_stream()

        return result

    def messages(
        self,
        messages: List[Message],
        save_history: bool = True,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Union[Reply, Generator[str, None, None]]:
        """Send messages to the mock LLM."""
        self.call_count += 1
        self.last_messages = messages

        # Get last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break

        response_text = f"{self.mock_response}: {len(messages)} messages"
        if last_user_msg:
            response_text = f"{self.mock_response}: {last_user_msg}"

        reply = Reply(text=response_text, usage=self.mock_usage)

        # Add to history if requested
        if save_history:
            self.history.extend(messages)
            self.history.append(Message(role="assistant", content=response_text))

        # Handle streaming
        if stream:

            def _stream():
                for word in response_text.split():
                    yield word + " "

            return _stream()

        return reply

    async def messages_async(
        self,
        messages: List[Message],
        save_history: bool = True,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Union[Reply, AsyncGenerator[str, None]]:
        """Send messages to the mock LLM asynchronously."""
        # Simulate async delay
        await asyncio.sleep(0.01)

        # Use sync implementation
        result = self.messages(
            messages=messages,
            save_history=save_history,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            **kwargs,
        )

        # Convert generator to async generator if streaming
        if stream and isinstance(result, Generator):

            async def _async_stream():
                for chunk in result:
                    yield chunk
                    await asyncio.sleep(0.001)

            return _async_stream()

        return result

    def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[Embed, EmbedList]:
        """Generate mock embeddings."""
        self.call_count += 1

        if isinstance(text, str):
            # Single text
            return Embed(array=self.mock_embedding, usage=Usage(input=5, output=0))
        else:
            # Multiple texts
            embeddings = [Embed(array=self.mock_embedding) for _ in text]
            return EmbedList(arrays=embeddings, usage=Usage(input=5 * len(text), output=0))

    async def embed_async(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[Embed, EmbedList]:
        """Generate mock embeddings asynchronously."""
        # Simulate async delay
        await asyncio.sleep(0.01)

        return self.embed(text, model, **kwargs)

    # Test helpers
    def set_mock_response(self, response: str) -> None:
        """Set the mock response text."""
        self.mock_response = response

    def set_mock_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Set the mock usage."""
        self.mock_usage = Usage(input=input_tokens, output=output_tokens)

    def set_mock_embedding(self, embedding: List[float]) -> None:
        """Set the mock embedding."""
        self.mock_embedding = embedding

    def reset(self) -> None:
        """Reset the mock state."""
        self.call_count = 0
        self.last_question = None
        self.last_messages = None
        self.clear()

    # Implement abstract methods
    def _make_request_params(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: LLMChatModelType,
    ) -> dict:
        """Generate request parameters for the mock LLM."""
        return {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _make_ask(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: LLMChatModelType,
    ) -> Reply:
        """Generate a response using the mock LLM."""
        response_text = f"{self.mock_response}: {human_message.content}"
        return Reply(text=response_text, usage=self.mock_usage)

    async def _make_ask_async(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: LLMChatModelType,
    ) -> Reply:
        """Generate a response asynchronously using the mock LLM."""
        await asyncio.sleep(0.01)
        return self._make_ask(input_context, human_message, messages, model)

    def _make_ask_stream(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: LLMChatModelType,
    ) -> Generator[Reply, None, None]:
        """Generate a streaming response using the mock LLM."""
        response_text = f"{self.mock_response}: {human_message.content}"
        for word in response_text.split():
            yield Reply(text=word + " ", usage=Usage(input=0, output=0))

    async def _make_ask_stream_async(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: LLMChatModelType,
    ) -> AsyncGenerator[Reply, None]:
        """Generate a streaming response asynchronously using the mock LLM."""
        response_text = f"{self.mock_response}: {human_message.content}"
        for word in response_text.split():
            yield Reply(text=word + " ", usage=Usage(input=0, output=0))
            await asyncio.sleep(0.001)
