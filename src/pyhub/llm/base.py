"""Base LLM class for PyHub LLM."""

import abc
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from .cache import BaseCache
from .exceptions import LLMError
from .templates import TemplateEngine
from .types import (
    ChainReply,
    Embed,
    EmbedList,
    LLMChatModelType,
    LLMEmbeddingModelType,
    LLMResponse,
    Message,
    Reply,
    StreamResponse,
)

logger = logging.getLogger(__name__)


class TemplateDict(dict):
    """템플릿 변수 중 존재하지 않는 키는 원래 형태({key})로 유지하는 딕셔너리"""

    def __missing__(self, key):
        return "{" + key + "}"


@dataclass
class DescribeImageRequest:
    image: Union[str, Path]
    image_path: str
    system_prompt: str
    user_prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    prompt_context: Optional[Dict[str, Any]] = None


class BaseLLM(abc.ABC):
    """Abstract base class for LLM implementations."""
    
    EMBEDDING_DIMENSIONS = {}

    def __init__(
        self,
        model: LLMChatModelType = "gpt-4o-mini",
        embedding_model: LLMEmbeddingModelType = "text-embedding-3-small",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        prompt: Optional[str] = None,
        output_key: str = "text",
        initial_messages: Optional[List[Message]] = None,
        api_key: Optional[str] = None,
        tools: Optional[List] = None,
        cache: Optional[BaseCache] = None,
        template_engine: Optional[TemplateEngine] = None,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.output_key = output_key
        self.history = initial_messages or []
        self.api_key = api_key
        self.cache = cache
        self.template_engine = template_engine or TemplateEngine()

        # 기본 도구 설정
        self.default_tools = []
        if tools:
            # tools 모듈을 동적 import (순환 import 방지)
            from .tools import ToolAdapter

            self.default_tools = ToolAdapter.adapt_tools(tools)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, embedding_model={self.embedding_model}, temperature={self.temperature}, max_tokens={self.max_tokens})"

    def __len__(self) -> int:
        return len(self.history)

    def __or__(self, next_llm: Union["BaseLLM", "SequentialChain"]) -> "SequentialChain":
        if isinstance(next_llm, BaseLLM):
            return SequentialChain(self, next_llm)
        elif isinstance(next_llm, SequentialChain):
            next_llm.insert_first(self)
            return next_llm
        else:
            raise TypeError("next_llm must be an instance of BaseLLM or SequentialChain")

    def __ror__(self, prev_llm: Union["BaseLLM", "SequentialChain"]) -> "SequentialChain":
        if isinstance(prev_llm, BaseLLM):
            return SequentialChain(prev_llm, self)
        elif isinstance(prev_llm, SequentialChain):
            prev_llm.append(self)
            return prev_llm
        else:
            raise TypeError("prev_llm must be an instance of BaseLLM or SequentialChain")

    def _render_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Render a prompt template with context."""
        if context is None:
            context = {}
        
        # Try template engine first
        if self.template_engine.has_template(prompt):
            return self.template_engine.render(prompt, context)
        
        # Otherwise treat as string template
        return self.template_engine.render_string(prompt, context)

    # 1. 문자열 질의

    @abc.abstractmethod
    def ask(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        save_history: bool = True,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        files: Optional[List[Union[str, Path]]] = None,
        choices: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[Reply, Generator[str, None, None]]:
        """Ask a question to the LLM."""
        pass

    def stream(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        save_history: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        files: Optional[List[Union[str, Path]]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream responses from the LLM."""
        return self.ask(
            question=question,
            system_prompt=system_prompt,
            save_history=save_history,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            files=files,
            **kwargs,
        )

    @abc.abstractmethod
    async def ask_async(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        save_history: bool = True,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        files: Optional[List[Union[str, Path]]] = None,
        choices: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[Reply, AsyncGenerator[str, None]]:
        """Ask a question to the LLM asynchronously."""
        pass

    async def stream_async(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        save_history: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        files: Optional[List[Union[str, Path]]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream responses from the LLM asynchronously."""
        result = await self.ask_async(
            question=question,
            system_prompt=system_prompt,
            save_history=save_history,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            files=files,
            **kwargs,
        )
        
        if isinstance(result, AsyncGenerator):
            return result
        else:
            # If not a generator, convert to one
            async def _gen():
                yield str(result)
            return _gen()

    # 2. 메시지 기반 질의

    @abc.abstractmethod
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
        """Send messages to the LLM."""
        pass

    @abc.abstractmethod
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
        """Send messages to the LLM asynchronously."""
        pass

    # 3. 임베딩

    @abc.abstractmethod
    def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[Embed, EmbedList]:
        """Generate embeddings for text."""
        pass

    @abc.abstractmethod
    async def embed_async(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[Embed, EmbedList]:
        """Generate embeddings for text asynchronously."""
        pass

    # 4. 이미지 설명

    def describe_image(
        self,
        image: Union[str, Path],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Reply:
        """Describe an image."""
        # Default implementation using ask with files
        if system_prompt is None:
            system_prompt = "You are an image analysis expert."
        
        if user_prompt is None:
            user_prompt = "Please describe this image in detail."
        
        return self.ask(
            question=user_prompt,
            system_prompt=system_prompt,
            files=[image],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def describe_image_async(
        self,
        image: Union[str, Path],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Reply:
        """Describe an image asynchronously."""
        # Default implementation using ask_async with files
        if system_prompt is None:
            system_prompt = "You are an image analysis expert."
        
        if user_prompt is None:
            user_prompt = "Please describe this image in detail."
        
        result = await self.ask_async(
            question=user_prompt,
            system_prompt=system_prompt,
            files=[image],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        # Handle generator case
        if isinstance(result, AsyncGenerator):
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
            return Reply(text="".join(chunks))
        
        return result

    # 5. Function calling / Tool use

    def ask_with_tools(
        self,
        question: str,
        tools: Optional[List] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Reply:
        """Ask with function calling support."""
        # Default implementation - subclasses should override
        raise NotImplementedError(f"{self.__class__.__name__} does not support function calling")

    async def ask_with_tools_async(
        self,
        question: str,
        tools: Optional[List] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Reply:
        """Ask with function calling support asynchronously."""
        # Default implementation - subclasses should override
        raise NotImplementedError(f"{self.__class__.__name__} does not support function calling")

    # 6. History management

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    def add_message(self, message: Message) -> None:
        """Add a message to history."""
        self.history.append(message)

    def get_history(self) -> List[Message]:
        """Get conversation history."""
        return self.history.copy()

    # 7. Cache key generation

    def _generate_cache_key(self, prefix: str, **params) -> str:
        """Generate a cache key for the given parameters."""
        import hashlib
        import json
        
        # Sort parameters for consistent keys
        sorted_params = json.dumps(params, sort_keys=True)
        param_hash = hashlib.sha256(sorted_params.encode()).hexdigest()[:16]
        
        return f"{self.__class__.__name__}:{prefix}:{self.model}:{param_hash}"


class SequentialChain:
    """Sequential chain of LLMs."""

    def __init__(self, *llms: BaseLLM):
        self.llms = list(llms)

    def __or__(self, next_llm: Union[BaseLLM, "SequentialChain"]) -> "SequentialChain":
        if isinstance(next_llm, BaseLLM):
            self.llms.append(next_llm)
        elif isinstance(next_llm, SequentialChain):
            self.llms.extend(next_llm.llms)
        else:
            raise TypeError("next_llm must be an instance of BaseLLM or SequentialChain")
        return self

    def __ror__(self, prev_llm: Union[BaseLLM, "SequentialChain"]) -> "SequentialChain":
        if isinstance(prev_llm, BaseLLM):
            self.llms.insert(0, prev_llm)
        elif isinstance(prev_llm, SequentialChain):
            self.llms = prev_llm.llms + self.llms
        else:
            raise TypeError("prev_llm must be an instance of BaseLLM or SequentialChain")
        return self

    def insert_first(self, llm: BaseLLM) -> None:
        """Insert LLM at the beginning of the chain."""
        self.llms.insert(0, llm)

    def append(self, llm: BaseLLM) -> None:
        """Append LLM to the end of the chain."""
        self.llms.append(llm)

    def ask(self, initial_input: str, **kwargs) -> ChainReply:
        """Execute the chain with an initial input."""
        chain_reply = ChainReply()
        current_input = initial_input

        for i, llm in enumerate(self.llms):
            if hasattr(llm, "output_key"):
                output_key = llm.output_key
            else:
                output_key = f"llm_{i}"

            reply = llm.ask(current_input, **kwargs)
            chain_reply.reply_list.append(reply)
            chain_reply.values[output_key] = reply.text

            # Use the output as input for the next LLM
            current_input = reply.text

        return chain_reply

    async def ask_async(self, initial_input: str, **kwargs) -> ChainReply:
        """Execute the chain asynchronously with an initial input."""
        chain_reply = ChainReply()
        current_input = initial_input

        for i, llm in enumerate(self.llms):
            if hasattr(llm, "output_key"):
                output_key = llm.output_key
            else:
                output_key = f"llm_{i}"

            result = await llm.ask_async(current_input, **kwargs)
            
            # Handle generator case
            if isinstance(result, AsyncGenerator):
                chunks = []
                async for chunk in result:
                    chunks.append(chunk)
                reply = Reply(text="".join(chunks))
            else:
                reply = result
                
            chain_reply.reply_list.append(reply)
            chain_reply.values[output_key] = reply.text

            # Use the output as input for the next LLM
            current_input = reply.text

        return chain_reply