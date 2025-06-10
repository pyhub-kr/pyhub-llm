import logging
import re
from typing import Any, AsyncGenerator, Generator, Optional, Union, cast

import pydantic
from pydantic import ValidationError

from pyhub.llm.base import BaseLLM
from pyhub.llm.cache.utils import generate_cache_key
from pyhub.llm.settings import llm_settings
from pyhub.llm.types import (
    Embed,
    EmbedList,
    Message,
    OllamaChatModelType,
    OllamaEmbeddingModelType,
    Reply,
    Usage,
)
from pyhub.llm.utils.files import IOType, encode_files
from pyhub.llm.utils.templates import Template

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """
    Ollama API를 사용하여 LLM 기능을 제공하는 클래스입니다.
    """

    SUPPORTED_FILE_TYPES = [IOType.IMAGE]  # Ollama는 기본적으로 이미지만 지원

    EMBEDDING_DIMENSIONS = {
        "nomic-embed-text": 768,
        "avr/sfr-embedding-mistral": 4096,
    }

    def __init__(
        self,
        model: OllamaChatModelType = "mistral",
        embedding_model: OllamaEmbeddingModelType = "nomic-embed-text",
        temperature: float = 0.2,
        # max_tokens: int = 1000,
        system_prompt: Optional[Union[str, Template]] = None,
        prompt: Optional[Union[str, Template]] = None,
        output_key: str = "text",
        initial_messages: Optional[list[Message]] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        **kwargs,
    ):
        """
        Ollama LLM 클래스 초기화

        Args:
            model: 사용할 Ollama 모델 이름
            embedding_model: 임베딩에 사용할 모델 이름
            temperature: 생성 다양성 조절 (0.0-1.0)
            system_prompt: 시스템 프롬프트
            prompt: 사용자 프롬프트 템플릿
            output_key: 출력 결과를 저장할 키
            initial_messages: 초기 대화 메시지 목록
            base_url: Ollama API 기본 URL
            timeout: API 요청 타임아웃 (초)
        """

        # Lazy import ollama
        try:
            from ollama import AsyncClient, ChatResponse
            from ollama import Client as SyncClient
            from ollama import EmbedResponse, ListResponse

            self._AsyncClient = AsyncClient
            self._SyncClient = SyncClient
            self._ChatResponse = ChatResponse
            self._EmbedResponse = EmbedResponse
            self._ListResponse = ListResponse
        except ImportError:
            raise ImportError("ollama package not installed. " "Install with: pip install pyhub-llm[ollama]")

        if ":" not in model:
            model += ":latest"

        if ":" not in embedding_model:
            embedding_model += ":latest"

        super().__init__(
            model=model,
            embedding_model=embedding_model,
            temperature=temperature,
            # max_tokens=max_tokens,
            system_prompt=system_prompt,
            prompt=prompt,
            output_key=output_key,
            initial_messages=initial_messages,
            **kwargs,
        )
        self.base_url = base_url or llm_settings.ollama_base_url
        self.timeout = timeout

    def check(self) -> list[dict]:
        errors = super().check()

        def add_error(msg: str, hint: str = None):
            errors.append({"msg": msg, "hint": hint, "obj": self})

        client = self._SyncClient(host=self.base_url)
        try:
            response: self._ListResponse = client.list()
        except ConnectionError:
            add_error(f"Unable to connect to Ollama server at {self.base_url}.")
        else:
            model_name_set = {model.model for model in response.models}

            if self.model not in model_name_set:
                add_error(
                    f"Ollama model '{self.model}' not found on server at {self.base_url}",
                    hint="Please check if the model is installed or use 'ollama pull {self.model}' to download it.",
                )

            if self.embedding_model not in model_name_set:
                add_error(
                    f"Ollama embedding model '{self.embedding_model}' not found on server at {self.base_url}",
                    hint="Please check if the embedding model is installed or use 'ollama pull {self.embedding_model}' to download it.",
                )

        return errors

    def _make_request_params(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: OllamaChatModelType,
    ) -> dict:
        """Ollama API 요청에 필요한 파라미터를 준비하고 시스템 프롬프트를 처리합니다."""
        message_history = [dict(message) for message in messages]
        system_prompt = self.get_system_prompt(input_context)

        if system_prompt:
            # history에는 system prompt는 누적되지 않고, 매 요청 시마다 적용합니다.
            system_message = {"role": "system", "content": system_prompt}
            message_history.insert(0, system_message)

        # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
        # Ollama는 PDF를 직접 지원하지 않으므로 이미지로 변환
        image_urls = encode_files(
            human_message.files,
            allowed_types=[IOType.IMAGE, IOType.PDF],  # PDF도 받지만 이미지로 변환됨
            convert_mode="base64",
            pdf_to_image_for_unsupported=True,  # PDF를 이미지로 변환
        )

        if image_urls:
            base64_url_pattern = r"^data:([^;]+);base64,(.+)"

            b64_str_list: list[str] = []
            for image_url in image_urls:
                base64_url_match = re.match(base64_url_pattern, image_url)
                if base64_url_match:
                    # mimetype = base64_url_match.group(1)
                    b64_str = base64_url_match.group(2)
                    b64_str_list.append(b64_str)

            message_history.append(
                {
                    "role": human_message.role,
                    "content": human_message.content,
                    "images": b64_str_list,
                }
            )
        else:
            message_history.append(
                {
                    "role": human_message.role,
                    "content": human_message.content,
                }
            )

        logger.debug("Ollama model: %s, temperature: %s", model, self.temperature)

        return {
            "model": model,
            "messages": message_history,
            "options": {
                "temperature": self.temperature,
                #  "max_tokens": self.max_tokens,  # ollama 에서는 미지원
            },
        }

    def _make_ask(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: OllamaChatModelType,
    ) -> Reply:
        """
        Ollama API를 사용하여 동기적으로 응답을 생성합니다.
        """

        sync_client = self._SyncClient(host=self.base_url)
        request_params = self._make_request_params(
            input_context=input_context,
            human_message=human_message,
            messages=messages,
            model=model,
        )

        # Cache handling
        cache_key = None
        cached_value = None
        is_cached = False

        if self.cache:
            cache_key = generate_cache_key("ollama", **request_params)
            cached_value = self.cache.get(cache_key)

        response: Optional[self._ChatResponse] = None
        if cached_value is not None:
            try:
                response = self._ChatResponse.model_validate_json(cached_value)
                is_cached = True
            except ValidationError:
                logger.error("Invalid cached value : %s", cached_value)

        if response is None:
            logger.debug("request to ollama")
            response = sync_client.chat(**request_params)

            # Store in cache if enabled
            if self.cache and cache_key:
                self.cache.set(cache_key, response.model_dump_json())

        assert response is not None

        # 캐시된 응답인 경우 usage를 0으로 설정하여 비용 중복 계산 방지
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage_input = 0 if is_cached else getattr(response.usage, "prompt_tokens", 0)
            usage_output = 0 if is_cached else getattr(response.usage, "completion_tokens", 0)
            usage = Usage(input=usage_input, output=usage_output)

        return Reply(text=response.message.content, usage=usage)

    async def _make_ask_async(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: OllamaChatModelType,
    ) -> Reply:
        """
        Ollama API를 사용하여 비동기적으로 응답을 생성합니다.
        """

        async_client = self._AsyncClient(host=self.base_url)
        request_params = self._make_request_params(
            input_context=input_context,
            human_message=human_message,
            messages=messages,
            model=model,
        )

        # Cache handling
        cache_key = None
        cached_value = None
        is_cached = False

        if self.cache:
            cache_key = generate_cache_key("ollama", **request_params)
            cached_value = self.cache.get(cache_key)  # Assuming cache is synchronous

        response: Optional[self._ChatResponse] = None
        if cached_value is not None:
            try:
                response = self._ChatResponse.model_validate_json(cached_value)
                is_cached = True
            except ValidationError:
                logger.error("Invalid cached value : %s", cached_value)

        if response is None:
            logger.debug("request to ollama")
            response: self._ChatResponse = await async_client.chat(**request_params)

            # Store in cache if enabled
            if self.cache and cache_key:
                self.cache.set(cache_key, response.model_dump_json())  # Assuming cache is synchronous

        assert response is not None

        # 캐시된 응답인 경우 usage를 0으로 설정하여 비용 중복 계산 방지
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage_input = 0 if is_cached else getattr(response.usage, "prompt_tokens", 0)
            usage_output = 0 if is_cached else getattr(response.usage, "completion_tokens", 0)
            usage = Usage(input=usage_input, output=usage_output)

        return Reply(text=response.message.content, usage=usage)

    def _make_ask_stream(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: OllamaChatModelType,
    ) -> Generator[Reply, None, None]:
        """
        Ollama API를 사용하여 동기적으로 스트리밍 응답을 생성합니다.
        """

        sync_client = self._SyncClient(host=self.base_url)
        request_params = self._make_request_params(
            input_context=input_context,
            human_message=human_message,
            messages=messages,
            model=model,
        )
        request_params["stream"] = True

        # TODO: Streaming cache support not implemented yet
        # For now, streaming responses are not cached
        if self.cache:
            # TODO: Implement streaming cache support
            pass

        logger.debug("request to ollama")

        response_stream = sync_client.chat(**request_params)

        reply_list: list[Reply] = []
        for chunk in response_stream:
            # 스트림 응답에서는 usage 정보가 제한적이므로 기본값 사용
            usage = None
            if hasattr(chunk, "usage") and chunk.usage:
                usage_input = getattr(chunk.usage, "prompt_tokens", 0)
                usage_output = getattr(chunk.usage, "completion_tokens", 0)
                usage = Usage(input=usage_input, output=usage_output)

            reply = Reply(text=chunk.message.content or "", usage=usage)
            reply_list.append(reply)
            yield reply

        # TODO: Streaming cache not implemented yet

    async def _make_ask_stream_async(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: OllamaChatModelType,
    ) -> AsyncGenerator[Reply, None]:
        """
        Ollama API를 사용하여 비동기적으로 스트리밍 응답을 생성합니다.
        """
        async_client = self._AsyncClient(host=self.base_url)
        request_params = self._make_request_params(
            input_context=input_context,
            human_message=human_message,
            messages=messages,
            model=model,
        )
        request_params["stream"] = True

        # TODO: Streaming cache support not implemented yet
        # For now, streaming responses are not cached
        if self.cache:
            # TODO: Implement streaming cache support
            pass

        logger.debug("request to ollama")

        response = await async_client.chat(**request_params)

        reply_list: list[Reply] = []
        async for chunk in response:
            # 스트림 응답에서는 usage 정보가 제한적이므로 기본값 사용
            usage = None
            if hasattr(chunk, "usage") and chunk.usage:
                usage_input = getattr(chunk.usage, "prompt_tokens", 0)
                usage_output = getattr(chunk.usage, "completion_tokens", 0)
                usage = Usage(input=usage_input, output=usage_output)

            reply = Reply(text=chunk.message.content or "", usage=usage)
            reply_list.append(reply)
            yield reply

        # TODO: Streaming cache not implemented yet

    def embed(
        self,
        input: Union[str, list[str]],
        model: Optional[OllamaEmbeddingModelType] = None,
    ) -> Union[Embed, EmbedList]:
        """
        Ollama API를 사용하여 텍스트를 임베딩합니다.
        """
        embedding_model = model or self.embedding_model

        sync_client = self._SyncClient(host=self.base_url)
        request_params = dict(
            model=cast(str, embedding_model),
            input=input,
        )

        # Cache handling
        cache_key = None
        cached_value = None
        is_cached = False

        if self.cache:
            cache_key = generate_cache_key("ollama", **request_params)
            cached_value = self.cache.get(cache_key)

        response: Optional[self._EmbedResponse] = None
        if cached_value is not None:
            try:
                response = self._EmbedResponse.model_validate_json(cached_value)
                is_cached = True
            except pydantic.ValidationError as e:
                logger.error("Invalid cached value : %s", e)

        if response is None:
            logger.debug("request to ollama")
            response = sync_client.embed(**request_params)

            # Store in cache if enabled
            if self.cache and cache_key:
                self.cache.set(cache_key, response.model_dump_json())

        # 캐시된 응답인 경우 usage를 0으로 설정하여 비용 중복 계산 방지
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage_input = 0 if is_cached else getattr(response.usage, "prompt_tokens", 0)
            usage = Usage(input=usage_input, output=0)

        if isinstance(input, str):
            return Embed(list(response.embeddings[0]), usage=usage)
        return EmbedList([Embed(list(e)) for e in response.embeddings], usage=usage)

    async def embed_async(
        self,
        input: Union[str, list[str]],
        model: Optional[str] = None,
    ) -> Union[Embed, EmbedList]:
        """
        Ollama API를 사용하여 비동기적으로 텍스트를 임베딩합니다.
        """

        embedding_model = model or self.embedding_model

        async_client = self._AsyncClient(host=self.base_url)
        request_params = dict(
            model=cast(str, embedding_model),
            input=input,
        )

        # Cache handling
        cache_key = None
        cached_value = None
        is_cached = False

        if self.cache:
            cache_key = generate_cache_key("ollama", **request_params)
            cached_value = self.cache.get(cache_key)  # Assuming cache is synchronous

        response: Optional[self._EmbedResponse] = None
        if cached_value is not None:
            try:
                response = self._EmbedResponse.model_validate_json(cached_value)
                is_cached = True
            except pydantic.ValidationError as e:
                logger.error("Invalid cached value : %s", e)

        if response is None:
            logger.debug("request to ollama")
            response = await async_client.embed(**request_params)

            # Store in cache if enabled
            if self.cache and cache_key:
                self.cache.set(cache_key, response.model_dump_json())  # Assuming cache is synchronous

        # 캐시된 응답인 경우 usage를 0으로 설정하여 비용 중복 계산 방지
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage_input = 0 if is_cached else getattr(response.usage, "prompt_tokens", 0)
            usage = Usage(input=usage_input, output=0)

        if isinstance(input, str):
            return Embed(list(response.embeddings[0]), usage=usage)
        return EmbedList([Embed(list(e)) for e in response.embeddings], usage=usage)


__all__ = ["OllamaLLM"]
