import logging
import re
from pathlib import Path
from typing import IO, Any, AsyncGenerator, Generator, Optional, Union, cast

import pydantic

from pyhub.llm.base import BaseLLM
from pyhub.llm.settings import llm_settings
from pyhub.llm.types import (
    Embed,
    EmbedList,
    GoogleChatModelType,
    GoogleEmbeddingModelType,
    Message,
    Reply,
    Usage,
)
from pyhub.llm.utils.files import IOType, encode_files
from pyhub.llm.utils.templates import Template

logger = logging.getLogger(__name__)


class GoogleLLM(BaseLLM):
    SUPPORTED_FILE_TYPES = [IOType.IMAGE, IOType.PDF]  # Google Gemini도 PDF 직접 지원
    EMBEDDING_DIMENSIONS = {
        "text-embedding-004": 768,
    }

    def __init__(
        self,
        model: GoogleChatModelType = "gemini-2.0-flash",
        embedding_model: GoogleEmbeddingModelType = "text-embedding-004",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        system_prompt: Optional[Union[str, Template]] = None,
        prompt: Optional[Union[str, Template]] = None,
        output_key: str = "text",
        initial_messages: Optional[list[Message]] = None,
        api_key: Optional[str] = None,
        tools: Optional[list] = None,
        **kwargs,
    ):
        # Lazy import google-genai
        try:
            import google.generativeai as genai
            from google.generativeai.types import (
                GenerateContentResponse,
            )

            self._genai = genai
            self._GenerateContentResponse = GenerateContentResponse
            # Create placeholder types for missing imports
            self._EmbedContentResponse = Any
            self._GenerateContentConfig = dict
        except ImportError:
            raise ImportError("google-genai package not installed. " "Install with: pip install pyhub-llm[google]")

        super().__init__(
            model=model,
            embedding_model=embedding_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            prompt=prompt,
            output_key=output_key,
            initial_messages=initial_messages,
            api_key=api_key or llm_settings.google_api_key,
            tools=tools,
            **kwargs,
        )

    def check(self) -> list[dict]:
        errors = super().check()

        if not self.api_key:
            errors.append(
                {
                    "msg": "Google API key is not set or is invalid.",
                    "hint": "Please check your Google API key.",
                    "obj": self,
                }
            )

        return errors

    def _make_request_params(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: GoogleChatModelType,
    ) -> dict:
        contents: list[dict] = [
            {
                "role": "user" if message.role == "user" else "model",
                "parts": [{"text": message.content}],
            }
            for message in messages
        ]

        # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/document-understanding
        image_urls = encode_files(
            human_message.files,
            allowed_types=self.SUPPORTED_FILE_TYPES,
            convert_mode="base64",
        )

        image_parts: list[dict] = []
        if image_urls:
            base64_url_pattern = r"^data:([^;]+);base64,(.+)"

            for image_url in image_urls:
                base64_url_match = re.match(base64_url_pattern, image_url)
                if base64_url_match:
                    mimetype = base64_url_match.group(1)
                    b64_str = base64_url_match.group(2)
                    image_part = {"inline_data": {"mime_type": mimetype, "data": b64_str}}  # Keep as base64 string
                    image_parts.append(image_part)
                else:
                    raise ValueError(
                        f"Invalid image data: {image_url}. Google Gemini API only supports base64 encoded images."
                    )

        contents.append(
            {
                "role": "user" if human_message.role == "user" else "model",
                "parts": [
                    *image_parts,
                    {"text": human_message.content},
                ],
            }
        )

        system_prompt: Optional[str] = self.get_system_prompt(input_context)

        # GenerationConfig only includes generation parameters
        config = self._GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return dict(
            model=model,
            contents=contents,
            config=config,
            system_instruction=system_prompt,  # Pass system instruction separately
        )

    def _make_ask(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: GoogleChatModelType,
    ) -> Reply:
        # Configure API key
        self._genai.configure(api_key=self.api_key)

        request_params = self._make_request_params(input_context, human_message, messages, model)

        response: Optional[self._GenerateContentResponse] = None
        is_cached = False
        cache_key = None

        # Check cache if enabled
        if self.cache:
            from pyhub.llm.cache.utils import generate_cache_key

            cache_key = generate_cache_key("google", **request_params)
            cached_value = self.cache.get(cache_key)

            if cached_value is not None:
                try:
                    # TODO: Implement proper cache deserialization for GenerateContentResponse
                    pass  # response = self._GenerateContentResponse.model_validate_json(cached_value)
                    # is_cached = True
                except pydantic.ValidationError as e:
                    logger.error("Invalid cached value : %s", e)

        if response is None:
            logger.debug("request to google genai")
            # Create GenerativeModel and generate content
            contents = request_params.pop("contents")
            config = request_params.pop("config")
            system_instruction = request_params.pop("system_instruction", None)

            # Create model with system instruction if provided
            generative_model = self._genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
            response = generative_model.generate_content(contents=contents, generation_config=config)

            # Store in cache if enabled
            if self.cache and cache_key:
                # TODO: Implement proper caching for GenerateContentResponse
                pass  # self.cache.set(cache_key, response.model_dump_json())

        assert response is not None

        # 캐시된 응답인 경우 usage를 0으로 설정
        usage_input = 0 if is_cached else (response.usage_metadata.prompt_token_count or 0)
        usage_output = 0 if is_cached else (response.usage_metadata.candidates_token_count or 0)

        return Reply(
            text=response.text,
            usage=Usage(
                input=usage_input,
                output=usage_output,
            ),
        )

    async def _make_ask_async(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: GoogleChatModelType,
    ) -> Reply:
        # Configure API key
        self._genai.configure(api_key=self.api_key)

        request_params = self._make_request_params(input_context, human_message, messages, model)

        response: Optional[self._GenerateContentResponse] = None
        is_cached = False
        cache_key = None

        # Check cache if enabled (using synchronous cache methods in async context)
        if self.cache:
            from pyhub.llm.cache.utils import generate_cache_key

            cache_key = generate_cache_key("google", **request_params)
            cached_value = self.cache.get(cache_key)

            if cached_value is not None:
                try:
                    # TODO: Implement proper cache deserialization for GenerateContentResponse
                    pass  # response = self._GenerateContentResponse.model_validate_json(cached_value)
                    # is_cached = True
                except pydantic.ValidationError as e:
                    logger.error("Invalid cached value : %s", e)

        if response is None:
            logger.debug("request to google genai")
            # Create GenerativeModel and generate content
            contents = request_params.pop("contents")
            config = request_params.pop("config")
            system_instruction = request_params.pop("system_instruction", None)

            # Create model with system instruction if provided
            generative_model = self._genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
            response = await generative_model.generate_content_async(contents=contents, generation_config=config)

            # Store in cache if enabled (using synchronous cache methods in async context)
            if self.cache and cache_key:
                # TODO: Implement proper caching for GenerateContentResponse
                pass  # self.cache.set(cache_key, response.model_dump_json())

        assert response is not None

        # 캐시된 응답인 경우 usage를 0으로 설정
        usage_input = 0 if is_cached else (response.usage_metadata.prompt_token_count or 0)
        usage_output = 0 if is_cached else (response.usage_metadata.candidates_token_count or 0)

        return Reply(
            text=response.text,
            usage=Usage(
                input=usage_input,
                output=usage_output,
            ),
        )

    def _make_ask_stream(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: GoogleChatModelType,
    ) -> Generator[Reply, None, None]:
        # TODO: Streaming cache is not implemented in the new cache injection pattern
        # Configure API key
        self._genai.configure(api_key=self.api_key)
        request_params = self._make_request_params(input_context, human_message, messages, model)

        # Check cache if enabled
        cache_key = None
        cached_value = None
        if self.cache:
            from pyhub.llm.cache.utils import generate_cache_key

            cache_key = generate_cache_key("google", stream=True, **request_params)
            cached_value = self.cache.get(cache_key)

        if cached_value is not None:
            reply_list = cast(list[Reply], cached_value)
            for reply in reply_list:
                if reply.usage is not None:
                    # 캐시된 응답인 경우 usage를 0으로 설정
                    reply.usage = Usage(input=0, output=0)
                yield reply

        else:
            # Create GenerativeModel and generate content stream
            contents = request_params.pop("contents")
            config = request_params.pop("config")
            system_instruction = request_params.pop("system_instruction", None)

            # Create model with system instruction if provided
            generative_model = self._genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
            response = generative_model.generate_content(contents=contents, generation_config=config, stream=True)

            input_tokens = 0
            output_tokens = 0

            reply_list: list[Reply] = []
            for chunk in response:
                reply = Reply(text=chunk.text)
                reply_list.append(reply)
                yield reply
                input_tokens += chunk.usage_metadata.prompt_token_count or 0
                output_tokens += chunk.usage_metadata.candidates_token_count or 0

            if input_tokens > 0 or output_tokens > 0:
                usage = Usage(input=input_tokens, output=output_tokens)
                reply = Reply(text="", usage=usage)
                reply_list.append(reply)
                yield reply

            # Store in cache if enabled
            if self.cache and cache_key:
                self.cache.set(cache_key, reply_list)

    async def _make_ask_stream_async(
        self,
        input_context: dict[str, Any],
        human_message: Message,
        messages: list[Message],
        model: GoogleChatModelType,
    ) -> AsyncGenerator[Reply, None]:
        # TODO: Streaming cache is not implemented in the new cache injection pattern
        # Configure API key
        self._genai.configure(api_key=self.api_key)
        request_params = self._make_request_params(input_context, human_message, messages, model)

        # Check cache if enabled (using synchronous cache methods in async context)
        cache_key = None
        cached_value = None
        if self.cache:
            from pyhub.llm.cache.utils import generate_cache_key

            cache_key = generate_cache_key("google", stream=True, **request_params)
            cached_value = self.cache.get(cache_key)

        if cached_value is not None:
            reply_list = cast(list[Reply], cached_value)
            for reply in reply_list:
                if reply.usage is not None:
                    # 캐시된 응답인 경우 usage를 0으로 설정
                    reply.usage = Usage(input=0, output=0)
                yield reply

        else:
            logger.debug("request to google genai")

            # Create GenerativeModel and generate content stream
            contents = request_params.pop("contents")
            config = request_params.pop("config")
            system_instruction = request_params.pop("system_instruction", None)

            # Create model with system instruction if provided
            generative_model = self._genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
            response = await generative_model.generate_content_async(
                contents=contents, generation_config=config, stream=True
            )

            input_tokens = 0
            output_tokens = 0

            reply_list: list[Reply] = []
            async for chunk in response:
                reply = Reply(text=chunk.text)
                reply_list.append(reply)
                yield reply
                input_tokens += chunk.usage_metadata.prompt_token_count or 0
                output_tokens += chunk.usage_metadata.candidates_token_count or 0

            if input_tokens > 0 or output_tokens > 0:
                usage = Usage(input=input_tokens, output=output_tokens)
                reply = Reply(text="", usage=usage)
                reply_list.append(reply)
                yield reply

            # Store in cache if enabled (using synchronous cache methods in async context)
            if self.cache and cache_key:
                self.cache.set(cache_key, reply_list)

    def ask(
        self,
        input: Union[str, dict[str, Any]],
        files: Optional[list[Union[str, Path, IO]]] = None,
        model: Optional[GoogleChatModelType] = None,
        context: Optional[dict[str, Any]] = None,
        *,
        choices: Optional[list[str]] = None,
        choices_optional: bool = False,
        stream: bool = False,
        use_history: bool = True,
        raise_errors: bool = False,
        tools: Optional[list] = None,
        tool_choice: str = "auto",
        max_tool_calls: int = 5,
    ) -> Union[Reply, Generator[Reply, None, None]]:
        return super().ask(
            input=input,
            files=files,
            model=model,
            context=context,
            choices=choices,
            choices_optional=choices_optional,
            stream=stream,
            use_history=use_history,
            raise_errors=raise_errors,
            tools=tools,
            tool_choice=tool_choice,
            max_tool_calls=max_tool_calls,
        )

    async def ask_async(
        self,
        input: Union[str, dict[str, Any]],
        files: Optional[list[Union[str, Path, IO]]] = None,
        model: Optional[GoogleChatModelType] = None,
        context: Optional[dict[str, Any]] = None,
        *,
        choices: Optional[list[str]] = None,
        choices_optional: bool = False,
        stream: bool = False,
        use_history: bool = True,
        raise_errors: bool = False,
        tools: Optional[list] = None,
        tool_choice: str = "auto",
        max_tool_calls: int = 5,
    ) -> Union[Reply, AsyncGenerator[Reply, None]]:
        return await super().ask_async(
            input=input,
            files=files,
            model=model,
            context=context,
            choices=choices,
            choices_optional=choices_optional,
            stream=stream,
            use_history=use_history,
            raise_errors=raise_errors,
            tools=tools,
            tool_choice=tool_choice,
            max_tool_calls=max_tool_calls,
        )

    def embed(
        self,
        input: Union[str, list[str]],
        model: Optional[GoogleEmbeddingModelType] = None,
    ) -> Union[Embed, EmbedList]:
        embedding_model = cast(GoogleEmbeddingModelType, model or self.embedding_model)

        # Configure API key
        self._genai.configure(api_key=self.api_key)
        request_params = dict(
            model=str(embedding_model),
            contents=input,
            # config=EmbedContentConfig(output_dimensionality=10),
        )

        response: Optional[self._EmbedContentResponse] = None
        cache_key = None

        # Check cache if enabled
        if self.cache:
            from pyhub.llm.cache.utils import generate_cache_key

            cache_key = generate_cache_key("google", **request_params)
            cached_value = self.cache.get(cache_key)

            if cached_value is not None:
                try:
                    response = self._EmbedContentResponse.model_validate_json(cached_value)
                    # is_cached = True  # Currently not used
                except pydantic.ValidationError as e:
                    logger.error("Invalid cached value : %s", e)

        if response is None:
            logger.debug("request to google embed")
            response = self._genai.embed_content(**request_params)

            # Store in cache if enabled
            if self.cache and cache_key:
                # TODO: Implement proper caching for GenerateContentResponse
                pass  # self.cache.set(cache_key, response.model_dump_json())

        # TODO: response에 usage_metadata가 없음 - 캐시된 응답인 경우에도 None 유지
        usage = None
        if isinstance(input, str):
            return Embed(response.embeddings[0].values, usage=usage)
        return EmbedList([Embed(v.values) for v in response.embeddings], usage=usage)

    async def embed_async(
        self,
        input: Union[str, list[str]],
        model: Optional[GoogleEmbeddingModelType] = None,
    ) -> Union[Embed, EmbedList]:
        embedding_model = cast(GoogleEmbeddingModelType, model or self.embedding_model)

        # Configure API key
        self._genai.configure(api_key=self.api_key)
        request_params = dict(
            model=str(embedding_model),
            contents=input,
            # config=EmbedContentConfig(output_dimensionality=10),
        )

        response: Optional[self._EmbedContentResponse] = None
        cache_key = None

        # Check cache if enabled (using synchronous cache methods in async context)
        if self.cache:
            from pyhub.llm.cache.utils import generate_cache_key

            cache_key = generate_cache_key("google", **request_params)
            cached_value = self.cache.get(cache_key)

            if cached_value is not None:
                try:
                    response = self._EmbedContentResponse.model_validate_json(cached_value)
                    # is_cached = True  # Currently not used
                except pydantic.ValidationError as e:
                    logger.error("Invalid cached value : %s", e)

        if response is None:
            response = await self._genai.embed_content_async(**request_params)

            # Store in cache if enabled (using synchronous cache methods in async context)
            if self.cache and cache_key:
                # TODO: Implement proper caching for GenerateContentResponse
                pass  # self.cache.set(cache_key, response.model_dump_json())

        # TODO: response에 usage_metadata가 없음 - 캐시된 응답인 경우에도 None 유지
        usage = None
        if isinstance(input, str):
            return Embed(response.embeddings[0].values, usage=usage)
        return EmbedList([Embed(v.values) for v in response.embeddings], usage=usage)

    def _convert_tools_for_provider(self, tools):
        """Google Function Calling 형식으로 도구 변환"""
        from .tools import ProviderToolConverter

        return [ProviderToolConverter.to_google_function(tool) for tool in tools]

    def _extract_tool_calls_from_response(self, response):
        """Google 응답에서 function_call 추출"""
        tool_calls = []

        # Response가 Reply 객체인 경우 원본 응답에서 function_call 추출
        if hasattr(response, "_raw_response") and hasattr(response._raw_response, "candidates"):
            candidates = response._raw_response.candidates
            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        # function_call 속성이 있고 None이 아닌지 확인
                        if hasattr(part, "function_call") and part.function_call is not None:
                            # function_call 객체에 name과 args 속성이 있는지 확인
                            if hasattr(part.function_call, "name") and hasattr(part.function_call, "args"):
                                tool_calls.append(
                                    {
                                        "id": f"call_{len(tool_calls)}",  # Google doesn't provide call IDs
                                        "name": part.function_call.name,
                                        "arguments": part.function_call.args,
                                    }
                                )
                            else:
                                # function_call 객체에 필요한 속성이 없는 경우 로깅
                                logger.warning(
                                    "function_call object missing required attributes: %s", part.function_call
                                )

        return tool_calls

    def _make_ask_with_tools_sync(self, human_prompt, messages, tools, tool_choice, model, files):
        """Google Function Calling을 사용한 동기 호출"""
        try:
            from google.genai.types import FunctionDeclaration, Tool
        except ImportError:
            raise ImportError("google-genai package not installed. " "Install with: pip install pyhub-llm[google]")

        # 메시지 준비
        google_messages = []
        for msg in messages:
            google_messages.append(
                {"role": "user" if msg.role == "user" else "model", "parts": [{"text": msg.content}]}
            )

        if human_prompt:
            google_messages.append({"role": "user", "parts": [{"text": human_prompt}]})

        # 도구를 Google Tool 형식으로 변환
        google_tools = []
        if tools:
            function_declarations = []
            for tool in tools:
                function_declarations.append(
                    FunctionDeclaration(
                        name=tool["name"], description=tool["description"], parameters=tool["parameters"]
                    )
                )
            google_tools = [Tool(function_declarations=function_declarations)]

        # Google API 호출
        # Configure API key
        self._genai.configure(api_key=self.api_key)

        system_prompt = None
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
            google_messages = google_messages[1:]  # 시스템 메시지 제거

        config = self._GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=google_tools if google_tools else None,
        )

        try:
            # Create GenerativeModel and generate content
            generative_model = self._genai.GenerativeModel(
                model_name=model or self.model, system_instruction=system_prompt
            )
            response = generative_model.generate_content(contents=google_messages, generation_config=config)

            # Reply 객체로 변환
            usage = Usage(
                input=response.usage_metadata.prompt_token_count or 0,
                output=response.usage_metadata.candidates_token_count or 0,
            )

            reply = Reply(text=response.text or "", usage=usage)

            # 원본 응답을 저장하여 function_call 추출에 사용
            reply._raw_response = response

            return reply

        except Exception as e:
            logger.error(f"Google API error: {e}")
            return Reply(text=f"API Error: {str(e)}")

    async def _make_ask_with_tools_async(self, human_prompt, messages, tools, tool_choice, model, files):
        """Google Function Calling을 사용한 비동기 호출"""
        try:
            from google.genai.types import FunctionDeclaration, Tool
        except ImportError:
            raise ImportError("google-genai package not installed. " "Install with: pip install pyhub-llm[google]")

        # 메시지 준비
        google_messages = []
        for msg in messages:
            google_messages.append(
                {"role": "user" if msg.role == "user" else "model", "parts": [{"text": msg.content}]}
            )

        if human_prompt:
            google_messages.append({"role": "user", "parts": [{"text": human_prompt}]})

        # 도구를 Google Tool 형식으로 변환
        google_tools = []
        if tools:
            function_declarations = []
            for tool in tools:
                function_declarations.append(
                    FunctionDeclaration(
                        name=tool["name"], description=tool["description"], parameters=tool["parameters"]
                    )
                )
            google_tools = [Tool(function_declarations=function_declarations)]

        # Google API 호출
        # Configure API key
        self._genai.configure(api_key=self.api_key)

        system_prompt = None
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
            google_messages = google_messages[1:]  # 시스템 메시지 제거

        config = self._GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=google_tools if google_tools else None,
        )

        try:
            # Create GenerativeModel and generate content
            generative_model = self._genai.GenerativeModel(
                model_name=model or self.model, system_instruction=system_prompt
            )
            response = await generative_model.generate_content_async(contents=google_messages, generation_config=config)

            # Reply 객체로 변환
            usage = Usage(
                input=response.usage_metadata.prompt_token_count or 0,
                output=response.usage_metadata.candidates_token_count or 0,
            )

            reply = Reply(text=response.text or "", usage=usage)

            # 원본 응답을 저장하여 function_call 추출에 사용
            reply._raw_response = response

            return reply

        except Exception as e:
            logger.error(f"Google API error: {e}")
            return Reply(text=f"API Error: {str(e)}")


__all__ = ["GoogleLLM"]
