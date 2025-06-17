from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import IO, Any, Literal, TypeAlias, Union

from pydantic import BaseModel
from typing_extensions import Optional

from pyhub.llm.exceptions import ValidationError
from pyhub.llm.utils.enums import TextChoices
from pyhub.llm.utils.type_utils import enum_to_flatten_set, type_to_flatten_set

# Optional imports
try:
    from anthropic.types import ModelParam as AnthropicChatModelType
except ImportError:
    # anthropic이 설치되지 않은 경우 Union[Literal, str] 타입으로 대체
    # 실제 anthropic API의 ModelParam과 동일한 구조
    AnthropicChatModelType = Union[
        Literal[
            "claude-3-opus-latest",
            "claude-3-opus-20240229",
            "claude-3-7-sonnet-latest",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-latest",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-sonnet-20240229",
            "claude-3-5-haiku-latest",
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
        ],
        str,
    ]

try:
    from openai.types import ChatModel as _OpenAIChatModel
except ImportError:
    # openai가 설치되지 않은 경우 실제 ChatModel과 동일한 Literal 타입으로 대체
    _OpenAIChatModel = Literal[
        "o3-mini",
        "o3-mini-2025-01-31",
        "o1",
        "o1-2024-12-17",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-mini",
        "o1-mini-2024-09-12",
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4o-audio-preview",
        "gpt-4o-audio-preview-2024-10-01",
        "gpt-4o-audio-preview-2024-12-17",
        "gpt-4o-mini-audio-preview",
        "gpt-4o-mini-audio-preview-2024-12-17",
        "gpt-4o-search-preview",
        "gpt-4o-mini-search-preview",
        "gpt-4o-search-preview-2025-03-11",
        "gpt-4o-mini-search-preview-2025-03-11",
        "chatgpt-4o-latest",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-16k-0613",
    ]

#
# Vendor
#

LLMVendorType: TypeAlias = Literal["openai", "anthropic", "google", "upstage", "ollama"]

#
# Language
#

LanguageType: TypeAlias = Union[
    Literal["korean", "english", "japanese", "chinese"],
    str,
]

#
# Embedding
#

OpenAIEmbeddingModelType: TypeAlias = Union[
    Literal[
        "text-embedding-ada-002",  # 1536 차원
        "text-embedding-3-small",  # 1536 차원
        "text-embedding-3-large",  # 3072 차원
    ],
    str,
]

# https://console.upstage.ai/docs/capabilities/embeddings
UpstageEmbeddingModelType: TypeAlias = Literal[
    "embedding-query",  # 검색어 목적 (4096차원)
    "embedding-passage",  # 문서의 일부, 문장 또는 긴 텍스트 목적 (4096차원)
]


OllamaEmbeddingModelType: TypeAlias = Union[
    Literal[
        "nomic-embed-text",  # 768 차원
        "avr/sfr-embedding-mistral",  # 4096 차원
    ],
    str,
]

# https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings?hl=ko
GoogleEmbeddingModelType: TypeAlias = Literal["text-embedding-004"]  # 768 차원

LLMEmbeddingModelType = Union[
    OpenAIEmbeddingModelType,
    UpstageEmbeddingModelType,
    OllamaEmbeddingModelType,
    GoogleEmbeddingModelType,
    str,
]


#
# Chat
#

OpenAIChatModelType: TypeAlias = Union[_OpenAIChatModel, str]

AnthropicChatModelType: TypeAlias = Union[AnthropicChatModelType, str]

# https://console.upstage.ai/docs/capabilities/chat
UpstageChatModelType: TypeAlias = Union[
    Literal[
        "solar-pro2-preview",
        "solar-pro",
        "solar-mini",
    ],
    str,
]

OllamaChatModelType: TypeAlias = Union[
    Literal[
        # tools, 70b : https://ollama.com/library/llama3.3
        "llama3.3",
        "llama3.3:70b",
        # tools, 1b, 3b : https://ollama.com/library/llama3.2
        "llama3.2",
        "llama3.2:1b",
        "llama3.2:3b",
        # tools, 8b, 70b, 405b : https://ollama.com/library/llama3.1
        "llama3.1",
        "llama3.1:8b",
        "llama3.1:70b",
        "llama3.1:405b",
        # tools, 7b : https://ollama.com/library/mistral
        "mistral",
        "mistral:7b",
        # tools, 0.5b, 1.5b, 7b, 72b : https://ollama.com/library/qwen2
        "qwen2",
        "qwen2:0.5b",
        "qwen2:1.5b",
        "qwen2:7b",
        "qwen2:72b",
        # vision, 1b, 4b, 12b, 27b : https://ollama.com/library/gemma3
        "gemma3",
        "gemma3:1b",
        "gemma3:4b",
        "gemma3:12b",
        "gemma3:27b",
    ],
    str,
]

# https://ai.google.dev/gemini-api/docs/models/gemini?hl=ko
GoogleChatModelType: TypeAlias = Union[
    Literal[
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ],
    str,
]


LLMChatModelType: TypeAlias = Union[
    OpenAIChatModelType,
    AnthropicChatModelType,
    UpstageChatModelType,
    GoogleChatModelType,
    OllamaChatModelType,
]


LLMModelType = Union[LLMChatModelType, LLMEmbeddingModelType]

#
# Groundedness Check
#

# https://console.upstage.ai/docs/capabilities/groundedness-check#available-models
UpstageGroundednessCheckModel: TypeAlias = Literal["groundedness-check",]


#
# Types
#


@dataclass
class GroundednessCheck:
    is_grounded: Optional[bool] = None  # grounded (True), notGrounded (False), notSure (None)
    usage: Optional["Usage"] = None

    def __bool__(self):
        return self.is_grounded


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "function"]
    content: str
    files: Optional[list[Union[str, Path, IO]]] = None
    # Tool calling 과정 기록 (assistant 메시지에만 사용)
    tool_interactions: Optional[list[dict]] = None

    def __iter__(self):
        for key, value in self.to_dict().items():
            yield key, value

    def to_dict(self) -> dict:
        d = asdict(self)
        del d["files"]  # LLM API에서는 없는 속성이기에 제거
        if "tool_interactions" in d:
            del d["tool_interactions"]  # LLM API에서는 없는 속성이기에 제거
        return d


@dataclass
class Usage:
    input: int = 0
    output: int = 0

    @property
    def total(self) -> int:
        """총 토큰 수 (input + output)"""
        return self.input + self.output

    def __add__(self, other):
        if isinstance(other, Usage):
            return Usage(input=self.input + other.input, output=self.output + other.output)
        return NotImplemented

    def __bool__(self):
        if self.input == 0 and self.output == 0:
            return False
        return True


@dataclass
class Price:
    input_usd: Optional[Decimal] = None
    output_usd: Optional[Decimal] = None
    usd: Optional[Decimal] = None
    krw: Optional[Decimal] = None
    rate_usd: int = 1500

    def __post_init__(self):
        self.input_usd = self.input_usd or Decimal("0")
        self.output_usd = self.output_usd or Decimal("0")

        if not isinstance(self.input_usd, Decimal):
            self.input_usd = Decimal(str(self.input_usd))
        if not isinstance(self.output_usd, Decimal):
            self.output_usd = Decimal(str(self.output_usd))
        if self.usd is not None and not isinstance(self.usd, Decimal):
            self.usd = Decimal(str(self.usd))
        if self.krw is not None and not isinstance(self.krw, Decimal):
            self.krw = Decimal(str(self.krw))

        if self.usd is None:
            self.usd = self.input_usd + self.output_usd

        if self.krw is None:
            self.krw = self.usd * Decimal(self.rate_usd)


@dataclass
class Reply:
    text: str = ""
    usage: Optional[Usage] = None
    # choices가 제공된 경우에만 설정
    choice: Optional[str] = None  # 선택된 값 (choices 중 하나 또는 None)
    choice_index: Optional[int] = None  # 선택된 인덱스
    confidence: Optional[float] = None  # 선택 신뢰도 (0.0 ~ 1.0)
    # 구조화된 출력 관련
    structured_data: Optional[BaseModel] = None  # 파싱된 Pydantic 모델 인스턴스
    validation_errors: Optional[list[str]] = None  # 스키마 검증 실패 시 에러 메시지

    def __str__(self) -> str:
        # choice가 있으면 choice를 반환, 없으면 text 반환
        return self.choice if self.choice is not None else self.text

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    @property
    def is_choice_response(self) -> bool:
        """choices 제약이 적용된 응답인지 확인"""
        return self.choice is not None or self.choice_index is not None

    @property
    def has_structured_data(self) -> bool:
        """구조화된 데이터가 있는지 확인"""
        return self.structured_data is not None
    
    def print(self, markdown: bool = True, **kwargs) -> str:
        """Print the response with optional markdown rendering.
        
        Args:
            markdown: Whether to render as markdown
            **kwargs: Additional arguments passed to display()
            
        Returns:
            str: The text content
        """
        from pyhub.llm.display import display
        return display(self, markdown=markdown, **kwargs)


@dataclass
class ChainReply:
    values: dict[str, Any] = field(default_factory=dict)
    reply_list: list[Reply] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.reply_list)

    @property
    def text(self) -> str:
        try:
            return self.reply_list[-1].text
        except IndexError:
            return ""

    @property
    def usage(self) -> Optional[Usage]:
        try:
            return self.reply_list[-1].usage
        except IndexError:
            return None

    def __getitem__(self, key) -> Any:
        return self.values.get(key)
    
    def print(self, markdown: bool = True, **kwargs) -> str:
        """Print the response with optional markdown rendering.
        
        Args:
            markdown: Whether to render as markdown
            **kwargs: Additional arguments passed to display()
            
        Returns:
            str: The text content
        """
        from pyhub.llm.display import display
        return display(self, markdown=markdown, **kwargs)


@dataclass
class Embed:
    array: list[float]  # noqa
    usage: Optional[Usage] = None

    def __iter__(self):
        return iter(self.array)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

    def __str__(self):
        return str(self.array)


@dataclass
class EmbedList:
    arrays: list[Embed]  # noqa
    usage: Optional[Usage] = None

    def __iter__(self):
        return iter(self.arrays)

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, index):
        return self.arrays[index]

    def __str__(self):
        return str(self.arrays)


@dataclass
class ImageReply:
    """Response from image generation."""
    url: Optional[str] = None
    base64_data: Optional[str] = None
    revised_prompt: Optional[str] = None
    size: str = "1024x1024"
    model: str = ""
    usage: Optional[Usage] = None
    
    def __str__(self) -> str:
        """String representation of the image reply."""
        if self.url:
            return f"Image URL: {self.url}"
        elif self.base64_data:
            return f"Image (base64): {self.size}"
        else:
            return "Image (no data)"
    
    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.size.split('x')[0])
    
    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.size.split('x')[1])
    
    def print(self, markdown: bool = True, **kwargs) -> str:
        """Print the image reply with optional formatting.
        
        Args:
            markdown: Whether to render as markdown
            **kwargs: Additional arguments
            
        Returns:
            str: The string representation
        """
        text = str(self)
        if self.revised_prompt:
            text += f"\nRevised prompt: {self.revised_prompt}"
        if self.usage:
            text += f"\nUsage: {self.usage}"
        print(text)
        return text
    
    def display(self) -> None:
        """Display image in Jupyter notebook or print URL."""
        try:
            from IPython.display import Image, display
            if self.url:
                display(Image(url=self.url))
            elif self.base64_data:
                import base64
                display(Image(data=base64.b64decode(self.base64_data)))
        except ImportError:
            # Not in Jupyter environment
            print(str(self))
    
    def _extract_filename_from_url(self) -> str:
        """Extract filename from URL or generate one."""
        if self.url:
            from urllib.parse import urlparse, unquote
            import re
            
            parsed = urlparse(self.url)
            # Extract filename from URL path
            filename = unquote(parsed.path.split('/')[-1])
            
            # If no filename or no extension, generate one
            if not filename or '.' not in filename:
                # Try to extract ID from URL pattern
                match = re.search(r'/([a-zA-Z0-9_-]+)(?:\?|$)', self.url)
                if match:
                    filename = f"{match.group(1)}.png"
                else:
                    filename = f"image_{hash(self.url) % 1000000}.png"
        else:
            # For base64 data
            filename = f"image_{hash(self.base64_data[:100] if self.base64_data else '') % 1000000}.png"
        
        return filename
    
    def _get_unique_filepath(self, directory: Path, filename: str) -> Path:
        """Get unique filepath to avoid overwrites."""
        filepath = directory / filename
        
        if not filepath.exists():
            return filepath
        
        # Split filename and extension
        stem = filepath.stem
        suffix = filepath.suffix
        
        # Add number to create unique filename
        counter = 1
        while True:
            new_filepath = directory / f"{stem}_{counter}{suffix}"
            if not new_filepath.exists():
                return new_filepath
            counter += 1
    
    def _prepare_filepath(
        self, 
        path: Optional[Union[str, Path]], 
        overwrite: bool, 
        create_dirs: bool
    ) -> Path:
        """Prepare the filepath for saving."""
        if path is None:
            # Save to current directory
            save_dir = Path.cwd()
            filename = self._extract_filename_from_url()
            filepath = self._get_unique_filepath(save_dir, filename)
        else:
            path = Path(path)
            
            if path.is_dir() or (not path.exists() and not path.suffix):
                # Directory specified
                save_dir = path
                filename = self._extract_filename_from_url()
                
                # Create directory if needed
                if create_dirs and not save_dir.exists():
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                filepath = self._get_unique_filepath(save_dir, filename)
            else:
                # File path specified
                filepath = path
                
                # Create parent directory if needed
                if create_dirs and not filepath.parent.exists():
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # Check overwrite
                if filepath.exists() and not overwrite:
                    raise FileExistsError(
                        f"File already exists: {filepath}. Use overwrite=True to replace."
                    )
        
        return filepath
    
    def save(
        self, 
        path: Optional[Union[str, Path]] = None, 
        *, 
        overwrite: bool = False,
        create_dirs: bool = True
    ) -> Path:
        """Save image to local file.
        
        Args:
            path: Save path (optional)
                - None: Save to current directory with auto filename
                - Directory path: Save to directory with auto filename
                - File path: Save with specified filename
            overwrite: Whether to overwrite existing file
            create_dirs: Whether to create directories automatically
            
        Returns:
            Path: Path to saved file
            
        Raises:
            ValueError: No image data available
            FileExistsError: File exists and overwrite=False
            IOError: Failed to save file
        """
        if not self.url and not self.base64_data:
            raise ValueError("No image data available to save")
        
        filepath = self._prepare_filepath(path, overwrite, create_dirs)
        
        # Get image data
        if self.url:
            import httpx
            response = httpx.get(self.url)
            response.raise_for_status()
            image_data = response.content
        else:  # base64_data
            import base64
            image_data = base64.b64decode(self.base64_data)
        
        # Check if we can detect format with Pillow
        try:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(image_data))
            format = img.format.lower() if img.format else 'png'
            
            # Add extension if missing
            if not filepath.suffix:
                filepath = filepath.with_suffix(f'.{format}')
        except ImportError:
            # Pillow not available, use default extension if missing
            if not filepath.suffix:
                filepath = filepath.with_suffix('.png')
        
        # Save file
        filepath.write_bytes(image_data)
        return filepath
    
    async def save_async(
        self, 
        path: Optional[Union[str, Path]] = None, 
        *, 
        overwrite: bool = False,
        create_dirs: bool = True
    ) -> Path:
        """Asynchronously save image to local file.
        
        Args:
            path: Save path (optional)
            overwrite: Whether to overwrite existing file
            create_dirs: Whether to create directories automatically
            
        Returns:
            Path: Path to saved file
        """
        if not self.url and not self.base64_data:
            raise ValueError("No image data available to save")
        
        filepath = self._prepare_filepath(path, overwrite, create_dirs)
        
        # Get image data asynchronously
        if self.url:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(self.url)
                response.raise_for_status()
                image_data = response.content
        else:  # base64_data
            import base64
            image_data = base64.b64decode(self.base64_data)
        
        # Check format with Pillow if available
        try:
            from PIL import Image
            import io
            import asyncio
            
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(
                None, 
                lambda: Image.open(io.BytesIO(image_data))
            )
            format = img.format.lower() if img.format else 'png'
            
            # Add extension if missing
            if not filepath.suffix:
                filepath = filepath.with_suffix(f'.{format}')
        except ImportError:
            # Pillow not available
            if not filepath.suffix:
                filepath = filepath.with_suffix('.png')
        
        # Save file asynchronously
        try:
            import aiofiles
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(image_data)
        except ImportError:
            # Fallback to sync write if aiofiles not available
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, filepath.write_bytes, image_data)
        
        return filepath
    
    def to_pil(self) -> "PIL.Image.Image":
        """Convert to PIL Image object.
        
        Returns:
            PIL.Image.Image: PIL image object
            
        Raises:
            ImportError: Pillow not installed
            ValueError: No image data available
        """
        try:
            from PIL import Image
            import io
        except ImportError:
            raise ImportError(
                "Pillow library is required for this feature. "
                "Install it with: pip install 'pyhub-llm[image]' or pip install Pillow"
            )
        
        if self.url:
            import httpx
            response = httpx.get(self.url)
            response.raise_for_status()
            image_data = response.content
        elif self.base64_data:
            import base64
            image_data = base64.b64decode(self.base64_data)
        else:
            raise ValueError("No image data available")
            
        return Image.open(io.BytesIO(image_data))
    
    async def to_pil_async(self) -> "PIL.Image.Image":
        """Asynchronously convert to PIL Image object.
        
        Returns:
            PIL.Image.Image: PIL image object
            
        Raises:
            ImportError: Pillow not installed
            ValueError: No image data available
        """
        try:
            from PIL import Image
            import io
        except ImportError:
            raise ImportError(
                "Pillow library is required for this feature. "
                "Install it with: pip install 'pyhub-llm[image]' or pip install Pillow"
            )
        
        # Download image data asynchronously
        if self.url:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(self.url)
                response.raise_for_status()
                image_data = response.content
        elif self.base64_data:
            import base64
            image_data = base64.b64decode(self.base64_data)
        else:
            raise ValueError("No image data available")
        
        # PIL operations in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: Image.open(io.BytesIO(image_data))
        )


class LanguageEnum(TextChoices):
    KOREAN = "korean"
    ENGLISH = "english"
    JAPANESE = "japanese"
    CHINESE = "chinese"


class LLMVendorEnum(TextChoices):
    OPENAI = "openai", "OpenAI"
    ANTHROPIC = "anthropic", "Anthropic"
    GOOGLE = "google", "Google"
    UPSTAGE = "upstage", "Upstage"
    OLLAMA = "ollama", "Ollama"


class EmbeddingDimensionsEnum(TextChoices):
    D_768 = "768"
    D_1536 = "1536"
    D_3072 = "3072"


class LLMEmbeddingModelEnum(TextChoices):
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small", "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large", "text-embedding-3-large"
    TEXT_EMBEDDING_004 = "text-embedding-004", "text-embedding-004"
    TEXT_EMBEDDING_ADA_02 = "text-embedding-ada-002", "text-embedding-ada-002"


class OpenAIChatModelEnum(TextChoices):
    GPT_4O = "gpt-4o", "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini", "gpt-4o-mini"
    CHATGPT_4O_LATEST = "chatgpt-4o-latest", "chatgpt-4o-latest"
    O1 = "o1", "o1"
    O1_MINI = "o1-mini", "o1-mini"
    # O3_MINI = "o3-mini", "o3-mini"


# https://docs.anthropic.com/en/docs/about-claude/models/overview
class AnthropicChatModelEnum(TextChoices):
    # CLAUDE_OPUS_4_LATEST = "claude-opus-4-latest", "claude-opus-4-latest"
    # CLAUDE_OPUS_4_20250514 = "claude-opus-4-20250514", "claude-opus-4-20250514"
    CLAUDE_OPUS_3_LATEST = "claude-3-opus-latest", "claude-3-opus-latest"

    # CLAUDE_SONNET_4_20250514 = "claude-sonnet-4-20250514", "claude-sonnet-4-20250514"
    CLAUDE_SONNET_3_7_LATEST = "claude-3-7-sonnet-latest", "claude-3-7-sonnet-latest"
    CLAUDE_SONNET_3_7_20250219 = "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219"
    CLAUDE_SONNET_3_5_LATEST = "claude-3-5-sonnet-latest", "claude-3-5-sonnet-latest"
    CLAUDE_SONNET_3_5_20241022 = "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"

    CLAUDE_HAIKU_3_5_LATEST = "claude-3-5-haiku-latest", "claude-3-5-haiku-latest"
    CLAUDE_HAIKU_3_5_20241022 = "claude-3-5-haiku-20241022", "claude-3-5-haiku-20241022"


class UpstageChatModelEnum(TextChoices):
    UPSTAGE_SOLAR_PRO2_PREVIEW = "solar-pro2-preview"
    UPSTAGE_SOLAR_PRO = "solar-pro", "solar-pro"
    UPSTAGE_SOLAR_MINI = "solar-mini", "solar-mini"


class GoogleChatModelEnum(TextChoices):
    GEMINI_2_0_FLASH = "gemini-2.0-flash", "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite", "gemini-2.0-flash-lite"
    GEMINI_1_5_FLASH = "gemini-1.5-flash", "gemini-1.5-flash"
    GEMINI_1_5_FLASH_8B = "gemini-1.5-flash-8b", "gemini-1.5-flash-8b"
    GEMINI_1_5_PRO = "gemini-1.5-pro", "gemini-1.5-pro"


class OllamaChatModelEnum(TextChoices):
    LLAMA_3_3 = "llama3.3", "llama3.3"
    LLAMA_3_3_70B = "llama3.3:70b", "llama3.3:70b"
    LLAMA_3_2 = "llama3.2", "llama3.2"
    LLAMA_3_2_1B = "llama3.2:1b", "llama3.2:1b"
    LLAMA_3_2_3B = "llama3.2:3b", "llama3.2:3b"
    LLAMA_3_1 = "llama3.1", "llama3.1"
    LLAMA_3_1_8B = "llama3.1:8b", "llama3.1:8b"
    LLAMA_3_1_70B = "llama3.1:70b", "llama3.1:70b"
    LLAMA_3_1_405B = "llama3.1:405b", "llama3.1:405b"
    MISTRAL = "mistral", "mistral"
    MISTRAL_7B = "mistral:7b", "mistral:7b"
    QWEN2 = "qwen2", "qwen2"
    QWEN2_0_5B = "qwen2:0.5b", "qwen2:0.5b"
    QWEN2_1_5B = "qwen2:1.5b", "qwen2:1.5b"
    QWEN2_7B = "qwen2:7b", "qwen2:7b"
    QWEN2_72B = "qwen2:72b", "qwen2:72b"
    GEMMA3 = "gemma3", "gemma3"
    GEMMA3_1B = "gemma3:1b", "gemma3:1b"
    GEMMA3_4B = "gemma3:4b", "gemma3:4b"
    GEMMA3_12B = "gemma3:12b", "gemma3:12b"
    GEMMA3_27B = "gemma3:27b", "gemma3:27b"


# enum은 상속을 지원하지 않습니다.
class LLMChatModelEnum(TextChoices):
    # openai
    GPT_4O = "gpt-4o", "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini", "gpt-4o-mini"
    CHATGPT_4O_LATEST = "chatgpt-4o-latest", "chatgpt-4o-latest"
    O1 = "o1", "o1"
    O1_MINI = "o1-mini", "o1-mini"
    # O3_MINI = "o3-mini", "o3-mini"
    # AnthropicChatModelEnum
    # CLAUDE_OPUS_4_LATEST = "claude-opus-4-latest", "claude-opus-4-latest"
    # CLAUDE_OPUS_4_20250514 = "claude-opus-4-20250514", "claude-opus-4-20250514"
    CLAUDE_OPUS_3_LATEST = "claude-3-opus-latest", "claude-3-opus-latest"
    # CLAUDE_SONNET_4_20250514 = "claude-sonnet-4-20250514", "claude-sonnet-4-20250514"
    CLAUDE_SONNET_3_7_LATEST = "claude-3-7-sonnet-latest", "claude-3-7-sonnet-latest"
    CLAUDE_SONNET_3_7_20250219 = "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219"
    CLAUDE_SONNET_3_5_LATEST = "claude-3-5-sonnet-latest", "claude-3-5-sonnet-latest"
    CLAUDE_SONNET_3_5_20241022 = "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"
    CLAUDE_HAIKU_3_5_LATEST = "claude-3-5-haiku-latest", "claude-3-5-haiku-latest"
    CLAUDE_HAIKU_3_5_20241022 = "claude-3-5-haiku-20241022", "claude-3-5-haiku-20241022"
    # upstage
    UPSTAGE_SOLAR_PRO2_PREVIEW = "solar-pro2-preview", "solar-pro2-preview"
    UPSTAGE_SOLAR_PRO = "solar-pro", "solar-pro"
    UPSTAGE_SOLAR_MINI = "solar-mini", "solar-mini"
    # google
    GEMINI_2_0_FLASH = "gemini-2.0-flash", "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite", "gemini-2.0-flash-lite"
    GEMINI_1_5_FLASH = "gemini-1.5-flash", "gemini-1.5-flash"
    GEMINI_1_5_FLASH_8B = "gemini-1.5-flash-8b", "gemini-1.5-flash-8b"
    GEMINI_1_5_PRO = "gemini-1.5-pro", "gemini-1.5-pro"
    # ollama
    LLAMA_3_3 = "llama3.3", "llama3.3"
    LLAMA_3_3_70B = "llama3.3:70b", "llama3.3:70b"
    LLAMA_3_2 = "llama3.2", "llama3.2"
    LLAMA_3_2_1B = "llama3.2:1b", "llama3.2:1b"
    LLAMA_3_2_3B = "llama3.2:3b", "llama3.2:3b"
    LLAMA_3_1 = "llama3.1", "llama3.1"
    LLAMA_3_1_8B = "llama3.1:8B", "llama3.1:8B"
    LLAMA_3_1_70B = "llama3.1:70B", "llama3.1:70B"
    LLAMA_3_1_405B = "llama3.1:405B", "llama3.1:405B"
    MISTRAL = "mistral", "mistral"
    MISTRAL_7B = "mistral:7b", "mistral:7b"
    QWEN2 = "qwen2", "qwen2"
    QWEN2_0_5B = "qwen2:0.5b", "qwen2:0.5b"
    QWEN2_1_5B = "qwen2:1.5b", "qwen2:1.5b"
    QWEN2_7B = "qwen2:7b", "qwen2:7b"
    QWEN2_72B = "qwen2:72b", "qwen2:72b"
    GEMMA3 = "gemma3", "gemma3"
    GEMMA3_1B = "gemma3:1b", "gemma3:1b"
    GEMMA3_4B = "gemma3:4b", "gemma3:4b"
    GEMMA3_12B = "gemma3:12b", "gemma3:12b"
    GEMMA3_27B = "gemma3:27b", "gemma3:27b"

    @classmethod
    def validate_model(cls, llm_vendor: LLMVendorType, chat_model: LLMChatModelType) -> None:
        """
        지정된 vendor에 해당 model이 존재하는지 검사합니다.

        Args:
            llm_vendor: 검사할 벤더 타입 ('openai', 'anthropic', 'google', 'ollama', 'upstage' 등)
            chat_model: 검사할 모델 이름

        Raises:
            ValidationError
        """

        if llm_vendor == LLMVendorEnum.OPENAI.value:
            if chat_model not in OpenAIChatModelEnum:
                raise ValidationError(f"{chat_model} : Invalid OpenAI Model")
        elif llm_vendor == LLMVendorEnum.ANTHROPIC.value:
            if chat_model not in AnthropicChatModelEnum:
                raise ValidationError(f"{chat_model} : Invalid Anthropic Model")
        elif llm_vendor == LLMVendorEnum.GOOGLE.value:
            if chat_model not in GoogleChatModelEnum:
                raise ValidationError(f"{chat_model} : Invalid Google Model")
        elif llm_vendor == LLMVendorEnum.OLLAMA.value:
            if chat_model not in OllamaChatModelEnum:
                raise ValidationError(f"{chat_model} : Invalid OLLAMA Model")
        elif llm_vendor == LLMVendorEnum.UPSTAGE.value:
            if chat_model not in UpstageChatModelEnum:
                raise ValidationError(f"{chat_model} : Invalid UPSTAGE Model")
        else:
            raise ValueError(f"Unknown llm vendor: {llm_vendor}")


def check():
    assert enum_to_flatten_set(LLMVendorEnum) == type_to_flatten_set(
        LLMVendorType
    ), "Values in LLMVendorEnum and LLMVendorType do not match."
    assert enum_to_flatten_set(LanguageEnum) == type_to_flatten_set(
        LanguageType
    ), "Values in LanguageEnum and LanguageType do not match."

    set1 = enum_to_flatten_set(OpenAIChatModelEnum)
    set2 = type_to_flatten_set(OpenAIChatModelType)
    assert set1.issubset(set2), "OpenAIChatModelEnum is not a subset of OpenAIChatModelType."

    assert enum_to_flatten_set(UpstageChatModelEnum) == type_to_flatten_set(
        UpstageChatModelType
    ), "Values in UpstageChatModelEnum and UpstageChatModelType do not match."

    set1 = enum_to_flatten_set(AnthropicChatModelEnum)
    set2 = type_to_flatten_set(AnthropicChatModelType)
    assert set1.issubset(set2), f"AnthropicChatModelEnum is not a subset of AnthropicChatModelType. ({set1 - set2})"
    assert enum_to_flatten_set(GoogleChatModelEnum) == type_to_flatten_set(
        GoogleChatModelType
    ), "Values in GoogleChatModelEnum and GoogleChatModelType do not match."
    assert enum_to_flatten_set(OllamaChatModelEnum) == type_to_flatten_set(
        OllamaChatModelType
    ), "Values in OllamaChatModelEnum and OllamaChatModelType do not match."

    set1 = enum_to_flatten_set(LLMChatModelEnum)
    set2 = (
        enum_to_flatten_set(OpenAIChatModelEnum)
        | enum_to_flatten_set(AnthropicChatModelEnum)
        | enum_to_flatten_set(UpstageChatModelEnum)
        | enum_to_flatten_set(GoogleChatModelEnum)
        | enum_to_flatten_set(OllamaChatModelEnum)
    )
    if set1 - set2:
        assert (
            False
        ), f"LLMChatModelEnum does not match the union of all vendor-specific ChatModelEnums. ({set1 - set2})"
    elif set2 - set1:
        assert (
            False
        ), f"LLMChatModelEnum does not match the union of all vendor-specific ChatModelEnums. ({set2 - set1})"


check()
