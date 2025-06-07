"""Type definitions for PyHub LLM."""

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union

from typing_extensions import NotRequired, TypedDict

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

OpenAIChatModelType: TypeAlias = Union[
    Literal[
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "chatgpt-4o-latest",
        "o1",
        "o1-mini",
    ],
    str,
]

AnthropicChatModelType: TypeAlias = Union[
    Literal[
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    str,
]

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
    files: Optional[List[Union[str, Path]]] = None

    def __iter__(self):
        for key, value in self.to_dict().items():
            yield key, value

    def to_dict(self) -> dict:
        d = asdict(self)
        if "files" in d:
            del d["files"]  # LLM API에서는 없는 속성이기에 제거
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

    def __str__(self) -> str:
        # choice가 있으면 choice를 반환, 없으면 text 반환
        return self.choice if self.choice is not None else self.text

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    @property
    def is_choice_response(self) -> bool:
        """choices 제약이 적용된 응답인지 확인"""
        return self.choice is not None or self.choice_index is not None


@dataclass
class ChainReply:
    values: Dict[str, Any] = field(default_factory=dict)
    reply_list: List[Reply] = field(default_factory=list)

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


@dataclass
class Embed:
    array: List[float]  # noqa
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
    arrays: List[Embed]  # noqa
    usage: Optional[Usage] = None

    def __iter__(self):
        return iter(self.arrays)

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, index):
        return self.arrays[index]

    def __str__(self):
        return str(self.arrays)


# Enums for standalone usage
class LanguageEnum(str, Enum):
    KOREAN = "korean"
    ENGLISH = "english"
    JAPANESE = "japanese"
    CHINESE = "chinese"


class LLMVendorEnum(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    UPSTAGE = "upstage"
    OLLAMA = "ollama"


class EmbeddingDimensionsEnum(str, Enum):
    D_768 = "768"
    D_1536 = "1536"
    D_3072 = "3072"
    D_4096 = "4096"


# Additional type definitions for responses
@dataclass
class LLMResponse:
    """Standard LLM response."""
    content: str
    usage: Optional[Usage] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    
    def __str__(self) -> str:
        return self.content


@dataclass
class EmbeddingResponse:
    """Embedding response."""
    embedding: List[float]
    usage: Optional[Usage] = None
    model: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.embedding)
    
    def __getitem__(self, index: int) -> float:
        return self.embedding[index]


@dataclass
class StreamResponse:
    """Streaming response chunk."""
    content: str
    is_final: bool = False
    usage: Optional[Usage] = None
    
    def __str__(self) -> str:
        return self.content


# Function/Tool calling types
@dataclass
class FunctionParameter:
    name: str
    type: str
    description: Optional[str] = None
    required: bool = True
    enum: Optional[List[str]] = None


@dataclass
class FunctionCall:
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class ToolCall:
    id: str
    type: Literal["function"]
    function: FunctionCall


# Model validation
def validate_model(vendor: LLMVendorType, model: str) -> bool:
    """Validate if a model is supported by a vendor."""
    if vendor == "openai":
        # Basic validation - in production, check against actual model list
        return model.startswith(("gpt-", "o1", "text-embedding"))
    elif vendor == "anthropic":
        return model.startswith("claude")
    elif vendor == "google":
        return model.startswith(("gemini", "text-embedding"))
    elif vendor == "ollama":
        # Ollama supports many models
        return True
    elif vendor == "upstage":
        return model.startswith(("solar", "embedding"))
    return False