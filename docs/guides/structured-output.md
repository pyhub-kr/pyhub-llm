# 구조화된 출력

pyhub-llm의 구조화된 출력 기능을 사용하여 타입 안전한 응답을 받는 방법을 알아봅니다.

## 개요

구조화된 출력은 LLM의 응답을 미리 정의된 형식으로 받을 수 있게 해주는 기능입니다. 이를 통해:

- ✅ 타입 안전성 보장
- ✅ 응답 형식 일관성
- ✅ 자동 검증 및 파싱
- ✅ IDE 자동 완성 지원

## 선택지에서 고르기

### 기본 사용법

```python
from pyhub.llm import LLM

llm = LLM.create("gpt-4o-mini")

# 단순 선택
reply = llm.ask(
    "다음 텍스트의 감정을 분석하세요: '오늘 정말 행복한 하루였어요!'",
    choices=["긍정", "부정", "중립"]
)

print(f"감정: {reply.choice}")        # "긍정"
print(f"확신도: {reply.confidence}")  # 0.95
print(f"인덱스: {reply.choice_index}") # 0
```

### 다양한 선택지 활용

```python
# 의도 분류
intents = ["질문", "요청", "불만", "칭찬", "정보제공", "기타"]

reply = llm.ask(
    "고객: '이 제품 언제 재입고 되나요?'",
    choices=intents
)
print(f"의도: {reply.choice}")  # "질문"

# 다중 레이블 분류 (각각 처리)
tags = ["기술", "과학", "정치", "경제", "문화", "스포츠"]
article = "AI 기술이 의료 분야에 혁신을 가져오고 있습니다..."

relevant_tags = []
for tag in tags:
    reply = llm.ask(
        f"다음 기사가 '{tag}' 카테고리에 해당하나요?\n\n{article}",
        choices=["예", "아니오"]
    )
    if reply.choice == "예":
        relevant_tags.append(tag)

print(f"관련 태그: {relevant_tags}")  # ["기술", "과학"]
```

### 동적 선택지 생성

```python
# 사용자 입력 기반 선택지
def get_dynamic_choices(category):
    if category == "음식":
        return ["한식", "중식", "일식", "양식", "기타"]
    elif category == "운동":
        return ["유산소", "근력", "스트레칭", "요가", "필라테스"]
    else:
        return ["옵션1", "옵션2", "옵션3"]

category = "음식"
choices = get_dynamic_choices(category)

reply = llm.ask(
    "김치찌개는 어떤 종류의 음식인가요?",
    choices=choices
)
print(reply.choice)  # "한식"
```

## Pydantic 스키마 사용

### 기본 스키마 정의

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Person(BaseModel):
    name: str = Field(description="사람의 이름")
    age: int = Field(description="나이", ge=0, le=150)
    email: Optional[str] = Field(None, description="이메일 주소")
    hobbies: List[str] = Field(default_factory=list, description="취미 목록")

# 스키마 사용
reply = llm.ask(
    "다음 정보로 사람 프로필을 만들어주세요: 홍길동, 25살, 축구와 독서를 좋아함",
    schema=Person
)

person = reply.structured_data
print(f"이름: {person.name}")
print(f"나이: {person.age}")
print(f"취미: {', '.join(person.hobbies)}")
```

### 중첩된 스키마

```python
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: Optional[str] = None

class Company(BaseModel):
    name: str
    industry: str
    employee_count: int
    address: Address
    is_public: bool

class JobPosting(BaseModel):
    title: str
    company: Company
    salary_range: tuple[int, int]
    requirements: List[str]
    benefits: List[str]
    remote_allowed: bool
    posted_date: datetime

# 복잡한 구조 파싱
reply = llm.ask(
    """
    다음 구인 공고 정보를 구조화하세요:
    
    삼성전자에서 AI 엔지니어를 모집합니다.
    - 연봉: 6000-8000만원
    - 요구사항: Python, ML 경험 3년 이상, 석사 우대
    - 복지: 4대보험, 스톡옵션, 유연근무
    - 근무지: 서울 강남구, 재택근무 가능
    """,
    schema=JobPosting
)

job = reply.structured_data
print(f"회사: {job.company.name}")
print(f"직무: {job.title}")
print(f"연봉: {job.salary_range[0]:,}만원 ~ {job.salary_range[1]:,}만원")
print(f"재택근무: {'가능' if job.remote_allowed else '불가'}")
```

### Enum과 제약 조건

```python
from enum import Enum
from pydantic import BaseModel, Field, validator

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Status(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"

class Task(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    description: str
    priority: Priority
    status: Status = Status.TODO
    estimated_hours: float = Field(..., gt=0, le=100)
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    
    @validator('due_date')
    def due_date_must_be_future(cls, v):
        if v and v < datetime.now():
            raise ValueError('Due date must be in the future')
        return v

# 사용
reply = llm.ask(
    "버그 수정 작업을 생성해주세요. 로그인 페이지 에러, 긴급, 3시간 예상",
    schema=Task
)

task = reply.structured_data
print(f"작업: {task.title}")
print(f"우선순위: {task.priority.value}")
print(f"예상 시간: {task.estimated_hours}시간")
```

## 고급 스키마 패턴

### Union 타입 처리

```python
from typing import Union, Literal

class TextContent(BaseModel):
    type: Literal["text"]
    content: str

class ImageContent(BaseModel):
    type: Literal["image"]
    url: str
    alt_text: Optional[str] = None

class VideoContent(BaseModel):
    type: Literal["video"]
    url: str
    duration_seconds: int
    thumbnail_url: Optional[str] = None

Content = Union[TextContent, ImageContent, VideoContent]

class Article(BaseModel):
    title: str
    author: str
    contents: List[Content]
    tags: List[str]

# 멀티미디어 기사 파싱
reply = llm.ask(
    """
    다음 기사를 구조화하세요:
    
    제목: AI의 미래
    저자: 김철수
    
    내용:
    1. AI는 우리의 미래입니다. (텍스트)
    2. [AI 로봇 이미지] (이미지)
    3. AI 데모 영상 (2분 30초) (비디오)
    
    태그: AI, 미래, 기술
    """,
    schema=Article
)

article = reply.structured_data
for content in article.contents:
    print(f"콘텐츠 타입: {content.type}")
```

### 재귀적 구조

```python
from typing import List, Optional

class TreeNode(BaseModel):
    name: str
    value: Optional[int] = None
    children: List['TreeNode'] = Field(default_factory=list)

# Forward reference 해결
TreeNode.model_rebuild()

class FileSystem(BaseModel):
    root: TreeNode

# 파일 시스템 구조 파싱
reply = llm.ask(
    """
    다음 디렉토리 구조를 트리로 만들어주세요:
    
    project/
    ├── src/
    │   ├── main.py (100)
    │   └── utils.py (50)
    ├── tests/
    │   └── test_main.py (80)
    └── README.md (20)
    
    괄호 안은 파일 크기(KB)입니다.
    """,
    schema=FileSystem
)

def print_tree(node: TreeNode, indent: int = 0):
    print("  " * indent + f"{node.name}" + (f" ({node.value}KB)" if node.value else ""))
    for child in node.children:
        print_tree(child, indent + 1)

print_tree(reply.structured_data.root)
```

### 동적 스키마 생성

```python
from typing import Dict, Any

def create_form_schema(fields: Dict[str, str]):
    """동적으로 Pydantic 모델 생성"""
    field_definitions = {}
    
    for field_name, field_type in fields.items():
        if field_type == "string":
            field_definitions[field_name] = (str, Field(...))
        elif field_type == "integer":
            field_definitions[field_name] = (int, Field(...))
        elif field_type == "boolean":
            field_definitions[field_name] = (bool, Field(...))
        elif field_type == "list":
            field_definitions[field_name] = (List[str], Field(default_factory=list))
    
    # 동적 모델 생성
    DynamicModel = type(
        'DynamicModel',
        (BaseModel,),
        {
            '__annotations__': {k: v[0] for k, v in field_definitions.items()},
            **{k: v[1] for k, v in field_definitions.items()}
        }
    )
    
    return DynamicModel

# 사용 예시
form_fields = {
    "name": "string",
    "age": "integer",
    "is_student": "boolean",
    "interests": "list"
}

DynamicForm = create_form_schema(form_fields)

reply = llm.ask(
    "다음 정보로 폼을 채워주세요: 김영희, 22살, 대학생, 프로그래밍과 디자인에 관심",
    schema=DynamicForm
)

form_data = reply.structured_data
print(form_data.model_dump())
```

## 실전 활용 예제

### 데이터 추출 파이프라인

```python
class ExtractedEntity(BaseModel):
    name: str
    type: str
    confidence: float = Field(..., ge=0, le=1)

class DocumentAnalysis(BaseModel):
    summary: str = Field(..., max_length=200)
    entities: List[ExtractedEntity]
    key_phrases: List[str]
    sentiment: Literal["positive", "negative", "neutral"]
    language: str

def analyze_document(text: str) -> DocumentAnalysis:
    llm = LLM.create("gpt-4o-mini")
    
    reply = llm.ask(
        f"""
        다음 문서를 분석하세요:
        
        {text}
        
        요약, 주요 엔티티, 핵심 구문, 감정, 언어를 추출하세요.
        """,
        schema=DocumentAnalysis
    )
    
    return reply.structured_data

# 사용
document = """
애플이 새로운 아이폰 16을 발표했습니다. 
팀 쿡 CEO는 "이번 제품은 혁신의 정점"이라고 말했습니다.
가격은 $999부터 시작하며, 9월 15일 출시 예정입니다.
"""

analysis = analyze_document(document)
print(f"요약: {analysis.summary}")
print(f"감정: {analysis.sentiment}")
print(f"엔티티: {[f'{e.name}({e.type})' for e in analysis.entities]}")
```

### API 응답 변환

```python
class APIResponse(BaseModel):
    """외부 API 응답을 표준화"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class UserData(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

def parse_api_response(raw_response: str) -> UserData:
    llm = LLM.create("gpt-4o-mini")
    
    # 먼저 API 응답 파싱
    api_reply = llm.ask(
        f"다음 API 응답을 파싱하세요:\n{raw_response}",
        schema=APIResponse
    )
    
    if not api_reply.structured_data.success:
        raise ValueError(f"API 에러: {api_reply.structured_data.error}")
    
    # 사용자 데이터 추출
    user_reply = llm.ask(
        f"다음 데이터에서 사용자 정보를 추출하세요:\n{api_reply.structured_data.data}",
        schema=UserData
    )
    
    return user_reply.structured_data
```

### 설정 파일 생성

```python
class DatabaseConfig(BaseModel):
    host: str
    port: int = 5432
    database: str
    username: str
    ssl_enabled: bool = True

class CacheConfig(BaseModel):
    type: Literal["redis", "memcached", "in-memory"]
    ttl_seconds: int = 3600
    max_size_mb: Optional[int] = None

class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None

class AppConfig(BaseModel):
    app_name: str
    version: str
    debug_mode: bool
    database: DatabaseConfig
    cache: CacheConfig
    logging: LoggingConfig

def generate_config(requirements: str) -> AppConfig:
    llm = LLM.create("gpt-4o-mini")
    
    reply = llm.ask(
        f"""
        다음 요구사항으로 애플리케이션 설정을 생성하세요:
        
        {requirements}
        
        모든 필수 설정을 포함하고 보안을 고려하세요.
        """,
        schema=AppConfig
    )
    
    return reply.structured_data

# 사용
requirements = """
- 프로덕션 환경용 웹 애플리케이션
- PostgreSQL 데이터베이스 사용
- Redis 캐싱 필요
- 에러 레벨 로깅
"""

config = generate_config(requirements)
print(f"앱 이름: {config.app_name}")
print(f"DB 호스트: {config.database.host}")
print(f"캐시 타입: {config.cache.type}")
```

## 에러 처리와 검증

### 스키마 검증 실패 처리

```python
from pydantic import ValidationError

class StrictUserProfile(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., ge=18, le=100)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')

def safe_parse_profile(text: str) -> Optional[StrictUserProfile]:
    llm = LLM.create("gpt-4o-mini")
    
    try:
        reply = llm.ask(
            f"다음에서 사용자 프로필 정보를 추출하세요: {text}",
            schema=StrictUserProfile
        )
        return reply.structured_data
    except ValidationError as e:
        print(f"검증 실패: {e}")
        
        # 재시도 with 힌트
        retry_reply = llm.ask(
            f"""
            다음에서 사용자 프로필을 추출하세요: {text}
            
            주의사항:
            - 이름은 2-50자
            - 나이는 18-100
            - 이메일은 유효한 형식
            
            검증 에러: {e}
            """,
            schema=StrictUserProfile
        )
        return retry_reply.structured_data
    except Exception as e:
        print(f"파싱 실패: {e}")
        return None
```

### 부분 스키마 매칭

```python
from typing import Any

class PartialData(BaseModel):
    class Config:
        extra = "allow"  # 추가 필드 허용
    
    required_field: str
    optional_field: Optional[str] = None

def extract_partial_data(text: str) -> Dict[str, Any]:
    llm = LLM.create("gpt-4o-mini")
    
    reply = llm.ask(
        f"다음에서 가능한 모든 정보를 추출하세요: {text}",
        schema=PartialData
    )
    
    data = reply.structured_data
    # 모든 필드 (extra 포함) 가져오기
    all_data = data.model_dump()
    
    return all_data
```

## 성능 최적화

### 스키마 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_schema(schema_name: str) -> type[BaseModel]:
    """자주 사용하는 스키마 캐싱"""
    schemas = {
        "person": Person,
        "task": Task,
        "article": Article,
    }
    return schemas.get(schema_name)

# Stateless 모드와 함께 사용
extractor = LLM.create("gpt-4o-mini", stateless=True)

# 반복 작업에서 스키마 재사용
schema = get_cached_schema("person")
for text in large_text_list:
    reply = extractor.ask(f"Extract: {text}", schema=schema)
    process_person(reply.structured_data)
```

### 배치 처리

```python
async def batch_extract(texts: List[str], schema: type[BaseModel]):
    """비동기 배치 처리"""
    llm = LLM.create("gpt-4o-mini", stateless=True)
    
    tasks = []
    for text in texts:
        task = llm.ask_async(
            f"Extract information from: {text}",
            schema=schema
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return [r.structured_data for r in results]

# 사용
texts = ["문서1", "문서2", "문서3", ...]
extracted_data = asyncio.run(batch_extract(texts, DocumentAnalysis))
```

## 모범 사례

### 1. 명확한 필드 설명

```python
class WellDocumentedSchema(BaseModel):
    """사용자 주문 정보"""
    
    order_id: str = Field(
        ...,
        description="주문 고유 ID (예: ORD-2024-001)",
        example="ORD-2024-001"
    )
    
    total_amount: float = Field(
        ...,
        description="총 주문 금액 (원화)",
        ge=0,
        example=50000
    )
    
    items: List[str] = Field(
        ...,
        description="주문한 상품명 목록",
        min_items=1,
        example=["노트북", "마우스"]
    )
```

### 2. 점진적 복잡도

```python
# 간단한 스키마부터 시작
class SimpleProduct(BaseModel):
    name: str
    price: float

# 필요에 따라 확장
class DetailedProduct(SimpleProduct):
    description: str
    category: str
    in_stock: bool
    
# 최종적으로 복잡한 스키마
class FullProduct(DetailedProduct):
    sku: str
    manufacturer: Company
    specifications: Dict[str, str]
    reviews: List[Review]
```

### 3. 프롬프트 엔지니어링

```python
def create_extraction_prompt(text: str, schema: type[BaseModel]) -> str:
    """스키마 기반 프롬프트 생성"""
    
    schema_description = schema.schema_json(indent=2)
    
    return f"""
    다음 텍스트에서 정보를 추출하여 JSON으로 변환하세요.
    
    텍스트:
    {text}
    
    요구되는 JSON 스키마:
    {schema_description}
    
    규칙:
    1. 정확히 스키마에 맞춰 추출
    2. 찾을 수 없는 정보는 null 또는 기본값 사용
    3. 타입을 정확히 맞춰주세요
    """
```

## 다음 단계

- [고급 기능](advanced.md) - 스트리밍, 비동기, 캐싱 등
- [API 레퍼런스](../api-reference/index.md) - 전체 API 문서
- [예제](../examples/index.md) - 실제 사용 사례