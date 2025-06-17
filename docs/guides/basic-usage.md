# 기본 사용법

pyhub-llm의 기본적인 사용 방법을 알아봅니다.

## LLM 인스턴스 생성

### 자동 프로바이더 감지

가장 간단한 방법은 모델명을 통한 자동 프로바이더 감지입니다:

```python
from pyhub.llm import LLM

# 모델명으로 프로바이더 자동 감지
llm = LLM.create("gpt-4o-mini")  # OpenAI
llm = LLM.create("claude-3-5-haiku-latest")  # Anthropic
llm = LLM.create("gemini-2.0-flash-exp")  # Google
```

### 명시적 프로바이더 사용

특정 프로바이더를 직접 사용할 수도 있습니다:

```python
from pyhub.llm import OpenAILLM, AnthropicLLM

# OpenAI
openai_llm = OpenAILLM(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000
)

# Anthropic
anthropic_llm = AnthropicLLM(
    model="claude-3-5-haiku-latest",
    temperature=0.5
)
```

## 기본 질문하기

### 텍스트 응답 받기

```python
# 간단한 질문
reply = llm.ask("파이썬의 장점을 3가지 알려주세요")
print(reply.text)

# 응답 메타데이터 확인
print(f"사용 토큰: {reply.usage.total_tokens}")
print(f"응답 시간: {reply.elapsed_time:.2f}초")
```

### 시스템 프롬프트 설정

```python
# 시스템 프롬프트로 페르소나 설정
llm = LLM.create(
    "gpt-4o-mini",
    system_prompt="당신은 친절한 교육 전문가입니다. 쉽고 명확하게 설명해주세요."
)

reply = llm.ask("재귀 함수란 무엇인가요?")
print(reply.text)
```

## 선택지에서 고르기

### 단순 선택

```python
# 감정 분석
reply = llm.ask(
    "다음 리뷰의 감정을 분석해주세요: '이 제품 정말 최고예요!'",
    choices=["긍정", "부정", "중립"]
)

print(f"감정: {reply.choice}")
print(f"확신도: {reply.confidence:.2%}")
```

### 다중 선택 분류

```python
# 카테고리 분류
categories = ["기술", "스포츠", "정치", "경제", "문화", "과학"]

reply = llm.ask(
    "다음 뉴스의 카테고리를 선택하세요: 'OpenAI가 새로운 GPT-5 모델을 발표했습니다.'",
    choices=categories
)

print(f"카테고리: {reply.choice}")
```

## 구조화된 출력

### Pydantic 모델 사용

```python
from pydantic import BaseModel
from typing import List

class ProductInfo(BaseModel):
    name: str
    price: float
    features: List[str]
    in_stock: bool

# 구조화된 데이터 요청
reply = llm.ask(
    "아이폰 15 Pro에 대한 제품 정보를 알려주세요",
    schema=ProductInfo
)

product = reply.structured_data
print(f"제품명: {product.name}")
print(f"가격: ${product.price:,.0f}")
print(f"주요 기능: {', '.join(product.features)}")
print(f"재고: {'있음' if product.in_stock else '없음'}")
```

### 복잡한 스키마

```python
from datetime import datetime
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str
    description: str
    priority: Priority
    due_date: datetime
    tags: List[str]
    estimated_hours: float

reply = llm.ask(
    "다음 프로젝트를 위한 작업을 생성해주세요: AI 챗봇 개발",
    schema=Task
)

task = reply.structured_data
print(f"작업: {task.title}")
print(f"우선순위: {task.priority.value}")
print(f"마감일: {task.due_date.strftime('%Y-%m-%d')}")
```

## 스트리밍 응답

### 실시간 출력

```python
# 스트리밍으로 긴 응답 받기
print("AI 답변: ", end="")
for chunk in llm.ask("머신러닝의 역사를 자세히 설명해주세요", stream=True):
    print(chunk.text, end="", flush=True)
print()  # 줄바꿈
```

### 스트리밍 중 처리

```python
# 단어 수 계산하며 스트리밍
total_words = 0
for chunk in llm.ask("파이썬의 활용 분야를 설명해주세요", stream=True):
    print(chunk.text, end="", flush=True)
    total_words += len(chunk.text.split())

print(f"\n\n총 단어 수: {total_words}")
```

## 이미지와 함께 질문하기

### 단일 이미지 분석

```python
# 이미지 설명
reply = llm.ask(
    "이 이미지에 무엇이 있나요?",
    files=["photo.jpg"]
)
print(reply.text)

# 이미지 기반 질문
reply = llm.ask(
    "이 차트에서 가장 높은 값은 얼마인가요?",
    files=["chart.png"]
)
```

### 여러 이미지 비교

```python
# 이미지 비교
reply = llm.ask(
    "이 두 이미지의 차이점을 찾아주세요",
    files=["before.jpg", "after.jpg"]
)

# 이미지 시퀀스 분석
reply = llm.ask(
    "이 이미지들이 보여주는 과정을 설명해주세요",
    files=["step1.png", "step2.png", "step3.png"]
)
```

## 템플릿 사용

### Jinja2 템플릿

```python
from pyhub.llm.templates import PromptTemplate

# 템플릿 정의
template = PromptTemplate("""
다음 제품에 대한 마케팅 문구를 작성해주세요:

제품명: {{ product_name }}
특징: {{ features | join(", ") }}
타겟: {{ target_audience }}

톤: {{ tone }}
""")

# 템플릿 사용
prompt = template.render(
    product_name="스마트 워치 Pro",
    features=["심박수 측정", "GPS", "방수"],
    target_audience="운동을 좋아하는 20-30대",
    tone="활기차고 동기부여가 되는"
)

reply = llm.ask(prompt)
print(reply.text)
```

### 재사용 가능한 템플릿

```python
# 번역 템플릿
translate_template = PromptTemplate(
    "다음 {{ source_lang }} 문장을 {{ target_lang }}로 번역해주세요:\n\n{{ text }}"
)

# 여러 번역에 재사용
texts = ["Hello, world!", "How are you?", "Nice to meet you."]

for text in texts:
    prompt = translate_template.render(
        source_lang="영어",
        target_lang="한국어",
        text=text
    )
    reply = llm.ask(prompt)
    print(f"{text} → {reply.text}")
```

## 에러 처리

### 기본 에러 처리

```python
from pyhub.llm.exceptions import LLMError, RateLimitError, AuthenticationError

try:
    reply = llm.ask("질문")
except AuthenticationError:
    print("API 키가 유효하지 않습니다")
except RateLimitError:
    print("API 호출 한도를 초과했습니다")
except LLMError as e:
    print(f"LLM 오류 발생: {e}")
```

### 재시도 로직

```python
import time

def ask_with_retry(llm, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm.ask(prompt)
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 지수 백오프
                print(f"재시도 중... {wait_time}초 대기")
                time.sleep(wait_time)
            else:
                raise

# 사용
reply = ask_with_retry(llm, "복잡한 질문...")
```

## 설정 관리

### 환경별 설정

```python
from pyhub.llm.settings import Settings

# 개발 환경
dev_settings = Settings(
    openai_api_key="dev-key",
    log_level="DEBUG",
    cache_enabled=False
)

# 프로덕션 환경
prod_settings = Settings.from_env()  # 환경 변수에서 로드

# 설정 적용
llm = LLM.create("gpt-4o-mini", settings=prod_settings)
```

### 동적 설정 변경

```python
# 온도 조절
creative_llm = LLM.create("gpt-4o-mini", temperature=0.9)  # 창의적
precise_llm = LLM.create("gpt-4o-mini", temperature=0.1)   # 정확한

# 토큰 제한
short_llm = LLM.create("gpt-4o-mini", max_tokens=50)   # 짧은 답변
long_llm = LLM.create("gpt-4o-mini", max_tokens=2000)  # 긴 답변
```

## 모범 사례

### 1. 명확한 프롬프트 작성

```python
# ❌ 모호한 프롬프트
reply = llm.ask("파이썬")

# ✅ 명확한 프롬프트
reply = llm.ask("파이썬 프로그래밍 언어의 주요 특징 5가지를 설명해주세요")
```

### 2. 적절한 모델 선택

```python
# 간단한 작업: 빠르고 저렴한 모델
simple_llm = LLM.create("gpt-4o-mini")

# 복잡한 작업: 고성능 모델
complex_llm = LLM.create("gpt-4o")

# 코드 생성: 코드 특화 모델
code_llm = LLM.create("gpt-4-turbo")
```

### 3. 비용 최적화

```python
# Stateless 모드로 불필요한 컨텍스트 제거
classifier = LLM.create("gpt-4o-mini", stateless=True)

# 캐싱 활용
from pyhub.llm.cache import FileCache

llm = LLM.create("gpt-4o-mini", cache=FileCache())
```

## 다음 단계

- [대화 관리](conversation.md) - 대화 히스토리와 컨텍스트 관리
- [구조화된 출력](structured-output.md) - 복잡한 데이터 구조 처리
- [고급 기능](advanced.md) - 성능 최적화와 고급 패턴