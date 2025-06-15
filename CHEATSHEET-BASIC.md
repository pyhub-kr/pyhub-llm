# pyhub-llm 초급 가이드

pyhub-llm을 처음 사용하는 분들을 위한 기본 가이드입니다. 이 문서에서는 설치부터 기본적인 사용법, 대화 관리, 파일 처리 등 핵심 기능들을 다룹니다.

## 목차

1. [설치](#설치)
2. [기본 사용법](#기본-사용법)
   - [환경변수 설정](#환경변수-설정)
   - [OpenAI](#openai)
   - [Anthropic](#anthropic)
   - [Google](#google)
   - [Ollama (로컬)](#ollama-로컬)
   - [Upstage](#upstage)
3. [스트리밍](#스트리밍)
4. [대화 관리](#대화-관리)
   - [대화 히스토리 유지](#대화-히스토리-유지)
   - [컨텍스트 윈도우 관리](#컨텍스트-윈도우-관리)
   - [페르소나 기반 대화](#페르소나-기반-대화)
5. [파일 처리](#파일-처리)
   - [이미지 분석](#이미지-분석)
   - [PDF 문서 처리](#pdf-문서-처리)
   - [이미지 생성 프롬프트](#이미지-생성-프롬프트)
6. [에러 처리](#에러-처리)
   - [기본 에러 처리](#기본-에러-처리)
   - [폴백 처리](#폴백-처리)
   - [타임아웃 처리](#타임아웃-처리)
7. [다음 단계](#다음-단계)

## 설치

```bash
# 전체 설치 (모든 프로바이더)
pip install "pyhub-llm[all]"

# 특정 프로바이더만 설치
pip install "pyhub-llm[openai]"      # OpenAI만
pip install "pyhub-llm[anthropic]"   # Anthropic만
pip install "pyhub-llm[google]"      # Google만
pip install "pyhub-llm[ollama]"      # Ollama만

# MCP 지원 포함
pip install "pyhub-llm[all,mcp]"
```

## 기본 사용법

### 환경변수 설정

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export UPSTAGE_API_KEY="..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

### OpenAI

```python
from pyhub.llm import LLM, OpenAILLM

# 팩토리 패턴 사용 (권장)
llm = LLM.create("gpt-4o-mini")
reply = llm.ask("안녕하세요!")
print(reply.text)

# 직접 생성
llm = OpenAILLM(model="gpt-4o-mini", temperature=0.7)
reply = llm.ask("파이썬의 장점을 3가지 알려주세요")
print(reply.text)
```

### Anthropic

```python
from pyhub.llm import LLM, AnthropicLLM

# Claude 사용
llm = LLM.create("claude-3-haiku-20240307")
reply = llm.ask("양자 컴퓨터를 쉽게 설명해주세요")
print(reply.text)

# 시스템 프롬프트 설정
llm = AnthropicLLM(
    model="claude-3-5-sonnet-20241022",
    system_prompt="당신은 친절한 교육 전문가입니다."
)
```

### Google

```python
from pyhub.llm import LLM, GoogleLLM

# Gemini 사용
llm = LLM.create("gemini-1.5-flash")
reply = llm.ask("오늘 날씨가 좋네요. 산책하기 좋은 장소를 추천해주세요.")
print(reply.text)

# 긴 컨텍스트 처리
llm = GoogleLLM(model="gemini-1.5-pro", max_tokens=8192)
```

### Ollama (로컬)

```python
from pyhub.llm import LLM, OllamaLLM

# Ollama는 API 키 불필요
llm = LLM.create("mistral")
reply = llm.ask("로컬에서 실행되는 LLM의 장점은?")
print(reply.text)

# 커스텀 서버 주소
llm = OllamaLLM(
    model="llama2",
    base_url="http://192.168.1.100:11434"
)
```

### Upstage

```python
from pyhub.llm import UpstageLLM

llm = UpstageLLM(model="solar-mini")
reply = llm.ask("한국어 자연어 처리의 어려운 점은?")
print(reply.text)
```

## 스트리밍

실시간으로 응답을 받아 처리합니다.

```python
from pyhub.llm import LLM

llm = LLM.create("gpt-4o-mini")

# 기본 스트리밍
for chunk in llm.ask("긴 이야기를 들려주세요", stream=True):
    print(chunk.text, end="", flush=True)
print()

# 스트리밍 중 처리
def process_stream(llm, prompt):
    full_text = ""
    for i, chunk in enumerate(llm.ask(prompt, stream=True)):
        full_text += chunk.text
        # 특정 단어가 나오면 중단
        if "종료" in chunk.text:
            break
        # 진행 상황 표시
        if i % 10 == 0:
            print(".", end="", flush=True)
    return full_text
```

## 대화 관리

### 대화 히스토리 유지

```python
from pyhub.llm import LLM
from pyhub.llm.types import Message

class ChatBot:
    def __init__(self, model="gpt-4o-mini"):
        self.llm = LLM.create(model)
        self.history = []
    
    def chat(self, user_input: str) -> str:
        # 사용자 메시지 추가
        self.history.append(Message(role="user", content=user_input))
        
        # LLM에게 전체 히스토리 전달
        reply = self.llm.messages(self.history)
        
        # 응답을 히스토리에 추가
        self.history.append(Message(role="assistant", content=reply.text))
        
        return reply.text
    
    def reset(self):
        """대화 초기화"""
        self.history = []
    
    def save_history(self, filename: str):
        """대화 내역 저장"""
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([msg.dict() for msg in self.history], f, ensure_ascii=False, indent=2)

# 사용 예
bot = ChatBot()
print(bot.chat("안녕하세요! 저는 프로그래밍을 배우고 싶어요."))
print(bot.chat("파이썬부터 시작하면 좋을까요?"))
print(bot.chat("그럼 첫 번째로 뭘 배워야 할까요?"))
```

### 컨텍스트 윈도우 관리

```python
class ContextManagedChat:
    def __init__(self, model="gpt-4o-mini", max_messages=10):
        self.llm = LLM.create(model)
        self.history = []
        self.max_messages = max_messages
    
    def chat(self, user_input: str) -> str:
        # 컨텍스트 윈도우 크기 제한
        if len(self.history) >= self.max_messages * 2:
            # 시스템 메시지는 유지하고 오래된 대화 제거
            self.history = self.history[-self.max_messages * 2:]
        
        self.history.append(Message(role="user", content=user_input))
        reply = self.llm.messages(self.history)
        self.history.append(Message(role="assistant", content=reply.text))
        
        return reply.text
```

### 페르소나 기반 대화

```python
class PersonaChat:
    def __init__(self, persona: str, model="gpt-4o-mini"):
        self.llm = LLM.create(
            model,
            system_prompt=persona
        )
        self.history = []
    
    def chat(self, message: str) -> str:
        reply = self.llm.ask(message)
        return reply.text

# 다양한 페르소나
teacher = PersonaChat("당신은 친절하고 인내심 있는 프로그래밍 교사입니다.")
chef = PersonaChat("당신은 미슐랭 3스타 셰프입니다. 요리에 대한 열정이 넘칩니다.")
doctor = PersonaChat("당신은 의학 전문가입니다. 정확하고 신중하게 답변합니다.")

print(teacher.chat("재귀함수가 뭔가요?"))
print(chef.chat("파스타 면을 삶는 최적의 시간은?"))
print(doctor.chat("두통이 자주 있어요"))  # 주의: 실제 의료 조언이 아님
```

## 파일 처리

### 이미지 분석

```python
from pyhub.llm import LLM

llm = LLM.create("gpt-4o-mini")  # 비전 지원 모델

# 단일 이미지 분석
reply = llm.ask(
    "이 이미지에 무엇이 보이나요?",
    files=["photo.jpg"]
)
print(reply.text)

# 여러 이미지 비교
reply = llm.ask(
    "이 두 이미지의 차이점을 설명해주세요",
    files=["before.png", "after.png"]
)
print(reply.text)

# 이미지와 함께 구조화된 출력
from pydantic import BaseModel
from typing import List

class ImageAnalysis(BaseModel):
    objects: List[str] = Field(description="이미지에서 발견된 객체들")
    scene: str = Field(description="전체적인 장면 설명")
    colors: List[str] = Field(description="주요 색상들")
    mood: str = Field(description="이미지의 분위기")

reply = llm.ask(
    "이 이미지를 분석해주세요",
    files=["landscape.jpg"],
    schema=ImageAnalysis
)
analysis = reply.structured_data
print(f"발견된 객체: {', '.join(analysis.objects)}")
print(f"분위기: {analysis.mood}")
```

### PDF 문서 처리

```python
# PDF는 자동으로 이미지로 변환됨
llm = LLM.create("gpt-4o-mini")

# PDF 요약
reply = llm.ask(
    "이 PDF 문서의 주요 내용을 요약해주세요",
    files=["report.pdf"]
)
print(reply.text)

# 여러 페이지 PDF 처리
class PDFSummary(BaseModel):
    title: str
    main_topics: List[str]
    key_findings: List[str]
    recommendations: List[str]

reply = llm.ask(
    "이 연구 보고서를 분석해주세요",
    files=["research_paper.pdf"],
    schema=PDFSummary
)
```

### 이미지 생성 프롬프트

```python
class ImagePrompt(BaseModel):
    style: str = Field(description="그림 스타일")
    subject: str = Field(description="주요 대상")
    background: str = Field(description="배경 설명")
    mood: str = Field(description="분위기")
    details: List[str] = Field(description="추가 세부사항")

llm = LLM.create("gpt-4o-mini")

# 이미지를 보고 유사한 이미지 생성을 위한 프롬프트 생성
reply = llm.ask(
    "이 이미지와 유사한 이미지를 생성하기 위한 프롬프트를 만들어주세요",
    files=["reference_image.jpg"],
    schema=ImagePrompt
)

prompt_data = reply.structured_data
print(f"스타일: {prompt_data.style}")
print(f"프롬프트: A {prompt_data.style} image of {prompt_data.subject} with {prompt_data.background}")
```

## 에러 처리

### 기본 에러 처리

```python
from pyhub.llm import LLM
from pyhub.llm.exceptions import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError
)

def safe_llm_call(prompt: str, max_retries: int = 3):
    """재시도 로직이 포함된 안전한 LLM 호출"""
    llm = LLM.create("gpt-4o-mini")
    
    for attempt in range(max_retries):
        try:
            return llm.ask(prompt)
        
        except RateLimitError as e:
            # 속도 제한 에러 - 대기 후 재시도
            wait_time = getattr(e, 'retry_after', 60)
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            
        except AuthenticationError:
            # 인증 에러 - 재시도 불가
            print("Authentication failed. Check your API key.")
            raise
            
        except InvalidRequestError as e:
            # 잘못된 요청 - 수정 필요
            print(f"Invalid request: {e}")
            raise
            
        except LLMError as e:
            # 기타 LLM 에러
            print(f"LLM error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 지수 백오프
    
    raise Exception("Max retries exceeded")
```

### 폴백 처리

```python
class LLMWithFallback:
    """폴백 LLM이 있는 래퍼"""
    
    def __init__(self, primary_model: str, fallback_model: str):
        self.primary = LLM.create(primary_model)
        self.fallback = LLM.create(fallback_model)
    
    def ask(self, prompt: str, **kwargs):
        try:
            return self.primary.ask(prompt, **kwargs)
        except Exception as e:
            print(f"Primary LLM failed: {e}. Using fallback...")
            return self.fallback.ask(prompt, **kwargs)

# 사용
llm = LLMWithFallback("gpt-4o", "gpt-4o-mini")
reply = llm.ask("복잡한 수학 문제...")
```

### 타임아웃 처리

```python
import asyncio
from asyncio import TimeoutError

async def ask_with_timeout(llm, prompt: str, timeout: float = 30.0):
    """타임아웃이 있는 LLM 호출"""
    try:
        return await asyncio.wait_for(
            llm.ask_async(prompt),
            timeout=timeout
        )
    except TimeoutError:
        print(f"Request timed out after {timeout} seconds")
        # 더 간단한 프롬프트로 재시도
        simplified = f"간단히 답해주세요: {prompt}"
        return await llm.ask_async(simplified)

# 사용
llm = LLM.create("gpt-4o-mini")
result = asyncio.run(ask_with_timeout(llm, "매우 복잡한 질문...", timeout=10))
```

## 다음 단계

축하합니다! pyhub-llm의 기본 기능들을 모두 살펴보았습니다. 이제 더 고급 기능들을 배워볼 준비가 되었습니다:

### 중급 가이드에서 다룰 내용
- **구조화된 출력**: Pydantic 스키마를 활용한 복잡한 응답 처리
- **비동기 처리**: 동시에 여러 요청 처리하기
- **고급 캐싱**: 성능 최적화를 위한 캐싱 전략
- **임베딩과 벡터 검색**: 텍스트 유사도 분석
- **체인과 파이프라인**: 복잡한 작업 흐름 구성
- **MCP (Model Context Protocol)**: 외부 도구와 통합

### 추가 리소스
- [전체 치트시트 (CHEATSHEET.md)](./CHEATSHEET.md) - 모든 기능의 상세한 예제
- [공식 문서](https://github.com/pyhub-kr/pyhub-llm) - 최신 업데이트 및 API 문서
- [예제 코드](https://github.com/pyhub-kr/pyhub-llm/tree/main/examples) - 실제 사용 사례

### 도움이 필요하신가요?
- GitHub Issues에서 질문하기
- 커뮤니티 포럼 참여하기
- 기여하기: 버그 리포트, 기능 제안, 문서 개선

즐거운 코딩 되세요! 🚀
