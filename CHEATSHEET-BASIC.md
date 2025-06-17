# pyhub-llm 초급 가이드

pyhub-llm을 처음 사용하는 분들을 위한 기본 가이드입니다. 이 문서에서는 설치부터 기본적인 사용법, 대화 관리, 파일 처리 등 핵심 기능들을 다룹니다.

> 💡 예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.

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
4. [출력 포맷팅](#출력-포맷팅)
5. [대화 관리](#대화-관리)
   - [대화 히스토리 유지](#대화-히스토리-유지)
   - [컨텍스트 윈도우 관리](#컨텍스트-윈도우-관리)
   - [페르소나 기반 대화](#페르소나-기반-대화)
6. [파일 처리](#파일-처리)
   - [이미지 분석](#이미지-분석)
   - [PDF 문서 처리](#pdf-문서-처리)
   - [이미지 생성 프롬프트](#이미지-생성-프롬프트)
7. [에러 처리](#에러-처리)
   - [기본 에러 처리](#기본-에러-처리)
   - [폴백 처리](#폴백-처리)
   - [타임아웃 처리](#타임아웃-처리)
8. [다음 단계](#다음-단계)

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

💻 [실행 가능한 예제](examples/basic/01_hello_world.py)

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

💻 [실행 가능한 예제](examples/basic/02_streaming.py)

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

## 출력 포맷팅

편리한 출력 기능으로 마크다운 렌더링을 지원합니다.

### display() 함수 사용

```python
from pyhub.llm import LLM, display

llm = LLM.create("gpt-4o-mini")

# 스트리밍과 함께 마크다운 렌더링
response = llm.ask("파이썬 함수 예제를 보여주세요", stream=True)
display(response)  # 자동으로 마크다운 렌더링!

# 일반 응답도 마크다운 렌더링
reply = llm.ask("# 제목\n\n**굵은 글씨**로 작성")
display(reply)

# 일반 텍스트로 출력
display(reply, markdown=False)
```

### Response.print() 메서드

```python
# 모든 Response 객체에 print() 메서드 제공
reply = llm.ask("마크다운 표 예제")

# 마크다운 렌더링
reply.print()  # 기본값: markdown=True

# 일반 텍스트
reply.print(markdown=False)

# 스트만도 동일하게 사용 가능
response = llm.ask("코드 예제", stream=True)
response.print()  # 스트리밍하면서 마크다운 렌더링
```

### Rich 라이브러리 설치

마크다운 렌더링을 위해서는 Rich 라이브러리가 필요합니다:

```bash
pip install "pyhub-llm[rich]"
# 또는
pip install rich
```

Rich가 설치되지 않은 경우, 일반 텍스트로 출력됩니다.

## 대화 관리

💻 [실행 가능한 예제](examples/basic/03_conversation.py)

### 대화 히스토리 유지

pyhub-llm은 내부적으로 대화 히스토리를 자동 관리합니다. 별도의 히스토리 관리 없이도 연속적인 대화가 가능합니다.

```python
from pyhub.llm import LLM

# LLM 인스턴스 생성
llm = LLM.create("gpt-4o-mini")

# 자동으로 대화 컨텍스트가 유지됨
print(llm.ask("안녕하세요! 저는 프로그래밍을 배우고 싶어요.").text)
print(llm.ask("파이썬부터 시작하면 좋을까요?").text)
print(llm.ask("그럼 첫 번째로 뭘 배워야 할까요?").text)

# 대화 초기화가 필요한 경우
llm.clear()  # 대화 히스토리 초기화
```


### Stateless 모드 (히스토리 없는 독립 처리)

반복적인 독립 작업에서는 대화 히스토리가 불필요합니다. Stateless 모드를 사용하면 각 요청이 완전히 독립적으로 처리됩니다.

```python
from pyhub.llm import LLM

# Stateless 모드로 생성 (히스토리 저장 안 함)
classifier = LLM.create("gpt-4o-mini", stateless=True)

# 대량의 독립적인 분류 작업
texts = ["환불해주세요", "언제 배송되나요?", "제품이 고장났어요"]
for text in texts:
    reply = classifier.ask(
        f"고객 문의 분류: {text}",
        choices=["환불", "배송", "AS", "기타"]
    )
    print(f"{text} -> {reply.choice}")
    # 각 요청이 독립적으로 처리되어 API 비용 절감
```

### 일반 모드 vs Stateless 모드 비교

| 특징 | 일반 모드 | Stateless 모드 |
|------|-----------|----------------|
| 대화 히스토리 | 자동 저장 | 저장 안 함 |
| 연속 대화 | 가능 | 불가능 |
| 메모리 사용 | 누적됨 | 항상 최소 |
| API 토큰 사용 | 누적됨 | 항상 최소 |
| 사용 사례 | 챗봇, 대화형 AI | 분류, 추출, 번역 |
| `use_history` | 동작함 | 무시됨 |
| `clear()` | 히스토리 삭제 | 아무 동작 안 함 |

### 컨텍스트 윈도우 관리

pyhub-llm은 내부적으로 컨텍스트 윈도우를 관리하지만, 필요에 따라 수동으로 제어할 수도 있습니다.

```python
from pyhub.llm import LLM

# 자동 컨텍스트 관리 (기본값)
llm = LLM.create("gpt-4o-mini")

# 긴 대화 진행 - LLM이 자동으로 컨텍스트 관리
for i in range(20):
    reply = llm.ask(f"질문 {i}: 이전 대화를 기억하나요?")
    print(f"답변 {i}: {reply.text[:50]}...")

# 수동으로 메시지 수 제한하기
if len(llm.history) > 10:
    # 최근 10개 메시지만 유지
    llm.history = llm.history[-10:]
```

### 페르소나 기반 대화

시스템 프롬프트를 설정하여 다양한 페르소나로 대화할 수 있습니다.

```python
from pyhub.llm import LLM

# 다양한 페르소나 설정
teacher = LLM.create(
    "gpt-4o-mini",
    system_prompt="당신은 친절하고 인내심 있는 프로그래밍 교사입니다."
)

chef = LLM.create(
    "gpt-4o-mini",
    system_prompt="당신은 미슐랭 3스타 셰프입니다. 요리에 대한 열정이 넘칩니다."
)

doctor = LLM.create(
    "gpt-4o-mini",
    system_prompt="당신은 의학 전문가입니다. 정확하고 신중하게 답변합니다."
)

# 각 페르소나와 대화 (컨텍스트 자동 유지)
print(teacher.ask("재귀함수가 뭔가요?").text)
print(teacher.ask("예제를 보여주세요.").text)  # 이전 대화 기억

print(chef.ask("파스타 면을 삶는 최적의 시간은?").text)
print(chef.ask("소금은 언제 넣나요?").text)  # 파스타 관련 대화 계속

print(doctor.ask("두통이 자주 있어요").text)  # 주의: 실제 의료 조언이 아님
```

## 파일 처리

💻 [실행 가능한 예제](examples/basic/04_file_processing.py)

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

## 실전 예제: Stateless 모드 활용

### 고객 문의 분류 시스템

```python
from pyhub.llm import LLM
from typing import List, Dict

def classify_customer_inquiries(inquiries: List[str]) -> List[Dict[str, str]]:
    """대량의 고객 문의를 분류"""
    # Stateless 모드로 분류기 생성
    classifier = LLM.create("gpt-4o-mini", stateless=True)
    
    categories = ["환불/반품", "배송문의", "제품문의", "AS요청", "기타"]
    results = []
    
    for inquiry in inquiries:
        reply = classifier.ask(
            f"다음 고객 문의를 분류하세요: {inquiry}",
            choices=categories
        )
        results.append({
            "inquiry": inquiry,
            "category": reply.choice,
            "confidence": reply.confidence
        })
    
    return results

# 사용 예
inquiries = [
    "제품이 파손되어 도착했어요",
    "주문한 지 일주일이 됐는데 아직 안 왔어요",
    "이 제품 사용법을 모르겠어요",
    "환불 처리는 얼마나 걸리나요?"
]

results = classify_customer_inquiries(inquiries)
for r in results:
    print(f"{r['inquiry'][:20]}... -> {r['category']} ({r['confidence']:.2f})")
```

### 대량 텍스트 감정 분석

```python
from pyhub.llm import LLM
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import time

class SentimentResult(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float
    keywords: List[str]

def analyze_sentiment_batch(texts: List[str], batch_size: int = 10):
    """대량의 텍스트 감정 분석 (병렬 처리)"""
    # Stateless 모드로 여러 인스턴스 생성
    analyzers = [
        LLM.create("gpt-4o-mini", stateless=True) 
        for _ in range(batch_size)
    ]
    
    def analyze_single(analyzer_text_pair):
        analyzer, text = analyzer_text_pair
        reply = analyzer.ask(
            f"Analyze sentiment of: {text}",
            schema=SentimentResult
        )
        return {
            "text": text,
            "result": reply.structured_data
        }
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # 각 텍스트를 분석기에 할당
        pairs = [(analyzers[i % batch_size], text) for i, text in enumerate(texts)]
        results = list(executor.map(analyze_single, pairs))
    
    return results

# 사용 예
reviews = [
    "This product is amazing! Best purchase ever.",
    "Terrible experience, would not recommend.",
    "It's okay, nothing special.",
    # ... 수백 개의 리뷰
]

start = time.time()
results = analyze_sentiment_batch(reviews[:50], batch_size=5)
print(f"Analyzed {len(results)} reviews in {time.time() - start:.2f}s")
```

### 문서 요약 배치 처리

```python
from pyhub.llm import LLM
from pathlib import Path
import json

def summarize_documents(doc_folder: str, output_file: str):
    """폴더 내 모든 문서를 요약"""
    # Stateless 모드 - 각 문서가 독립적으로 처리됨
    summarizer = LLM.create("gpt-4o-mini", stateless=True)
    
    summaries = {}
    doc_path = Path(doc_folder)
    
    for file_path in doc_path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 각 문서를 독립적으로 요약
        reply = summarizer.ask(
            f"다음 문서를 3줄로 요약하세요:\n\n{content[:2000]}"
        )
        
        summaries[file_path.name] = {
            "summary": reply.text,
            "file_size": len(content),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"✓ {file_path.name} 처리 완료")
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    
    return summaries

# 사용 예
summaries = summarize_documents("./documents", "./summaries.json")
print(f"총 {len(summaries)}개 문서 요약 완료")
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
