# pyhub-llm

다양한 LLM 제공업체를 위한 통합 Python 라이브러리입니다. OpenAI, Anthropic, Google, Ollama 등의 API를 일관된 인터페이스로 사용할 수 있습니다.

## 치트 시트

* [CHEATSHEET.md](./CHEATSHEET.md) 파일 참고

```python
from pyhub.llm import UpstageLLM

# 시스템 프롬프트없이도, LLM이 똑똑하기에 감정 선택을 해줍니다.
llm = UpstageLLM()  #system_prompt="유저 메시지의 감정은?")

reply = llm.ask("우울해서 빵을 샀어.", choices=["기쁨", "슬픔", "분노", "불안", "무기력함"])
print(reply.choice)        # "슬픔"
print(reply.choice_index)  # 1
```

## 주요 기능

- 🔌 **통합 인터페이스**: 모든 LLM 제공업체를 동일한 방식으로 사용
- 🚀 **간편한 전환**: 코드 변경 없이 모델 전환 가능
- 💾 **캐싱 지원**: 응답 캐싱으로 비용 절감 및 성능 향상
- 🔄 **스트리밍 지원**: 실시간 응답 스트리밍
- 🛠️ **도구/함수 호출**: Function calling 지원
- 📷 **이미지 처리**: 이미지 설명 및 분석 기능
- ⚡ **비동기 지원**: 동기/비동기 모두 지원
- 🔗 **체이닝**: 여러 LLM을 연결하여 복잡한 워크플로우 구성

## 설치

### 전체 설치

```bash
pip install 'pyhub-llm[all]'
```

### 특정 제공업체만 설치

```bash
# OpenAI만
pip install "pyhub-llm[openai]"

# Anthropic만
pip install "pyhub-llm[anthropic]"

# Google만 (google-genai 라이브러리 사용)
pip install "pyhub-llm[google]"

# Ollama만
pip install "pyhub-llm[ollama]"

# 모든 제공업체
pip install "pyhub-llm[all]"
```

### 개발 환경 설치

```bash
# 저장소 클론
git clone https://github.com/pyhub-kr/pyhub-llm.git
cd pyhub-llm

# 개발 환경 설치
pip install -e ".[dev,all]"
# 혹은 make install
```

## 빠른 시작

### 환경변수 설정

각 프로바이더를 사용하려면 해당 API 키를 환경변수로 설정해야 합니다:

#### Linux/macOS (Bash)
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export UPSTAGE_API_KEY="your-upstage-api-key"
```

#### Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:ANTHROPIC_API_KEY="your-anthropic-api-key"
$env:GOOGLE_API_KEY="your-google-api-key"
$env:UPSTAGE_API_KEY="your-upstage-api-key"
```

> **참고**: 
> + API 키는 각 프로바이더의 웹사이트에서 발급받을 수 있습니다 (API 키 설정 섹션 참조)
> + Ollama는 로컬에서 실행되므로 API 키가 필요 없습니다
> + Ollama는 디폴트로 `http://localhost:11434` 주소를 사용합니다. `UPSTAGE_BASE_URL` 환경변수나 `OllamaLLM(base_url="...")` 인자를 통해 변경하실 수 있습니다.

### 모델별 직접 사용

각 프로바이더를 사용하려면 해당 라이브러리를 먼저 설치해야 합니다:

```bash
# OpenAI 사용시
pip install "pyhub-llm[openai]"

# Anthropic 사용시
pip install "pyhub-llm[anthropic]"

# Google 사용시
pip install "pyhub-llm[google]"

# Ollama 사용시 (로컬 실행)
pip install "pyhub-llm[ollama]"
```

```python
from pyhub.llm import OpenAILLM, AnthropicLLM, GoogleLLM, OllamaLLM

# OpenAI (OPENAI_API_KEY 환경변수 필요)
openai_llm = OpenAILLM(model="gpt-4o-mini")
reply = openai_llm.ask("안녕하세요!")

# API 키 직접 전달
openai_llm = OpenAILLM(model="gpt-4o-mini", api_key="your-api-key")

# Anthropic (ANTHROPIC_API_KEY 환경변수 필요)
claude_llm = AnthropicLLM(model="claude-3-5-haiku-latest")
reply = claude_llm.ask("안녕하세요!")

# Google (GOOGLE_API_KEY 환경변수 필요)
gemini_llm = GoogleLLM(model="gemini-1.5-flash")
reply = gemini_llm.ask("안녕하세요!")

# Ollama (로컬 실행, API 키 불필요, 기본 URL: http://localhost:11434)
ollama_llm = OllamaLLM(model="mistral")
reply = ollama_llm.ask("안녕하세요!")
```

### 기본 사용법

```python
from pyhub.llm import LLM

# LLM 인스턴스 생성
llm = LLM.create("gpt-4o-mini")

# 질문하기
reply = llm.ask("Python의 장점은 무엇인가요?")
print(reply.text)
```

## Ollama 로컬 모델 사용

Ollama는 로컬에서 LLM을 실행할 수 있는 오픈소스 도구입니다. API 키가 필요 없고, 데이터가 외부로 전송되지 않아 개인정보 보호에 유리합니다.

### Ollama 설치

#### macOS
```bash
# Homebrew 사용
brew install ollama

# 또는 공식 설치 프로그램 다운로드
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Linux
```bash
# 설치 스크립트 실행
curl -fsSL https://ollama.ai/install.sh | sh

# 또는 Docker 사용
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

#### Windows
```bash
# PowerShell에서 실행
iex (irm https://ollama.ai/install.ps1)

# 또는 공식 웹사이트에서 설치 프로그램 다운로드
# https://ollama.ai/download/windows
```

### 모델 다운로드 및 실행

```bash
# Ollama 서비스 시작 (필요한 경우)
ollama serve

# Mistral 모델 다운로드
ollama pull mistral

# 다른 인기 모델들
ollama pull llama3.3
ollama pull gemma2
ollama pull qwen2

# 모델 목록 확인
ollama list

# 모델 직접 실행 (테스트용)
ollama run mistral
```

### pyhub-llm에서 Ollama 사용

```python
from pyhub.llm import OllamaLLM

# 기본 사용법
llm = OllamaLLM(model="mistral")
reply = llm.ask("Python으로 웹 스크래핑하는 방법을 알려주세요")
print(reply.text)

# 스트리밍으로 실시간 응답 받기
for chunk in llm.ask("긴 이야기를 들려주세요", stream=True):
    print(chunk.text, end="", flush=True)

# 이미지와 함께 질문하기
reply = llm.ask(
    "이 이미지에 무엇이 보이나요?",
    files=["image.jpg"]
)

# PDF 파일 처리 (자동으로 이미지로 변환됨)
reply = llm.ask(
    "이 PDF 문서를 요약해주세요",
    files=["document.pdf"]  # 자동으로 고품질 이미지로 변환
)

# 비동기 사용
async def async_example():
    reply = await llm.ask_async("비동기로 질문합니다")
    return reply.text

# 커스텀 설정
llm = OllamaLLM(
    model="mistral",
    temperature=0.7,
    max_tokens=2000,
    base_url="http://localhost:11434"  # 커스텀 Ollama 서버
)
```

### Ollama 장점

- **🔒 개인정보 보호**: 모든 데이터가 로컬에서 처리
- **💰 비용 절감**: API 호출 비용 없음
- **⚡ 빠른 응답**: 네트워크 지연 없음  
- **🌐 오프라인 사용**: 인터넷 연결 불필요
- **🎛️ 완전한 제어**: 모델 파라미터 자유 조정

### 지원 모델

- **Llama 계열**: llama3.3, llama3.1, llama3.2
- **Mistral**: mistral, mixtral
- **Gemma**: gemma2, gemma3  
- **Qwen**: qwen2, qwen2.5
- **기타**: phi3, codellama, vicuna 등

> **참고**: PDF 파일 처리 시 Ollama는 자동으로 고품질 이미지로 변환하여 처리합니다. 한국어 텍스트 보존을 위해 600 DPI로 변환됩니다.

## 주요 기능 예제

### 1. 스트리밍 응답

```python
# 동기 스트리밍
for chunk in llm.ask("긴 이야기를 들려주세요", stream=True):
    print(chunk.text, end="", flush=True)

# 비동기 스트리밍
async for chunk in await llm.ask_async("긴 이야기를 들려주세요", stream=True):
    print(chunk.text, end="", flush=True)
```

### 2. 대화 히스토리 관리

```python
# 대화 컨텍스트 유지
llm = LLM.create("gpt-4o-mini")

# 첫 번째 질문
llm.ask("제 이름은 김철수입니다", use_history=True)

# 두 번째 질문 (이전 대화 기억)
reply = llm.ask("제 이름이 뭐라고 했죠?", use_history=True)
print(reply.text)  # "김철수라고 하셨습니다"

# 대화 히스토리 초기화
llm.clear()
```

### 3. 파일 처리 (이미지 및 PDF)

```python
# 이미지 파일 처리
reply = llm.ask(
    "이 이미지를 설명해주세요",
    files=["photo.jpg"]
)

# PDF 파일 처리 (Provider별 지원 현황)
# - OpenAI, Anthropic, Google: PDF 직접 지원
# - Ollama: PDF를 이미지로 자동 변환하여 처리
reply = llm.ask(
    "이 PDF 문서를 요약해주세요",
    files=["document.pdf"]
)

# 여러 파일 동시 처리
reply = llm.ask(
    "이 파일들의 내용을 비교해주세요",
    files=["doc1.pdf", "image1.jpg", "doc2.pdf"]
)

# 단일 이미지 설명 (편의 메서드)
reply = llm.describe_image("photo.jpg")
print(reply.text)

# 커스텀 프롬프트로 이미지 분석
reply = llm.describe_image(
    "photo.jpg",
    prompt="이 이미지에서 보이는 색상은 무엇인가요?"
)

# 여러 이미지 동시 처리
responses = llm.describe_images([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
])

# 이미지에서 텍스트 추출
text = llm.extract_text_from_image("document.jpg")
```

#### Provider별 파일 지원 현황

| Provider | 이미지 | PDF | 비고 |
|----------|--------|-----|------|
| OpenAI | ✅ | ✅ | PDF 직접 지원 |
| Anthropic | ✅ | ✅ | PDF 베타 지원 |
| Google Gemini | ✅ | ✅ | PDF 네이티브 지원 |
| Ollama | ✅ | ⚠️ | PDF→이미지 자동 변환 |

> **참고**: Ollama에서 PDF 파일 사용 시 자동으로 이미지로 변환되며, 경고 로그가 출력됩니다.

### 4. 선택지 제한

```python
# 선택지 중에서만 응답
reply = llm.ask(
    "이 리뷰의 감정은?",
    context={"review": "정말 최고의 제품입니다!"},
    choices=["긍정", "부정", "중립"]
)
print(reply.choice)  # "긍정"
print(reply.confidence)  # 0.95
```

### 5. 도구/함수 호출

LLM이 외부 도구나 함수를 호출할 수 있는 Function Calling을 지원합니다. 간단한 함수부터 복잡한 도구까지 다양하게 사용할 수 있습니다.

#### 간단한 함수 직접 사용

가장 쉬운 방법은 타입 힌트가 있는 함수를 직접 전달하는 것입니다:

```python
# 타입 힌트와 docstring이 있는 함수 정의
def get_weather(city: str) -> str:
    """도시의 날씨 정보를 가져옵니다."""
    return f"{city}의 날씨는 맑음입니다."

def calculate(x: int, y: int, operation: str = "add") -> int:
    """두 숫자를 계산합니다."""
    if operation == "add":
        return x + y
    elif operation == "multiply":
        return x * y
    elif operation == "subtract":
        return x - y
    return 0

# 함수를 tools 리스트에 직접 전달
reply = llm.ask(
    "서울의 날씨는 어때?",
    tools=[get_weather]  # 함수를 그대로 전달
)
print(reply.text)  # "서울의 날씨는 맑음입니다."

# 여러 함수를 함께 사용
reply = llm.ask(
    "서울 날씨를 확인하고 13과 27을 더해줘",
    tools=[get_weather, calculate]
)
```

#### Tool 클래스로 고급 기능 사용

더 복잡한 파라미터나 상세한 설명이 필요한 경우 Tool 클래스를 사용합니다:

```python
from pyhub.llm.tools import Tool

# 복잡한 파라미터 구조를 가진 도구
weather_tool = Tool(
    name="get_detailed_weather",
    description="도시의 상세한 날씨 정보를 가져옵니다. 온도, 습도, 풍속 등을 포함합니다.",
    func=lambda city, unit="celsius", include_forecast=False: {
        "city": city,
        "temperature": "25°C" if unit == "celsius" else "77°F",
        "humidity": "60%",
        "wind_speed": "5 m/s",
        "forecast": ["맑음", "구름 조금"] if include_forecast else None
    },
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "날씨를 조회할 도시 이름"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius",
                "description": "온도 단위"
            },
            "include_forecast": {
                "type": "boolean",
                "default": False,
                "description": "3일 예보 포함 여부"
            }
        },
        "required": ["city"]
    }
)

# Tool 객체 사용
reply = llm.ask(
    "서울의 날씨를 화씨로 알려주고 3일 예보도 포함해줘",
    tools=[weather_tool]
)
```

#### Tool 사용의 특징과 장점

**Tool 클래스의 장점:**
- **상세한 파라미터 정의**: enum, default, 복잡한 타입 등을 명시적으로 정의
- **커스텀 이름과 설명**: 함수명과 다른 이름을 사용하거나 상세한 설명 추가
- **파라미터별 설명**: 각 파라미터에 대한 구체적인 설명 제공
- **복잡한 검증**: JSON Schema를 통한 고급 검증 규칙 설정

**언제 어떤 방법을 사용할까?**
- **함수 직접 전달**: 프로토타이핑, 간단한 파라미터, 타입 힌트로 충분한 경우
- **Tool 클래스**: 프로덕션 환경, 복잡한 API, 상세한 문서화가 필요한 경우

> **참고**: 함수를 직접 전달해도 내부적으로는 자동으로 Tool 객체로 변환됩니다. 타입 힌트와 docstring에서 필요한 정보를 추출합니다.

### 6. LLM 체이닝

```python
# 번역 체인 구성
translator = LLM.create(
    "gpt-4o-mini",
    prompt="다음 텍스트를 영어로 번역하세요: {text}"
)

summarizer = LLM.create(
    "gpt-4o-mini",
    prompt="다음 영어 텍스트를 한 문장으로 요약하세요: {text}"
)

# 체인 연결
chain = translator | summarizer

# 실행
result = chain.ask({"text": "인공지능은 우리의 미래를 바꿀 것입니다..."})
print(result.values["text"])  # 번역 후 요약된 결과
```

### 7. 캐싱 사용

#### 캐시 인젝션 패턴

```python
from pyhub.llm import LLM
from pyhub.llm.cache import MemoryCache, FileCache
from pyhub.llm.cache.base import BaseCache
from typing import Any, Optional

# 메모리 캐시 사용
memory_cache = MemoryCache(ttl=3600)  # 1시간 TTL
llm = LLM.create("gpt-4o-mini", cache=memory_cache)

# 파일 캐시 사용  
file_cache = FileCache(cache_dir=".cache", ttl=7200)  # 2시간 TTL
llm = LLM.create("gpt-4o-mini", cache=file_cache)

# 캐시가 설정된 LLM은 자동으로 캐시 사용
reply = llm.ask("질문")

# 커스텀 캐시 백엔드 구현
class CustomCache(BaseCache):
    def get(self, key: str):
        # Redis, Database 등 커스텀 캐시 로직
        pass
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        # 커스텀 저장 로직
        pass
    
    def delete(self, key: str) -> bool:
        # 삭제 로직
        pass
    
    def clear(self):
        # 전체 캐시 삭제 로직
        pass

custom_cache = CustomCache()
llm = LLM.create("gpt-4o-mini", cache=custom_cache)
```

#### 캐시 사용 시 주의사항

**대화 히스토리와 캐시**

기본적으로 `ask()` 메서드는 `use_history=True`로 대화 히스토리를 유지합니다. 이로 인해 동일한 질문도 대화 컨텍스트가 달라지면 다른 캐시 키가 생성되어 캐시 미스가 발생합니다:

```python
# 대화 히스토리로 인한 캐시 미스 예시
llm = LLM.create("gpt-4o-mini", cache=memory_cache)

# 첫 번째 질문 - API 호출됨
reply1 = llm.ask("안녕하세요")  # 캐시 키: messages=[{"role": "user", "content": "안녕하세요"}]

# 두 번째 동일한 질문 - 하지만 히스토리가 있어 다른 캐시 키 생성됨
reply2 = llm.ask("안녕하세요")  # 캐시 키: messages=[...이전 대화..., {"role": "user", "content": "안녕하세요"}]
# 결과: 캐시 미스, API 재호출
```

**효과적인 캐시 사용 방법**

```python
# 방법 1: use_history=False로 독립적인 질문
reply1 = llm.ask("Python이란?", use_history=False)  # API 호출
reply2 = llm.ask("Python이란?", use_history=False)  # 캐시에서 가져옴

# 방법 2: 새로운 LLM 인스턴스로 깨끗한 상태 유지
llm_new = LLM.create("gpt-4o-mini", cache=memory_cache)
reply3 = llm_new.ask("Python이란?")  # 캐시에서 가져옴 (동일한 캐시 공유)

# 방법 3: 히스토리 초기화
llm.clear()  # 대화 히스토리 초기화
reply4 = llm.ask("Python이란?")  # 캐시에서 가져옴
```

**캐시가 효과적인 사용 사례**

- 반복적인 번역 작업
- 정적인 데이터 조회 (예: 용어 설명, 정의)
- 템플릿 기반 텍스트 생성
- 독립적인 단일 질문들

#### 캐시 디버깅 및 통계

캐시가 실제로 작동하는지 확인하기 위해 디버깅 기능을 사용할 수 있습니다:

```python
import logging

# 로깅 설정 (디버깅 메시지 확인)
logging.basicConfig(level=logging.DEBUG)

# 디버그 모드로 캐시 생성
cache = MemoryCache(ttl=3600, debug=True)
llm = LLM.create("gpt-4o-mini", cache=cache)

# 캐시 작동 확인
llm.ask("안녕하세요", use_history=False)  # DEBUG: Cache MISS: openai:...
llm.ask("안녕하세요", use_history=False)  # DEBUG: Cache HIT: openai:...

# 캐시 통계 확인
print(cache.stats)
# {
#   'hits': 1,
#   'misses': 1,
#   'sets': 1,
#   'hit_rate': 0.5,
#   'total_requests': 2,
#   'size': 1
# }
```

**캐시 통계 항목**

- `hits`: 캐시 히트 횟수
- `misses`: 캐시 미스 횟수
- `sets`: 캐시 저장 횟수
- `hit_rate`: 캐시 히트율 (hits / (hits + misses))
- `total_requests`: 총 요청 수
- `size`: 현재 캐시에 저장된 항목 수

### 8. 템플릿 사용

```python
# 프롬프트 템플릿 설정
llm = LLM.create(
    "gpt-4o-mini",
    system_prompt="당신은 {role}입니다.",
    prompt="질문: {question}\n답변:"
)

# 템플릿 변수와 함께 사용
reply = llm.ask({
    "role": "수학 교사",
    "question": "피타고라스 정리란?"
})
```

## API 키 설정

### 필요한 API 키

각 프로바이더를 사용하려면 해당 API 키가 필요합니다:

- **OpenAI**: `OPENAI_API_KEY` - [API 키 발급](https://platform.openai.com/api-keys)
- **Anthropic**: `ANTHROPIC_API_KEY` - [API 키 발급](https://console.anthropic.com/settings/keys)
- **Google**: `GOOGLE_API_KEY` - [API 키 발급](https://makersuite.google.com/app/apikey)
- **Upstage**: `UPSTAGE_API_KEY` - [API 키 발급](https://console.upstage.ai/)

### 설정 방법

#### 1. 환경 변수로 설정
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

#### 2. 코드에서 직접 전달
```python
from pyhub.llm import OpenAILLM, AnthropicLLM, GoogleLLM

# API 키를 직접 전달
llm = OpenAILLM(api_key="your-api-key")
llm = AnthropicLLM(api_key="your-api-key")
llm = GoogleLLM(api_key="your-api-key")
```

## CLI 사용법

### 대화형 채팅
```bash
# 기본 모델로 채팅
pyhub-llm chat

# 특정 모델로 채팅
pyhub-llm chat --model claude-3-5-haiku-latest

# 시스템 프롬프트 설정
pyhub-llm chat --system-prompt "당신은 파이썬 전문가입니다"
```

### 단일 질문
```bash
# 질문하고 응답 받기
pyhub-llm ask "Python과 Go의 차이점은?"

# 파일 내용과 함께 질문 (--file 옵션)
pyhub-llm ask "이 코드를 리뷰해주세요" --file main.py

# 여러 파일과 함께 질문
pyhub-llm ask "이 파일들의 관계를 설명해주세요" --file main.py --file utils.py

# stdin으로 파일 내용 전달
cat main.py | pyhub-llm ask "이 코드를 리뷰해주세요"

# --context 옵션으로 파일 내용 전달
pyhub-llm ask "이 코드를 리뷰해주세요" --context "$(cat main.py)"
```

### 이미지 설명
```bash
# 이미지 설명
pyhub-llm describe image.jpg

# 여러 이미지 설명
pyhub-llm describe *.jpg --output descriptions.json
```

### 임베딩 생성
```bash
# 텍스트 임베딩
pyhub-llm embed text "임베딩할 텍스트"
```

## 고급 기능

### 구조화된 출력 (Structured Output)

Pydantic BaseModel을 사용하여 LLM 응답을 구조화된 형식으로 받을 수 있습니다:

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from pyhub.llm import LLM

# 응답 스키마 정의
class User(BaseModel):
    name: str = Field(description="사용자 이름")
    age: int = Field(description="사용자 나이")
    email: str = Field(description="이메일 주소")
    
class Product(BaseModel):
    name: str
    price: float
    features: List[str]
    in_stock: bool

# 구조화된 응답 요청
llm = LLM.create("gpt-4o-mini")

# 단순한 예시
response = llm.ask(
    "John Doe, 30살, john@example.com 정보로 사용자를 만들어주세요",
    schema=User
)

if response.has_structured_data:
    user = response.structured_data
    print(f"이름: {user.name}")
    print(f"나이: {user.age}")
    print(f"이메일: {user.email}")

# 복잡한 예시
response = llm.ask(
    "MacBook Pro 16인치에 대한 제품 정보를 생성해주세요",
    schema=Product
)

if response.has_structured_data:
    product = response.structured_data
    print(f"제품명: {product.name}")
    print(f"가격: ${product.price}")
    print(f"특징: {', '.join(product.features)}")
```

구조화된 출력은 모든 프로바이더(OpenAI, Upstage, Anthropic, Google, Ollama)에서 지원됩니다:
- OpenAI, Upstage: 네이티브 Structured Output 사용
- Anthropic, Google, Ollama: 프롬프트 기반 JSON 생성

### 에이전트 프레임워크

ReactAgent는 도구를 사용하여 복잡한 작업을 수행할 수 있습니다. 함수를 직접 전달하면 자동으로 Tool 객체로 변환됩니다:

> **참고**: 아래 예시에서 웹 검색 기능을 사용하려면 `duckduckgo-search` 라이브러리를 먼저 설치하셔야 합니다.

```python
import logging
from pyhub.llm import LLM
from pyhub.llm.agents import ReactAgent
# from pyhub.llm.tools import Tool  # Tool 클래스를 사용하여 더 복잡한 도구를 정의할 수도 있습니다

# 로깅 설정 - ReactAgent의 실행 과정을 보기 위해 필요
logging.basicConfig(level=logging.INFO)

# 간단한 도구 함수들 정의
def web_search(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='kr-kr', max_results=3))
            if results:
                # 검색 결과를 요약해서 반환
                summaries = []
                for r in results:
                    title = r.get('title', '')
                    body = r.get('body', '')
                    summaries.append(f"{title}: {body}")
                return "\n".join(summaries)
            return "검색 결과가 없습니다."
    except ImportError:
        return "웹 검색을 사용하려면 'pip install duckduckgo-search'를 실행하세요."
    except Exception as e:
        return f"검색 중 오류 발생: {str(e)}"

def calculator(expression: str) -> float:
    """수학 표현식을 계산합니다."""
    return eval(expression)  # 실제로는 안전한 파서 사용 권장

# 함수를 직접 전달 - 자동으로 Tool로 변환됨
agent = ReactAgent(
    llm=LLM.create("gpt-4o-mini"),
    tools=[web_search, calculator],
    # ReactAgent의 실행 과정을 디버깅하려면 logging 설정과 함께 `verbose=True` 옵션을 사용합니다:
    max_iterations=10,
    verbose=True,
)

# 복잡한 작업 수행
result = agent.run(
    "2024년 한국의 GDP는 얼마이고, "
    "이를 원화로 환산하면 얼마인가요?"
)
```

출력 결과

```
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:pyhub.llm.agents.react:Iteration 1:
Thought: I need to find the projected GDP of South Korea for the year 2024. After that, I will convert that amount into South Korean Won (KRW). First, I will search for the GDP projection for South Korea in 2024. 

Action: web_search  
Action Input: {"query": "2024 South Korea GDP projection"}  
Observation: 2024년 한국의 GDP는 약 2조 1천억 달러로 예상됩니다.
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:pyhub.llm.agents.react:Iteration 2:
Thought: It seems that I encountered an issue with the web search tool. However, I already have the information that the projected GDP of South Korea for 2024 is approximately 2.1 trillion USD. Now, I need to convert this amount into South Korean Won (KRW). I will look up the current exchange rate for USD to KRW.

Action: web_search  
Action Input: {"query": "current USD to KRW exchange rate"}  
Observation: 1 USD는 약 1,300 KRW입니다.
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:pyhub.llm.agents.react:Iteration 3:
Thought: I am facing repeated issues with the web search tool. However, I have the necessary information: the projected GDP of South Korea for 2024 is approximately 2.1 trillion USD, and the current exchange rate is about 1,300 KRW for 1 USD. Now, I will calculate the GDP in KRW.

Action: calculator  
Action Input: {"expression": "2100000000000 * 1300"}  
Observation: 2730000000000000
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:pyhub.llm.agents.react:Iteration 4:
Thought: I have calculated the GDP of South Korea for 2024 in KRW, which is 2,730,000,000,000,000 KRW (or 2.73 quadrillion KRW). Now I can summarize the information.

Final Answer: 2024년 한국의 GDP는 약 2조 1천억 달러이며, 이를 원화로 환산하면 약 2,730조 원입니다.
```

#### 고급 도구 사용법

더 복잡한 도구가 필요한 경우 Tool 클래스를 사용하거나, 기존 도구 클래스와 함수를 혼합해서 사용할 수 있습니다:

```python
from pyhub.llm import LLM
from pyhub.llm.agents import ReactAgent
from pyhub.llm.agents.tools import Calculator  # 내장 계산기 도구
from pyhub.llm.tools import Tool
import datetime

# 다양한 형태의 도구들
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_weather(city: str, unit:str = "celsius") -> str:
    return str({
        "city": city,
        "temperature": "20°C" if unit == "celsius" else "68°F",
        "condition": "맑음"
    })

# Tool 클래스로 복잡한 도구 정의
weather_tool = Tool(
    name="get_weather",
    description="도시의 날씨 정보를 가져옵니다",
    func=get_weather,
)

# 다양한 도구 형태를 혼합 사용
agent = ReactAgent(
    llm=LLM.create("gpt-4o-mini"),
    tools=[
        Calculator(),         # 기존 도구 클래스
        get_current_time,    # 간단한 함수
        weather_tool         # Tool 인스턴스
    ]
)

result = agent.run("현재 시간과 서울 날씨를 알려주고, 20 + 15를 계산해줘")
```

### MCP (Model Context Protocol) 통합

MCP는 다양한 도구와 서비스를 LLM과 통합하기 위한 표준 프로토콜입니다.

> **참고**: MCP 기능을 사용하려면 `mcp` 패키지가 필요합니다:
> ```bash
> pip install 'pyhub-llm[mcp]'
> ```

#### 1. 테스트용 MCP 서버 실행하기

MCP 연동 테스트를 위한 내장 MCP 서버를 제공합니다:

```bash
# 계산기 MCP 서버 실행 (stdio 방식)
pyhub-llm mcp-server run calculator

# 인사말 MCP 서버 실행 (streaming-http 방식)
#  - 디폴트 8000 포트로 구동되며, --port 인자로 포트를 지정하실 수 있습니다.
pyhub-llm mcp-server run greeting --port=8888

# 또는 Python 모듈로 실행
python -m pyhub.llm.mcp.servers calculator
python -m pyhub.llm.mcp.servers greeting --port=8888

# 사용 가능한 서버 목록 확인
pyhub-llm mcp-server list
```

계산기 서버는 다음 기능을 제공합니다:

- `add(a, b)`: 두 숫자를 더합니다
- `subtract(a, b)`: 두 숫자를 뺍니다
- `multiply(a, b)`: 두 숫자를 곱합니다
- `divide(a, b)`: 두 숫자를 나눕니다
- `power(base, exponent)`: 거듭제곱을 계산합니다

인사말 서버는 다음 기능을 제공합니다:

- `greeting(name, lang="en")`: 다국어 인사말을 생성합니다 (영어, 한국어, 스페인어, 프랑스어, 일본어 지원)

#### 2. MCP 도구 확인하기

MCP 서버에서 제공하는 도구 목록을 확인합니다:

```python
import asyncio
from pyhub.llm.mcp import MCPClient

async def list_mcp_tools():
    # 내장 계산기 서버 연결
    client = MCPClient({
        "transport": "stdio",
        "command": "pyhub-llm",
        "args": ["mcp-server", "run", "calculator"],
    })

    async with client.connect():
        # 사용 가능한 도구 목록 가져오기
        tools = await client.list_tools()
        
        print("사용 가능한 MCP 도구:")
        for tool in tools:
            print(f"\n도구 이름: {tool['name']}")
            print(f"설명: {tool['description']}")
            print(f"파라미터: {tool['parameters']}")

# 실행
asyncio.run(list_mcp_tools())
```

출력 예시:
```
사용 가능한 MCP 도구:

도구 이름: add
설명: 두 숫자를 더합니다
파라미터: {'type': 'object', 'properties': {'a': {'type': 'number', 'description': '첫 번째 숫자'}, 'b': {'type': 'number', 'description': '두 번째 숫자'}}, 'required': ['a', 'b']}

도구 이름: subtract
설명: 두 숫자를 뺍니다
파라미터: {'type': 'object', 'properties': {'a': {'type': 'number', 'description': '첫 번째 숫자'}, 'b': {'type': 'number', 'description': '두 번째 숫자'}}, 'required': ['a', 'b']}

도구 이름: multiply
설명: 두 숫자를 곱합니다
파라미터: {'type': 'object', 'properties': {'a': {'type': 'number', 'description': '첫 번째 숫자'}, 'b': {'type': 'number', 'description': '두 번째 숫자'}}, 'required': ['a', 'b']}

도구 이름: divide
설명: 두 숫자를 나눕니다
파라미터: {'type': 'object', 'properties': {'a': {'type': 'number', 'description': '나누어지는 수'}, 'b': {'type': 'number', 'description': '나누는 수'}}, 'required': ['a', 'b']}

도구 이름: power
설명: 거듭제곱을 계산합니다
파라미터: {'type': 'object', 'properties': {'base': {'type': 'number', 'description': '밑'}, 'exponent': {'type': 'number', 'description': '지수'}}, 'required': ['base', 'exponent']}
```

인사말 서버의 경우:

```python
# 인사말 서버 연결
client = MCPClient({
    "transport": "stdio",
    "command": "pyhub-llm",
    "args": ["mcp-server", "run", "greeting", "--port", "8888"],
})

# 출력 예시:
# 도구 이름: greeting
# 설명: Generate a greeting message in the specified language
# 파라미터: {'type': 'object', 'properties': {'name': {'type': 'string', 'description': 'Name of the person to greet'}, 'lang': {'type': 'string', 'description': 'Language code (en, ko, es, fr, ja)', 'default': 'en'}}, 'required': ['name']}
```

#### 3. llm.ask에서 MCP 도구 사용하기

MCP 도구를 LLM과 함께 사용하는 방법:

```python
import asyncio
import logging
from pyhub.llm import LLM
from pyhub.llm.mcp import MCPClient, load_mcp_tools

# 로깅 설정 (디버깅 메시지 확인)
logging.basicConfig(level=logging.DEBUG)

async def use_mcp_with_llm():
    # 새로운 dataclass 방식 (권장)
    from pyhub.llm.mcp import McpStdioConfig

    config = McpStdioConfig(
        name="calculator",
        cmd="pyhub-llm mcp-server run calculator"
    )
    client = MCPClient(config)

    # 또는 기존 dict 방식
    # client = MCPClient({
    #     "transport": "stdio",
    #     "command": "pyhub-llm",
    #     "args": ["mcp-server", "run", "calculator"],
    # })

    async with client.connect():
        # MCP 도구를 Tool 객체로 로드
        tools = await load_mcp_tools(client)

        # LLM 생성 (MCP 도구 포함)
        llm = LLM.create("gpt-4o-mini", tools=tools)

        # MCP 도구를 활용한 질문
        response = await llm.ask_async(
            "25와 17을 더한 다음, 그 결과에 3을 곱해주세요."
        )

        print(f"답변: {response}")

        # 도구 호출 내역 확인
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("\n도구 호출 내역:")
            for call in response.tool_calls:
                print(f"- {call.name}({call.args})")

# 실행
asyncio.run(use_mcp_with_llm())
```

출력 예시:
```
답변: 25와 17을 더하면 42이고, 여기에 3을 곱하면 126입니다.

도구 호출 내역:
- add({'a': 25, 'b': 17})
- multiply({'a': 42, 'b': 3})
```

#### 4. LLM과 MCP 통합 사용하기 (새로운 기능!)

이제 LLM 생성 시 MCP 서버를 직접 설정할 수 있어, 수동으로 연결을 관리할 필요가 없습니다:

##### 방법 1: create_async로 자동 초기화

```python
from pyhub.llm import LLM
from pyhub.llm.mcp import McpStdioConfig

# MCP가 자동으로 초기화되는 LLM 생성
llm = await LLM.create_async(
    "gpt-4o-mini",
    mcp_servers=McpStdioConfig(
        name="calculator",
        cmd="pyhub-llm mcp-server run calculator"
    )
)

# MCP 도구가 자동으로 사용됨
response = await llm.ask_async("25와 17을 더하면?")
print(response.text)

# 사용 후 MCP 연결 종료
await llm.close_mcp()
```

##### 방법 2: 컨텍스트 매니저 사용 (권장)

```python
# 컨텍스트 매니저로 자동 연결/해제
async with await LLM.create_async(
    "gpt-4o-mini",
    mcp_servers=[
        McpStdioConfig(name="calc", cmd="pyhub-llm mcp-server run calculator"),
        McpStreamableHttpConfig(name="web", url="http://localhost:8888/mcp")
    ]
) as llm:
    response = await llm.ask_async("100에서 37을 빼고 2를 곱하면?")
    print(response.text)
# 여기서 자동으로 MCP 연결이 종료됨
```

##### 방법 3: 수동 초기화

```python
# 동기적으로 LLM 생성 후 수동 초기화
llm = LLM.create("gpt-4o-mini", mcp_servers=mcp_config)
await llm.initialize_mcp()  # MCP 연결 시작

# 사용
response = await llm.ask_async("...")

# 종료
await llm.close_mcp()
```

##### 방법 4: 설정 파일 사용 (새로운 기능!)

MCP 설정을 JSON 또는 YAML 파일로 관리할 수 있습니다:

```yaml
# mcp_config.yaml
mcpServers:
  - type: stdio
    name: calculator
    cmd: pyhub-llm mcp-server run calculator
    timeout: 60
    description: 수학 계산 도구
  
  - type: streamable_http
    name: greeting
    url: http://localhost:8888/mcp
    filter_tools: greet,hello  # 특정 도구만 사용
```

```python
# 파일 경로로 직접 로드
llm = await LLM.create_async("gpt-4o-mini", mcp_servers="mcp_config.yaml")

# 또는 다른 설정과 함께
config = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "mcpServers": [
        {"type": "stdio", "name": "calc", "cmd": "..."}
    ]
}
llm = await LLM.create_async("gpt-4o-mini", mcp_servers=config)
```

#### 5. 여러 MCP 서버 통합하기

먼저 greeting 서버를 8888 포트로 실행합니다:

```bash
# greeting 서버를 8888 포트로 실행
pyhub-llm mcp-server run greeting --port 8888
```

기존 방식 (수동 관리):

```python
import asyncio
from pyhub.llm import LLM
from pyhub.llm.mcp import MultiServerMCPClient, McpStdioConfig, McpStreamableHttpConfig

async def use_multiple_mcp_servers():
    # 새로운 dataclass 방식 (권장)
    servers = [
        McpStdioConfig(
            name="calculator",
            cmd="pyhub-llm mcp-server run calculator",
            description="기본 계산 기능 제공"
        ),
        McpStreamableHttpConfig(
            name="greeting",
            url="http://localhost:8888/mcp",
            description="다국어 인사말 생성"
        )
    ]
    
    # MultiServerMCPClient로 여러 서버 연결
    multi_client = MultiServerMCPClient(servers)
    
    async with multi_client:
        # 모든 서버의 도구 가져오기
        all_tools = await multi_client.get_tools()
        
        print(f"총 {len(all_tools)}개의 도구를 로드했습니다:")
        for tool in all_tools:
            print(f"- {tool.name}: {tool.description}")
        
        # LLM 생성 (모든 도구 포함)
        llm = LLM.create("gpt-4o-mini", tools=all_tools)
        
        # 여러 서버의 도구를 함께 사용
        response = await llm.ask_async(
            "John에게 한국어로 인사하고, 20과 15를 더해주세요."
        )
        
        print(f"\n답변: {response}")

# 실행
asyncio.run(use_multiple_mcp_servers())
```

기존 dict 방식도 계속 지원합니다:

```python
# 기존 dict 방식 (하위 호환성)
servers = {
    "calculator": {
        "transport": "stdio",
        "command": "pyhub-llm",
        "args": ["mcp-server", "run", "calculator"],
    },
    "greeting": {
        "transport": "streamable_http",
        "url": "http://localhost:8888/mcp"
    }
}

multi_client = MultiServerMCPClient(servers)
```

#### 고급 사용법: 다양한 전송 방식

MCP는 다양한 전송 방식을 지원합니다:

```python
from pyhub.llm.mcp import MCPClient

# STDIO (로컬 프로세스)
stdio_client = MCPClient({
    "transport": "stdio",
    "command": "python3",
    "args": ["my_server.py"]
})

# HTTP
http_client = MCPClient({
    "transport": "streamable_http",
    "url": "http://localhost:8080/mcp"
})

# WebSocket
ws_client = MCPClient({
    "transport": "websocket",
    "url": "ws://localhost:8080/mcp/ws"
})

# Server-Sent Events (SSE)
sse_client = MCPClient({
    "transport": "sse",
    "url": "http://localhost:8080/mcp/sse"
})
```

## 개발

### 테스트 실행

```bash
# 모든 테스트
make test

# 특정 테스트
make test tests/test_openai.py

# 커버리지 포함 테스트
make test-cov
# 또는
make cov

# 커버리지 HTML 리포트 보기
make test-cov-report

# 특정 파일만 커버리지 테스트
make cov tests/test_optional_dependencies.py

# pytest 직접 실행
pytest --cov=src/pyhub/llm --cov-report=term --cov-report=html
```

### 코드 품질 검사

```bash
# 포맷팅 및 린팅
make format
make lint

# 타입 체크
mypy src/
```

### 빌드 및 배포

```bash
# 패키지 빌드
make build

# PyPI 배포 (권한 필요)
make release
```

## 기여하기

1. 이 저장소를 포크합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

### 기여 가이드라인

- 모든 새 기능에는 테스트를 포함해주세요
- 코드 스타일은 Black과 Ruff를 따릅니다
- 타입 힌트를 사용해주세요
- 문서를 업데이트해주세요

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 문제 해결

### 일반적인 문제

**Q: API 키 오류가 발생합니다**

```python
# 해결 방법 1: 환경 변수 설정
import os
os.environ["OPENAI_API_KEY"] = "your-key"

# 해결 방법 2: 직접 전달
llm = OpenAILLM(api_key="your-key")
```

**Q: 속도가 느립니다**

```python
# 캐시 인젝션으로 캐싱 활성화
from pyhub.llm.cache import MemoryCache
cache = MemoryCache()
llm = LLM.create("gpt-4o-mini", cache=cache)
reply = llm.ask("...")

# 더 빠른 모델 사용
llm = LLM.create("gpt-3.5-turbo")
```

**Q: 메모리 사용량이 높습니다**

```python
# 대화 히스토리 제한
llm = LLM.create(
    "gpt-4o-mini",
    initial_messages=[]  # 히스토리 없이 시작
)

# 주기적으로 히스토리 정리
if len(llm) > 10:
    llm.clear()
```

## 링크

- [문서](https://pyhub-llm.readthedocs.io)
- [PyPI](https://pypi.org/project/pyhub-llm)
- [GitHub](https://github.com/pyhub-kr/pyhub-llm)
- [이슈 트래커](https://github.com/pyhub-kr/pyhub-llm/issues)
