# pyhub-llm

다양한 LLM 제공업체를 위한 통합 Python 라이브러리입니다. OpenAI, Anthropic, Google, Ollama 등의 API를 일관된 인터페이스로 사용할 수 있습니다.

## 📚 문서

### [✨ CHEATSHEET](./CHEATSHEET.md) - 수준별 완전한 가이드

pyhub-llm의 모든 기능을 수준별로 나누어 정리했습니다:
- 🌱 [초급 가이드](./CHEATSHEET-BASIC.md): 설치, 기본 사용법, 파일 처리
- 🚀 [중급 가이드](./CHEATSHEET-INTERMEDIATE.md): 구조화된 출력, 캐싱, 도구 호출, History Backup
- 🔥 [고급 가이드](./CHEATSHEET-ADVANCED.md): MCP 통합, 체이닝, 웹 프레임워크 통합
- 📖 [기능별 찾아보기](./CHEATSHEET.md#기능별-찾아보기): 원하는 기능을 빠르게 검색

## 주요 기능

- 🔌 **통합 인터페이스**: 모든 LLM 제공업체를 동일한 방식으로 사용
- 🚀 **간편한 전환**: 코드 변경 없이 모델 전환 가능
- 💾 **캐싱 지원**: 응답 캐싱으로 비용 절감 및 성능 향상
- 🔄 **스트리밍 지원**: 실시간 응답 스트리밍
- 🛠️ **도구/함수 호출**: Function calling 지원
- 📷 **이미지 처리**: 이미지 설명 및 분석 기능
- ⚡ **비동기 지원**: 동기/비동기 모두 지원
- 🔗 **체이닝**: 여러 LLM을 연결하여 복잡한 워크플로우 구성
- 💾 **대화 히스토리 백업**: 대화 내역을 외부 저장소에 백업 및 복원

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

# 이미지 기능 사용시 (Pillow 포함)
pip install "pyhub-llm[image]"

# 모든 기능 설치
pip install "pyhub-llm[all]"
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

> 더 많은 예시와 고급 사용법은 [초급 가이드](./CHEATSHEET-BASIC.md#기본-사용법)를 참고하세요.

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
# 실시간으로 응답 받기
for chunk in llm.ask("긴 이야기를 들려주세요", stream=True):
    print(chunk.text, end="", flush=True)
```

### 2. 출력 포맷팅 (NEW! 🎨)

```python
from pyhub.llm import LLM, display

# 마크다운 렌더링과 함께 스트리밍
response = llm.ask("파이썬 함수 작성법을 알려주세요", stream=True)
display(response)  # 자동으로 마크다운 렌더링!

# 또는 Response 객체의 print() 메서드 사용
response = llm.ask("안녕하세요")
response.print(markdown=True)

# 일반 텍스트로 출력
response.print(markdown=False)
```

### 3. 이미지 생성 (NEW! 🎨)

```python
from pyhub.llm import OpenAILLM

# DALL-E 3로 이미지 생성
llm = OpenAILLM(model="dall-e-3")
reply = llm.generate_image(
    "A beautiful sunset over mountains",
    size="1024x1792",  # 세로 형식
    quality="hd"       # 고품질
)

# 이미지 저장
path = reply.save("sunset.png")  # 또는 reply.save() 로 자동 파일명
print(f"Saved to: {path}")

# BytesIO에 저장 (NEW!)
from io import BytesIO
buffer = BytesIO()
reply.save(buffer)
buffer.seek(0)  # 읽기 위해 처음으로

# Django ImageField 통합 (v0.9.1+)
from django.db import models
class MyModel(models.Model):
    image = models.ImageField(upload_to='generated/')

# 간단하게 ImageField에 저장
instance = MyModel(image=reply.to_django_file())
instance.save()

# 이미지 표시 (Jupyter)
reply.display()

# PIL로 변환 (Pillow 필요)
img = reply.to_pil()
img.thumbnail((512, 512))
img.save("thumbnail.png")

# 비동기 처리
import asyncio

async def generate_multiple():
    tasks = [
        llm.generate_image_async(f"Image {i}") 
        for i in range(3)
    ]
    images = await asyncio.gather(*tasks)
    
    # 병렬 저장
    save_tasks = [img.save_async(f"img_{i}.png") for i, img in enumerate(images)]
    await asyncio.gather(*save_tasks)
```

### 4. 파일 처리 (이미지 및 PDF)

```python
# 이미지 설명
reply = llm.ask("이 이미지를 설명해주세요", files=["photo.jpg"])

# PDF 요약
reply = llm.ask("이 문서를 요약해주세요", files=["document.pdf"])
```

### 5. 도구/함수 호출

```python
# 간단한 함수를 도구로 사용
def get_weather(city: str) -> str:
    """도시의 날씨 정보를 가져옵니다."""
    return f"{city}의 날씨는 맑음입니다."

reply = llm.ask("서울 날씨 알려줘", tools=[get_weather])
```

> 📖 더 많은 고급 기능과 상세한 예시는 [가이드 문서](./CHEATSHEET.md)를 참고하세요:
> - 대화 히스토리 관리
> - 구조화된 출력 (Pydantic)
> - LLM 체이닝
> - 캐싱 전략
> - MCP 통합
> - 웹 프레임워크 통합 (FastAPI, Django)
> - 에러 처리 및 재시도
> - 대화 히스토리 백업 및 복원

### 5. 대화 히스토리 백업

```python
from pyhub.llm import LLM
from pyhub.llm.history import InMemoryHistoryBackup

# 백업 저장소 생성
backup = InMemoryHistoryBackup(user_id="user123", session_id="session456")

# 백업이 활성화된 LLM 생성
llm = LLM.create("gpt-4o-mini", history_backup=backup)

# 대화 진행 (자동으로 백업됨)
llm.ask("Python의 장점은 무엇인가요?")
llm.ask("더 자세히 설명해주세요")

# 사용량 확인
usage = backup.get_usage_summary()
print(f"총 사용 토큰: {usage.input + usage.output}")

# SQLAlchemy로 영구 저장
from pyhub.llm.history import SQLAlchemyHistoryBackup
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///chat_history.db")
Session = sessionmaker(bind=engine)
session = Session()

db_backup = SQLAlchemyHistoryBackup(session, user_id="user123", session_id="session456")
llm_with_db = LLM.create("gpt-4o-mini", history_backup=db_backup)
```

> 🔍 대화 히스토리 백업은 도구(Tool) 사용 내역도 자동으로 저장합니다. 자세한 사용법은 [중급 가이드](./CHEATSHEET-INTERMEDIATE.md#history-backup)를 참고하세요.

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

```bash
# 대화형 채팅
pyhub-llm chat --model gpt-4o-mini

# 단일 질문
pyhub-llm ask "Python과 Go의 차이점은?"

# 파일과 함께 질문
pyhub-llm ask "이 코드를 리뷰해주세요" --file main.py
```

> 🔧 더 많은 CLI 옵션과 사용법은 [초급 가이드](./CHEATSHEET-BASIC.md)를 참고하세요.


## 고급 기능

### Stateless 모드

반복적인 독립 작업(분류, 정보 추출 등)을 수행할 때 불필요한 대화 히스토리 누적을 방지하는 모드입니다.

```python
from pyhub.llm import LLM

# Stateless 모드로 LLM 생성
classifier = LLM.create("gpt-4o-mini", stateless=True)

# 각 요청이 독립적으로 처리되며 히스토리가 저장되지 않음
for text in customer_queries:
    reply = classifier.ask(
        f"고객 문의 의도 분류: {text}",
        choices=["환불", "배송", "문의", "불만"]
    )
    print(f"{text} -> {reply.choice}")
    # 히스토리가 누적되지 않아 API 비용 절감
```

**Stateless 모드의 특징:**
- 대화 히스토리를 전혀 저장하지 않음
- `use_history=True`를 지정해도 무시됨
- `clear()` 메서드가 아무 동작도 하지 않음
- 반복 작업에서 메모리 사용량과 API 비용 절감

**일반 모드와의 비교:**
```python
# 일반 모드 (히스토리 누적)
normal_llm = LLM.create("gpt-4o-mini")
normal_llm.ask("분류1", choices=["A", "B"])  # 2개 메시지 저장
normal_llm.ask("분류2", choices=["C", "D"])  # 4개 메시지 저장 (이전 대화 포함)

# Stateless 모드 (히스토리 없음)
stateless_llm = LLM.create("gpt-4o-mini", stateless=True)
stateless_llm.ask("분류1", choices=["A", "B"])  # 0개 메시지
stateless_llm.ask("분류2", choices=["C", "D"])  # 0개 메시지 (독립 처리)
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

## 알려진 이슈

### OpenAI JSON Schema 구조화된 출력의 토큰 생성 이슈 (v0.8.0에서 개선)

OpenAI의 JSON Schema 구조화된 출력 모드(strict=true)에서 간헐적으로 무한 개행 문자나 제어 문자가 생성되는 문제가 보고되었습니다. 이는 GPT-4o-2024-08-06 모델에서 발견된 알려진 문제입니다.

**증상:**
```python
# 무한 개행 문자 생성
{"choice": "\n\n\n\n\n..."}  # max_tokens까지 계속

# 제어 문자 포함
{"choice": "\u001cA/S\u001d0\u001d\u001d..."}
```

**v0.8.0+ 개선사항:**
- `strict: true` 설정으로 스키마 준수 강화
- `choice_index` 필드를 통한 안정적인 선택
- 시스템 프롬프트 자동 생성 (choices 사용 시)

```python
# 시스템 프롬프트 없이도 작동
llm = OpenAILLM()
reply = llm.ask("분류해주세요", choices=["환불/반품", "A/S요청"])
print(reply.choice)  # "환불/반품" 또는 "A/S요청"
print(reply.choice_index)  # 0 또는 1
```

**이전 버전 해결 방법:**
1. 특수문자를 제거하거나 대체: `"A/S요청"` → `"AS요청"`
2. 버전 0.7.1-0.7.x에서는 제어 문자가 부분적으로 필터링됩니다

자세한 내용은 [Issue #22](https://github.com/pyhub-kr/pyhub-llm/issues/22)를 참조하세요.

## 링크

- [문서](https://pyhub-llm.readthedocs.io)
- [PyPI](https://pypi.org/project/pyhub-llm)
- [GitHub](https://github.com/pyhub-kr/pyhub-llm)
- [이슈 트래커](https://github.com/pyhub-kr/pyhub-llm/issues)
