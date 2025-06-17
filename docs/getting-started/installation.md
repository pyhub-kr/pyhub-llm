# 설치

## 시스템 요구사항

- **Python**: 3.10 이상
- **운영체제**: Windows, macOS, Linux
- **메모리**: 최소 4GB RAM 권장

## 설치 방법

### pip를 사용한 설치

=== "기본 설치"

    ```bash
    pip install pyhub-llm
    ```
    
    !!! info "기본 설치 포함 내용"
        - 핵심 기능
        - 기본 의존성
        - 타입 힌트 지원

=== "특정 프로바이더"

    ```bash
    # OpenAI만 설치
    pip install pyhub-llm[openai]
    
    # 여러 프로바이더 설치
    pip install pyhub-llm[openai,anthropic]
    
    # 사용 가능한 프로바이더
    # - openai: OpenAI GPT 모델
    # - anthropic: Claude 모델
    # - google: Gemini 모델
    # - ollama: 로컬 모델
    # - upstage: Solar 모델
    ```

=== "전체 설치"

    ```bash
    # 모든 프로바이더와 추가 기능
    pip install pyhub-llm[all]
    ```
    
    !!! warning "전체 설치 시 주의사항"
        모든 프로바이더의 의존성을 설치하므로 설치 시간이 오래 걸릴 수 있습니다.

=== "개발 환경"

    ```bash
    # 개발 도구 포함 설치
    pip install pyhub-llm[dev]
    
    # 문서 도구 포함
    pip install pyhub-llm[docs]
    
    # 모든 것 포함
    pip install pyhub-llm[all,dev,docs]
    ```

### poetry를 사용한 설치

```bash
# 기본 설치
poetry add pyhub-llm

# 특정 프로바이더
poetry add pyhub-llm[openai,anthropic]

# 개발 의존성
poetry add --group dev pyhub-llm[dev]
```

### 소스에서 설치

```bash
# 저장소 클론
git clone https://github.com/pyhub-kr/pyhub-llm.git
cd pyhub-llm

# 개발 모드로 설치
pip install -e ".[all,dev]"
```

## API 키 설정

### 환경 변수로 설정

=== "Linux/macOS"

    ```bash
    export OPENAI_API_KEY="your-openai-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"
    export GOOGLE_API_KEY="your-google-key"
    export UPSTAGE_API_KEY="your-upstage-key"
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:OPENAI_API_KEY="your-openai-key"
    $env:ANTHROPIC_API_KEY="your-anthropic-key"
    $env:GOOGLE_API_KEY="your-google-key"
    $env:UPSTAGE_API_KEY="your-upstage-key"
    ```

=== "Windows (CMD)"

    ```cmd
    set OPENAI_API_KEY=your-openai-key
    set ANTHROPIC_API_KEY=your-anthropic-key
    set GOOGLE_API_KEY=your-google-key
    set UPSTAGE_API_KEY=your-upstage-key
    ```

### .env 파일 사용

프로젝트 루트에 `.env` 파일을 생성하고 API 키를 저장합니다:

```bash
# .env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
UPSTAGE_API_KEY=your-upstage-key
```

!!! warning "보안 주의사항"
    `.env` 파일을 git에 커밋하지 않도록 `.gitignore`에 추가하세요!

### 코드에서 직접 설정

```python
from pyhub.llm import OpenAILLM, AnthropicLLM

# API 키 직접 전달
openai_llm = OpenAILLM(
    model="gpt-4o-mini",
    api_key="your-openai-key"
)

anthropic_llm = AnthropicLLM(
    model="claude-3-5-haiku-latest",
    api_key="your-anthropic-key"
)
```

## API 키 발급 방법

| 프로바이더 | 발급 링크 | 무료 크레딧 |
|----------|---------|-----------|
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | $5 (신규 계정) |
| Anthropic | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) | $5 (신규 계정) |
| Google | [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey) | 무료 티어 제공 |
| Upstage | [console.upstage.ai/](https://console.upstage.ai/) | 무료 크레딧 제공 |

## 설치 확인

설치가 완료되었는지 확인해보세요:

```python
# Python에서 확인
import pyhub.llm
print(pyhub.llm.__version__)

# 사용 가능한 프로바이더 확인
from pyhub.llm import LLM
print(LLM.available_providers())
```

명령줄에서 확인:

```bash
# 버전 확인
python -c "import pyhub.llm; print(pyhub.llm.__version__)"

# CLI 도구 확인
pyhub-llm --version
```

## 문제 해결

### ImportError 발생 시

```python
# ImportError: cannot import name 'OpenAILLM'
```

해당 프로바이더가 설치되지 않았습니다:

```bash
pip install pyhub-llm[openai]
```

### API 키 오류

```python
# AuthenticationError: Invalid API key
```

1. API 키가 올바른지 확인
2. 환경 변수명이 정확한지 확인
3. `.env` 파일 위치가 올바른지 확인

### 의존성 충돌

```bash
# 기존 패키지 제거 후 재설치
pip uninstall pyhub-llm
pip install pyhub-llm[all] --upgrade
```

## 다음 단계

설치가 완료되었다면 [빠른 시작 가이드](quickstart.md)를 통해 첫 번째 애플리케이션을 만들어보세요!