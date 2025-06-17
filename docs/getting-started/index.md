# 시작하기

pyhub-llm을 시작하는 방법을 안내합니다.

## 이 섹션의 내용

<div class="grid cards" markdown>

-   :material-download: **[설치](installation.md)**
    
    ---
    
    pyhub-llm을 설치하고 환경을 설정하는 방법을 알아봅니다.

-   :material-rocket: **[빠른 시작](quickstart.md)**
    
    ---
    
    5분 안에 첫 번째 LLM 애플리케이션을 만들어봅니다.

</div>

## 전제 조건

- Python 3.10 이상
- pip 또는 poetry
- API 키 (사용하려는 LLM 프로바이더)

## 지원 프로바이더

| 프로바이더 | 필요한 API 키 | 설치 명령 |
|----------|-------------|----------|
| OpenAI | `OPENAI_API_KEY` | `pip install pyhub-llm[openai]` |
| Anthropic | `ANTHROPIC_API_KEY` | `pip install pyhub-llm[anthropic]` |
| Google | `GOOGLE_API_KEY` | `pip install pyhub-llm[google]` |
| Ollama | 없음 (로컬) | `pip install pyhub-llm[ollama]` |

## 첫 번째 코드

```python
from pyhub.llm import LLM

# LLM 인스턴스 생성
llm = LLM.create("gpt-4o-mini")

# 질문하기
reply = llm.ask("안녕하세요! 파이썬에 대해 알려주세요.")
print(reply.text)
```

준비되셨나요? [설치 가이드](installation.md)로 시작해보세요!