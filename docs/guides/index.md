# 가이드

pyhub-llm의 다양한 기능을 자세히 알아봅니다.

## 가이드 목록

<div class="grid cards" markdown>

-   :material-book-open: **[기본 사용법](basic-usage.md)**
    
    ---
    
    LLM 인스턴스 생성, 질문하기, 응답 처리 등 기본적인 사용 방법을 알아봅니다.

-   :material-message-processing: **[대화 관리](conversation.md)**
    
    ---
    
    대화 히스토리 관리, Stateless 모드, 컨텍스트 윈도우 등을 다룹니다.

-   :material-swap-horizontal: **[프로바이더](providers.md)**
    
    ---
    
    OpenAI, Anthropic, Google 등 각 프로바이더의 특징과 사용법을 알아봅니다.

-   :material-code-json: **[구조화된 출력](structured-output.md)**
    
    ---
    
    Pydantic 스키마를 사용한 타입 안전한 응답 처리 방법을 배웁니다.

-   :material-rocket: **[고급 기능](advanced.md)**
    
    ---
    
    임베딩, 캐싱, 스트리밍, 비동기 처리 등 고급 기능을 다룹니다.

</div>

## 학습 경로

### 초급자

1. [기본 사용법](basic-usage.md)부터 시작하세요
2. [대화 관리](conversation.md)로 대화형 애플리케이션 만들기
3. [프로바이더](providers.md)를 비교하고 선택하기

### 중급자

1. [구조화된 출력](structured-output.md)으로 복잡한 데이터 처리
2. [고급 기능](advanced.md)의 스트리밍과 비동기 처리 활용
3. 성능 최적화 기법 적용

### 고급자

1. 커스텀 프로바이더 구현
2. MCP(Model Context Protocol) 통합
3. 대규모 시스템 아키텍처 설계

## 주요 개념

!!! info "통합 인터페이스"
    모든 LLM 프로바이더는 동일한 인터페이스를 사용합니다. 한 번 배우면 모든 프로바이더에 적용할 수 있습니다.

!!! tip "타입 안전성"
    pyhub-llm은 완전한 타입 힌트를 제공합니다. IDE의 자동 완성과 타입 검사를 활용하세요.

!!! warning "API 비용"
    대부분의 LLM API는 토큰 단위로 과금됩니다. 캐싱과 Stateless 모드를 활용해 비용을 절감하세요.

## 예제 중심 학습

각 가이드는 실제 코드 예제를 중심으로 구성되어 있습니다:

```python
# 모든 예제는 바로 실행 가능합니다
from pyhub.llm import LLM

llm = LLM.create("gpt-4o-mini")
reply = llm.ask("Hello, World!")
print(reply.text)
```

## 도움이 필요하신가요?

- 💡 각 페이지의 예제 코드를 직접 실행해보세요
- 📝 코드 블록의 복사 버튼을 활용하세요
- 🔍 검색 기능으로 필요한 내용을 빠르게 찾으세요
- 💬 질문이 있다면 [GitHub Discussions](https://github.com/pyhub-kr/pyhub-llm/discussions)에 남겨주세요