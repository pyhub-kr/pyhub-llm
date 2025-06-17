# pyhub-llm

<div align="center">
  <img src="https://img.shields.io/pypi/v/pyhub-llm.svg" alt="PyPI Version">
  <img src="https://img.shields.io/pypi/pyversions/pyhub-llm.svg" alt="Python Versions">
  <img src="https://img.shields.io/pypi/l/pyhub-llm.svg" alt="License">
  <img src="https://img.shields.io/pypi/dm/pyhub-llm.svg" alt="Downloads">
</div>

## 여러 LLM 프로바이더를 통합하는 Python 라이브러리

pyhub-llm은 OpenAI, Anthropic, Google, Ollama 등 다양한 LLM 프로바이더를 하나의 통합된 인터페이스로 사용할 수 있게 해주는 Python 라이브러리입니다.

!!! tip "주요 특징"
    - 🔄 **통합 인터페이스**: 모든 LLM을 동일한 방식으로 사용
    - 🚀 **간편한 전환**: 한 줄의 코드로 프로바이더 변경
    - ⚡ **비동기 지원**: 동기/비동기 모두 지원
    - 🔧 **확장 가능**: 쉬운 커스터마이징과 확장
    - 📝 **타입 안전**: 완전한 타입 힌트 지원

## 빠른 시작

=== "기본 사용법"

    ```python
    from pyhub.llm import LLM
    
    # 모델명으로 자동 프로바이더 감지
    llm = LLM.create("gpt-4o-mini")
    
    # 간단한 질문
    reply = llm.ask("파이썬의 장점을 설명해주세요")
    print(reply.text)
    ```

=== "프로바이더별 사용"

    ```python
    from pyhub.llm import OpenAILLM, AnthropicLLM, GoogleLLM
    
    # OpenAI
    openai_llm = OpenAILLM(model="gpt-4o-mini")
    
    # Anthropic  
    anthropic_llm = AnthropicLLM(model="claude-3-5-haiku-latest")
    
    # Google
    google_llm = GoogleLLM(model="gemini-2.0-flash-exp")
    
    # 모두 동일한 인터페이스 사용
    for llm in [openai_llm, anthropic_llm, google_llm]:
        reply = llm.ask("안녕하세요!")
        print(f"{llm.__class__.__name__}: {reply.text}")
    ```

## 주요 기능

<div class="grid cards" markdown>

-   :material-swap-horizontal: **통합 인터페이스**
    
    ---
    
    모든 LLM 프로바이더를 동일한 방식으로 사용할 수 있습니다.
    
    [:octicons-arrow-right-24: 프로바이더 가이드](guides/providers.md)

-   :material-message-processing: **대화 관리**
    
    ---
    
    자동 대화 히스토리 관리와 Stateless 모드를 지원합니다.
    
    [:octicons-arrow-right-24: 대화 관리 가이드](guides/conversation.md)

-   :material-code-json: **구조화된 출력**
    
    ---
    
    Pydantic 스키마를 사용한 타입 안전한 응답을 받을 수 있습니다.
    
    [:octicons-arrow-right-24: 구조화된 출력](guides/structured-output.md)

-   :material-rocket-launch: **성능 최적화**
    
    ---
    
    캐싱, 스트리밍, 비동기 처리로 최적의 성능을 제공합니다.
    
    [:octicons-arrow-right-24: 고급 기능](guides/advanced.md)

</div>

## 설치

=== "기본 설치"

    ```bash
    pip install pyhub-llm
    ```

=== "특정 프로바이더"

    ```bash
    # OpenAI만 설치
    pip install pyhub-llm[openai]
    
    # 여러 프로바이더 설치
    pip install pyhub-llm[openai,anthropic,google]
    ```

=== "전체 설치"

    ```bash
    # 모든 프로바이더와 추가 기능 설치
    pip install pyhub-llm[all]
    ```

## 다음 단계

- [설치 가이드](getting-started/installation.md)를 통해 상세한 설치 방법을 확인하세요
- [빠른 시작 가이드](getting-started/quickstart.md)로 기본 사용법을 익혀보세요
- [예제 코드](examples/index.md)를 통해 실제 사용 사례를 확인하세요

## 도움이 필요하신가요?

- 📧 문의: me@pyhub.kr
- 🐛 버그 리포트: [GitHub Issues](https://github.com/pyhub-kr/pyhub-llm/issues)
- 💬 토론: [GitHub Discussions](https://github.com/pyhub-kr/pyhub-llm/discussions)