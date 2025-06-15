# pyhub-llm CHEATSHEET

이 문서는 pyhub-llm의 모든 기능을 다루는 종합 가이드입니다. 사용자 수준에 맞춰 구성되어 있습니다.

## 📚 수준별 가이드

### 🌱 [초급 (BASIC)](./CHEATSHEET-BASIC.md)
처음 시작하는 분들을 위한 기본 가이드
- 설치 및 환경 설정
- 기본 사용법
- 간단한 대화 관리
- 파일 처리 (이미지, PDF)
- 기본 에러 처리

### 🚀 [중급 (INTERMEDIATE)](./CHEATSHEET-INTERMEDIATE.md)
더 많은 기능을 활용하고 싶은 분들을 위한 가이드
- 구조화된 출력 (Pydantic)
- 분류 및 선택
- 비동기 처리
- 캐싱 전략
- 도구/함수 호출
- 템플릿 활용
- History Backup

### 🔥 [고급 (ADVANCED)](./CHEATSHEET-ADVANCED.md)
복잡한 애플리케이션 개발을 위한 가이드
- MCP 통합
- LLM 체이닝
- 웹 프레임워크 통합 (FastAPI, Django, Streamlit)
- 임베딩
- 고급 에러 처리
- 성능 최적화

## 🔍 빠른 참조

### 설치
```bash
# 전체 설치 (모든 프로바이더)
pip install "pyhub-llm[all]"

# 특정 프로바이더만
pip install "pyhub-llm[openai]"      # OpenAI만
pip install "pyhub-llm[anthropic]"   # Anthropic만
pip install "pyhub-llm[google]"      # Google만
pip install "pyhub-llm[ollama]"      # Ollama만
```

### 기본 사용
```python
from pyhub.llm import LLM

# LLM 생성
llm = LLM.create("gpt-4o-mini")

# 질문하기
reply = llm.ask("안녕하세요!")
print(reply.text)

# 스트리밍
for chunk in llm.ask("긴 답변을 주세요", stream=True):
    print(chunk.text, end="", flush=True)
```

### 환경변수
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## 📖 기능별 찾아보기

| 기능 | 수준 | 링크 |
|------|------|------|
| 설치 및 환경 설정 | 초급 | [BASIC - 설치](./CHEATSHEET-BASIC.md#설치) |
| 기본 사용법 | 초급 | [BASIC - 기본 사용법](./CHEATSHEET-BASIC.md#기본-사용법) |
| 스트리밍 | 초급 | [BASIC - 스트리밍](./CHEATSHEET-BASIC.md#스트리밍) |
| 대화 관리 | 초급 | [BASIC - 대화 관리](./CHEATSHEET-BASIC.md#대화-관리) |
| 파일 처리 | 초급 | [BASIC - 파일 처리](./CHEATSHEET-BASIC.md#파일-처리) |
| 구조화된 출력 | 중급 | [INTERMEDIATE - 구조화된 출력](./CHEATSHEET-INTERMEDIATE.md#구조화된-출력) |
| 비동기 처리 | 중급 | [INTERMEDIATE - 비동기 처리](./CHEATSHEET-INTERMEDIATE.md#비동기-처리) |
| 캐싱 | 중급 | [INTERMEDIATE - 캐싱](./CHEATSHEET-INTERMEDIATE.md#캐싱) |
| 도구/함수 호출 | 중급 | [INTERMEDIATE - 도구함수 호출](./CHEATSHEET-INTERMEDIATE.md#도구함수-호출) |
| 템플릿 | 중급 | [INTERMEDIATE - 템플릿 활용](./CHEATSHEET-INTERMEDIATE.md#템플릿-활용) |
| History Backup | 중급 | [INTERMEDIATE - History Backup](./CHEATSHEET-INTERMEDIATE.md#history-backup) |
| MCP 통합 | 고급 | [ADVANCED - MCP 통합](./CHEATSHEET-ADVANCED.md#mcp-통합) |
| 체이닝 | 고급 | [ADVANCED - 체이닝](./CHEATSHEET-ADVANCED.md#체이닝) |
| 웹 프레임워크 | 고급 | [ADVANCED - 웹 프레임워크 통합](./CHEATSHEET-ADVANCED.md#웹-프레임워크-통합) |
| 임베딩 | 고급 | [ADVANCED - 임베딩](./CHEATSHEET-ADVANCED.md#임베딩) |

## 💡 도움말

- **처음 시작하신다면**: [초급 가이드](./CHEATSHEET-BASIC.md)부터 시작하세요
- **특정 기능을 찾으신다면**: 위의 기능별 찾아보기 표를 참고하세요
- **예제가 필요하다면**: 각 수준별 가이드의 실용적인 예제 섹션을 확인하세요

## 🤝 기여

문서 개선에 기여하고 싶으시다면:
1. 오타나 잘못된 정보를 발견하면 이슈를 생성해주세요
2. 더 나은 예제나 설명이 있다면 PR을 보내주세요
3. 새로운 기능이 추가되면 적절한 수준의 문서에 추가해주세요