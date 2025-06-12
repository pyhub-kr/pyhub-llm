# LLM MCP 통합 TODO 리스트

## 목표
LLM 생성자에서 MCP 설정을 직접 받아 자동으로 연결을 관리하도록 개선

## TODO 리스트

### 1. BaseLLM 클래스 확장
- [x] BaseLLM에 `mcp_servers` 파라미터 추가
- [x] MCP 클라이언트 인스턴스 변수 추가 (`_mcp_client`)
- [x] MCP 연결 상태 플래그 추가 (`_mcp_connected`)

### 2. 비동기 초기화 지원
- [x] `initialize_mcp()` 비동기 메서드 추가
- [x] MCP 서버 연결 및 도구 로드 로직 구현
- [x] 연결 실패 시 graceful degradation

### 3. LLMFactory 수정
- [x] `create()` 메서드에 `mcp_servers` 파라미터 추가
- [x] `create_async()` 메서드 추가 (MCP 자동 초기화)
- [x] 타입 힌트 업데이트

### 4. 도구 통합
- [x] MCP 도구와 일반 도구 병합 로직
- [x] 도구 이름 충돌 처리
- [x] 런타임 도구 업데이트 지원

### 5. 생명주기 관리
- [x] `__aenter__` / `__aexit__` 메서드 구현
- [x] 명시적 `close_mcp()` 메서드 추가
- [x] 가비지 컬렉션 시 자동 정리

### 6. 테스트 작성
- [x] 단일 MCP 서버 테스트
- [x] 다중 MCP 서버 테스트
- [x] 연결 실패 시나리오 테스트
- [x] 도구 병합 테스트

### 7. 문서화
- [x] README.md 업데이트
- [x] 새로운 사용법 예제 추가
- [x] 마이그레이션 가이드 작성

### 8. 예제 코드
- [x] 간단한 사용 예제
- [x] 복잡한 시나리오 예제
- [x] 에러 처리 예제

## 구현 순서
1. BaseLLM 클래스 확장 (기본 구조)
2. 비동기 초기화 로직
3. LLMFactory 수정
4. 테스트 작성 및 검증
5. 문서화 및 예제

## 예상 API

```python
# 방법 1: 동기 생성 + 수동 초기화
llm = LLM.create("gpt-4o-mini", mcp_servers=[...])
await llm.initialize_mcp()  # MCP 연결

# 방법 2: 비동기 팩토리 (권장)
llm = await LLM.create_async(
    "gpt-4o-mini",
    mcp_servers=[
        McpStdioConfig(name="calc", cmd="..."),
        McpStreamableHttpConfig(name="web", url="...")
    ]
)

# 방법 3: 컨텍스트 매니저
async with LLM.create("gpt-4o-mini", mcp_servers=[...]) as llm:
    response = await llm.ask_async("...")
```

## 주의사항
- 기존 API와의 하위 호환성 유지
- MCP는 선택적 기능으로 유지
- 동기/비동기 사용 모두 지원