# MCP 리소스 관리 가이드

이 문서는 pyhub-llm에서 MCP(Model Context Protocol) 연결의 리소스 관리에 대해 설명합니다.

## 개요

MCP를 사용할 때 외부 프로세스나 네트워크 연결이 생성됩니다. 이러한 리소스들은 프로그램 종료 시 적절히 정리되어야 합니다. pyhub-llm은 다음과 같은 메커니즘을 통해 자동 리소스 정리를 지원합니다:

1. **Finalizer 패턴**: 객체가 가비지 컬렉션될 때 자동 정리
2. **시그널 핸들링**: SIGTERM/SIGINT 수신 시 graceful shutdown
3. **타임아웃 처리**: 정리 작업이 무한 대기하지 않도록 보장
4. **전역 레지스트리**: 모든 활성 연결 추적 및 관리

## 사용 방법

### 기본 사용법

```python
from pyhub.llm import LLM

# MCP 서버 설정
mcp_config = [{
    "type": "stdio",
    "name": "calculator",
    "cmd": ["python", "-m", "calculator_server"]
}]

# LLM 인스턴스 생성 (자동 리소스 관리)
llm = LLM.create("gpt-4o-mini", mcp_servers=mcp_config)

# 사용
response = llm.ask("2 + 2는?")

# 명시적 정리 (선택사항 - 자동으로도 처리됨)
await llm.close_mcp()
```

### 비동기 컨텍스트 매니저 (권장)

```python
async with await LLM.create_async("gpt-4o-mini", mcp_servers=mcp_config) as llm:
    response = await llm.ask_async("계산해줘: 15 * 24")
    # 컨텍스트 종료 시 자동 정리
```

### 수동 리소스 관리

```python
llm = await LLM.create_async("gpt-4o-mini", mcp_servers=mcp_config)
try:
    # LLM 사용
    response = await llm.ask_async("...")
finally:
    # 타임아웃 지정 가능 (기본 5초)
    await llm.close_mcp(timeout=10.0)
```

## 리소스 정리 메커니즘

### 1. Finalizer를 통한 자동 정리

MCP가 설정된 LLM 인스턴스는 생성 시 자동으로 finalizer가 등록됩니다:

```python
# 내부적으로 다음과 같이 동작
if self.mcp_servers:
    self._finalizer = weakref.finalize(self, cleanup_function)
```

객체가 가비지 컬렉션될 때 자동으로 MCP 연결이 종료됩니다.

### 2. 시그널 핸들링

프로그램이 SIGTERM이나 SIGINT를 받으면 모든 MCP 연결이 graceful하게 종료됩니다:

```python
# 자동으로 처리됨 - 별도 설정 불필요
# Ctrl+C를 누르거나 프로세스가 종료 시그널을 받으면
# 모든 활성 MCP 연결이 정리됨
```

### 3. 타임아웃 처리

리소스 정리 시 무한 대기를 방지하기 위해 타임아웃이 적용됩니다:

```python
# 기본 5초 타임아웃
await llm.close_mcp()

# 커스텀 타임아웃
await llm.close_mcp(timeout=10.0)
```

### 4. 전역 레지스트리

모든 MCP 연결은 전역 레지스트리에서 추적됩니다:

```python
from pyhub.llm.resource_manager import MCPResourceRegistry

# 싱글톤 인스턴스
registry = MCPResourceRegistry()

# 현재 활성 연결 수 확인
active_count = len(registry._instances)
```

## 주의사항

### 1. 장시간 실행 애플리케이션

장시간 실행되는 애플리케이션에서는 명시적으로 리소스를 관리하는 것이 좋습니다:

```python
# 주기적으로 사용하지 않는 연결 정리
async def periodic_cleanup():
    for llm in inactive_llms:
        await llm.close_mcp()
```

### 2. 프로세스 풀 사용 시

multiprocessing을 사용할 때는 각 프로세스에서 독립적으로 LLM 인스턴스를 생성해야 합니다:

```python
def worker_function():
    # 각 워커 프로세스에서 별도 인스턴스 생성
    llm = LLM.create("gpt-4o-mini", mcp_servers=config)
    try:
        # 작업 수행
        pass
    finally:
        asyncio.run(llm.close_mcp())
```

### 3. 테스트 환경

테스트에서는 시그널 핸들러가 자동으로 비활성화됩니다:

```python
# pytest 실행 시 자동으로 감지되어 시그널 핸들러 비활성화
# 필요한 경우 수동으로 활성화 가능
registry.enable_signal_handlers()
```

## 디버깅

리소스 정리 관련 로그를 확인하려면:

```python
import logging

# 리소스 관리자 로그 활성화
logging.getLogger('pyhub.llm.resource_manager').setLevel(logging.DEBUG)
logging.getLogger('pyhub.llm.mcp').setLevel(logging.DEBUG)
```

로그 예시:
```
INFO: MCP connections closed
WARNING: MCP cleanup timed out after 5.0s
ERROR: Error closing MCP connections: Connection refused
DEBUG: Finalizer called for instance 140234567890
```

## 문제 해결

### 프로세스가 종료되지 않는 경우

1. 타임아웃을 늘려보세요:
   ```python
   await llm.close_mcp(timeout=30.0)
   ```

2. 수동으로 모든 연결 정리:
   ```python
   registry = MCPResourceRegistry()
   await registry._async_cleanup_all()
   ```

### "There is no current event loop" 에러

비동기 컨텍스트가 아닌 곳에서 cleanup이 호출될 때 발생합니다. 이는 정상적인 동작이며, finalizer가 새 이벤트 루프를 생성하여 처리합니다.

### 리소스 누수 의심 시

1. 활성 연결 확인:
   ```python
   registry = MCPResourceRegistry()
   print(f"Active connections: {len(registry._instances)}")
   ```

2. 명시적 정리 강제:
   ```python
   for instance_id in list(registry._instances.keys()):
       await registry._cleanup_instance(instance_id)
   ```

## Best Practices

1. **항상 컨텍스트 매니저 사용**: 가능하면 `async with` 구문을 사용하세요.

2. **장시간 연결 피하기**: MCP 연결을 필요할 때만 생성하고 사용 후 정리하세요.

3. **에러 핸들링**: MCP 작업 시 항상 예외 처리를 포함하세요:
   ```python
   try:
       response = await llm.ask_async("...")
   except Exception as e:
       logger.error(f"MCP operation failed: {e}")
   finally:
       await llm.close_mcp()
   ```

4. **리소스 모니터링**: 프로덕션 환경에서는 활성 연결 수를 모니터링하세요.

## 관련 링크

- [MCP 프로토콜 문서](https://github.com/anthropics/mcp)
- [pyhub-llm MCP 통합 가이드](./mcp_integration.md)
- [비동기 프로그래밍 가이드](./async_guide.md)