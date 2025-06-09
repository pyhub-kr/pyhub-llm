# MCP Server Examples

이 디렉토리는 pyhub-llm과 함께 사용할 수 있는 MCP (Model Context Protocol) 서버 예제들을 포함하고 있습니다. 모든 transport 방식(STDIO, SSE, WebSocket, HTTP)에 대한 완전한 예제와 클라이언트 테스트 코드를 제공합니다.

## 📁 디렉토리 구조

```
examples/mcp-servers/
├── README.md                    # 이 문서
├── requirements.txt             # Python 의존성
├── docker-compose.yml          # Docker 실행 설정
├── stdio/                       # STDIO transport
│   ├── calculator_server.py     # 계산기 MCP 서버
│   └── run.sh                   # 실행 스크립트
├── sse/                         # SSE transport  
│   ├── sse_server.py           # SSE MCP 서버
│   └── run.sh
├── websocket/                   # WebSocket transport
│   ├── ws_server.py            # WebSocket MCP 서버
│   └── run.sh
├── http/                        # Streamable HTTP transport
│   ├── http_server.py          # HTTP MCP 서버
│   └── run.sh
└── client_examples/             # 클라이언트 예제
    ├── test_all_transports.py  # 모든 transport 테스트
    └── usage_examples.py       # 실제 사용 예제
```

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
# 이 디렉토리에서 실행
pip install -r requirements.txt

# 또는 pyhub-llm이 이미 설치되어 있다면
pip install mcp fastapi uvicorn websockets sse-starlette
```

### 2. pyhub-llm 설치

```bash
# 프로젝트 루트에서 실행
cd ../..
pip install -e ".[dev,all]"
```

## 🚀 서버 실행

### STDIO 서버 (항상 사용 가능)

```bash
# 직접 실행
cd stdio
python3 calculator_server.py

# 또는 스크립트 사용
./run.sh
```

### 네트워크 서버들

각 서버는 서로 다른 포트에서 실행됩니다:

```bash
# SSE 서버 (포트 8001)
cd sse
./run.sh

# WebSocket 서버 (포트 8002) 
cd websocket
./run.sh

# HTTP 서버 (포트 8003)
cd http
./run.sh
```

### Docker로 모든 서버 실행

```bash
# 모든 네트워크 서버 한번에 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 정리
docker-compose down
```

## 🧪 테스트

### 1. 기본 연결 테스트

```bash
# 모든 transport 테스트
cd client_examples
python3 test_all_transports.py
```

### 2. 실제 사용 예제

```bash
# 다양한 사용 시나리오 데모
python3 usage_examples.py
```

### 3. 수동 테스트

#### STDIO 테스트
```python
import asyncio
from pyhub.llm.agents.mcp import MCPClient

async def test_stdio():
    config = {
        "command": "python3",
        "args": ["stdio/calculator_server.py"]
    }
    
    async with MCPClient(config).connect() as client:
        tools = await client.list_tools()
        print("Tools:", [t['name'] for t in tools])
        
        result = await client.execute_tool("add", {"a": 10, "b": 5})
        print("Result:", result)

asyncio.run(test_stdio())
```

#### 네트워크 테스트
```python
# SSE
config = {"url": "http://localhost:8001/mcp/sse"}

# WebSocket  
config = {"url": "ws://localhost:8002/mcp/ws"}

# HTTP
config = {"url": "http://localhost:8003/mcp"}
```

## 🔧 제공되는 도구

모든 서버는 동일한 계산기 도구들을 제공합니다:

| 도구 이름 | 설명 | 파라미터 |
|---------|------|----------|
| `add` | 두 숫자 더하기 | `a: float, b: float` |
| `subtract` | 두 숫자 빼기 | `a: float, b: float` |
| `multiply` | 두 숫자 곱하기 | `a: float, b: float` |
| `divide` | 두 숫자 나누기 | `a: float, b: float` |
| `calculate_expression` | 수식 계산 | `expression: str` |
| `get_time` | 현재 시간 | 없음 |
| `echo` | 메시지 반환 | `message: str` |

## 💡 pyhub-llm과 함께 사용하기

### 기본 사용법

```python
from pyhub.llm import LLMFactory
from pyhub.llm.agents.mcp import load_mcp_tools

# 1. MCP 도구 로드
tools = await load_mcp_tools({
    "command": "python3",
    "args": ["stdio/calculator_server.py"]
})

# 2. LLM과 함께 사용
llm = LLMFactory.create("gpt-4o-mini", tools=tools)

# 3. AI 질문
response = await llm.ask_async("25 + 17은 얼마인가요?")
print(response.text)
```

### 고급 사용법

```python
# 여러 서버 동시 사용
stdio_tools = await load_mcp_tools({
    "command": "python3", 
    "args": ["stdio/calculator_server.py"]
})

sse_tools = await load_mcp_tools({
    "url": "http://localhost:8001/mcp/sse"
})

# 모든 도구 결합
all_tools = stdio_tools + sse_tools

# LLM에 연결
llm = LLMFactory.create("claude-3-haiku", tools=all_tools)
```

### 특정 도구만 로드

```python
# 필터링해서 로드
calculator_tools = await load_mcp_tools(
    {"command": "python3", "args": ["stdio/calculator_server.py"]},
    filter_tools=["add", "multiply"]  # 이 도구들만 로드
)
```

## 🌐 Transport 별 특징

### STDIO
- **장점**: 설정이 간단, 안정적, 로컬 전용
- **단점**: 네트워크 접근 불가
- **용도**: 로컬 개발, 간단한 도구

### SSE (Server-Sent Events)
- **장점**: HTTP 기반, 방화벽 친화적, 단방향 스트리밍
- **단점**: 단방향 통신만 가능
- **용도**: 실시간 알림, 로그 스트리밍

### WebSocket
- **장점**: 양방향 실시간 통신, 낮은 지연시간
- **단점**: 방화벽/프록시 이슈 가능
- **용도**: 채팅, 게임, 실시간 협업

### HTTP
- **장점**: 표준 프로토콜, 캐싱 가능, 확장성 좋음
- **단점**: 상대적으로 높은 오버헤드
- **용도**: REST API, 마이크로서비스

## 🔍 문제 해결

### 서버가 시작되지 않는 경우

```bash
# 포트 사용 중인지 확인
lsof -i :8001  # SSE
lsof -i :8002  # WebSocket  
lsof -i :8003  # HTTP

# 의존성 설치 확인
pip list | grep -E "(mcp|fastapi|uvicorn)"
```

### 연결 오류

```bash
# 서버 상태 확인
curl http://localhost:8001/health  # SSE
curl http://localhost:8002/health  # WebSocket
curl http://localhost:8003/health  # HTTP

# 로그 확인
docker-compose logs [service-name]
```

### 도구 실행 오류

```python
# 디버그 모드로 실행
import logging
logging.basicConfig(level=logging.DEBUG)

# 도구 목록 확인
async with MCPClient(config).connect() as client:
    tools = await client.list_tools()
    for tool in tools:
        print(f"Tool: {tool['name']}")
        print(f"Description: {tool['description']}")
        print(f"Parameters: {tool['parameters']}")
```

## 📚 추가 예제

### 1. 대화형 계산기

```python
async def interactive_calculator():
    tools = await load_mcp_tools({
        "command": "python3",
        "args": ["stdio/calculator_server.py"]
    })
    
    llm = LLMFactory.create("gpt-4o-mini", tools=tools)
    
    while True:
        question = input("계산 질문: ")
        if question.lower() in ['quit', 'exit']:
            break
            
        response = await llm.ask_async(question)
        print(f"답변: {response.text}\n")
```

### 2. 배치 계산

```python
async def batch_calculations():
    tools = await load_mcp_tools({
        "command": "python3", 
        "args": ["stdio/calculator_server.py"]
    })
    
    llm = LLMFactory.create("gpt-4o-mini", tools=tools)
    
    calculations = [
        "15 + 25",
        "30 * 4", 
        "100 / 5",
        "(10 + 5) * 2"
    ]
    
    for calc in calculations:
        response = await llm.ask_async(f"Calculate: {calc}")
        print(f"{calc} = {response.text}")
```

### 3. 성능 테스트

```python
import time

async def performance_test():
    config = {"command": "python3", "args": ["stdio/calculator_server.py"]}
    
    start_time = time.time()
    
    tasks = []
    async with MCPClient(config).connect() as client:
        for i in range(10):
            task = client.execute_tool("add", {"a": i, "b": i*2})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"10개 계산 완료: {end_time - start_time:.2f}초")
```

## 🤝 기여하기

새로운 MCP 서버 예제나 개선사항이 있으시면:

1. 이 패턴을 따라 새 서버 구현
2. 테스트 코드 추가
3. README 업데이트
4. Pull Request 제출

## 📞 지원

- [pyhub-llm GitHub Issues](https://github.com/yourusername/pyhub-llm/issues)
- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [Discord 커뮤니티](#) (추후 추가)