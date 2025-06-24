# FastAPI + pyhub-llm 통합 예제

이 디렉토리는 pyhub-llm을 FastAPI와 연동하는 완전한 예제를 제공합니다. 기본적인 채팅 API부터 고급 인증, Rate Limiting, 번역/요약 서비스까지 실용적인 웹 서비스 구현 방법을 보여줍니다.

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
# 기본 의존성 설치
pip install -r requirements.txt

# 또는 pyhub-llm과 함께 설치
pip install "pyhub-llm[fastapi]"  # 향후 지원 예정
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# 필수 환경변수 설정
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 3. 기본 서비스 실행

```bash
# 기본 FastAPI 서비스 실행
python main.py

# 또는 uvicorn으로 직접 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 고급 서비스 실행

```bash
# 인증, Rate Limiting 등이 포함된 고급 서비스
python advanced.py
```

## 📂 파일 구조

```
fastapi/
├── README.md              # 이 파일
├── requirements.txt        # Python 의존성
├── .env.example           # 환경변수 템플릿
├── main.py                # 기본 FastAPI 애플리케이션
├── advanced.py            # 고급 기능 포함 애플리케이션
├── services/              # 비즈니스 로직 모듈
│   ├── __init__.py
│   ├── chat.py           # 채팅 서비스
│   └── translation.py    # 번역/요약 서비스
├── middleware/            # 미들웨어 모듈
│   ├── __init__.py
│   ├── auth.py           # 인증 미들웨어
│   └── rate_limit.py     # Rate Limiting
├── tests/                 # 테스트 파일들
├── client_examples/       # 클라이언트 사용 예제
├── docker-compose.yml     # Docker 배포 설정
└── Dockerfile            # 컨테이너 이미지
```

## 🌐 API 엔드포인트

### 기본 서비스 (main.py)

| 엔드포인트 | 메서드 | 설명 | 인증 필요 |
|-----------|--------|------|----------|
| `/` | GET | 서비스 정보 | ❌ |
| `/health` | GET | 헬스체크 | ❌ |
| `/chat` | POST | 단일 질문 처리 | ❌ |
| `/batch` | POST | 배치 처리 | ❌ |
| `/stream` | POST | 스트리밍 응답 | ❌ |
| `/chat/session` | POST | 세션 기반 채팅 | ❌ |

### 고급 서비스 (advanced.py)

| 엔드포인트 | 메서드 | 설명 | 인증 필요 |
|-----------|--------|------|----------|
| `/api/chat` | POST | 보호된 채팅 | ✅ |
| `/api/batch` | POST | 보호된 배치 처리 | ✅ |
| `/api/stream` | POST | 보호된 스트리밍 | ✅ |
| `/api/translate` | POST | 번역 서비스 | ✅ |
| `/api/summarize` | POST | 요약 서비스 | ✅ |
| `/api/supported-languages` | GET | 지원 언어 목록 | 🔶 |
| `/admin/stats` | GET | 서비스 통계 | ✅ (관리자) |

🔶 = 선택적 인증 (키가 있으면 더 많은 정보)

## 💻 사용 예제

### 1. 기본 채팅

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "안녕하세요! FastAPI와 pyhub-llm이 어떻게 연동되나요?",
    "model": "gpt-4o-mini"
  }'
```

### 2. 배치 처리

```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      "Python이란 무엇인가요?",
      "FastAPI의 장점은?",
      "LLM 배치 처리의 이점은?"
    ],
    "model": "gpt-4o-mini",
    "max_parallel": 3
  }'
```

### 3. 인증이 필요한 API (고급 서비스)

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "보안이 적용된 채팅입니다!",
    "model": "gpt-4o-mini"
  }'
```

### 4. 번역 서비스

```bash
curl -X POST "http://localhost:8000/api/translate" \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you today?",
    "target_language": "ko"
  }'
```

### 5. 요약 서비스

```bash
curl -X POST "http://localhost:8000/api/summarize" \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "매우 긴 텍스트를 여기에 입력하면 요약해드립니다...",
    "max_length": 100,
    "language": "ko"
  }'
```

## 🐍 Python 클라이언트 예제

```python
import httpx
import asyncio

class PyHubLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def chat(self, message: str, model: str = "gpt-4o-mini"):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat",
                headers=self.headers,
                json={"message": message, "model": model}
            )
            return response.json()
    
    async def batch(self, messages: list, model: str = "gpt-4o-mini"):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/batch",
                headers=self.headers,
                json={"messages": messages, "model": model}
            )
            return response.json()

# 사용 예제
async def main():
    client = PyHubLLMClient()
    
    # 단일 질문
    result = await client.chat("FastAPI와 pyhub-llm의 조합은 어떤가요?")
    print(result["response"])
    
    # 배치 처리
    questions = [
        "Python의 장점은?",
        "FastAPI가 인기 있는 이유는?",
        "LLM API 서비스 구축 시 고려사항은?"
    ]
    
    batch_result = await client.batch(questions)
    for i, response in enumerate(batch_result["responses"]):
        print(f"Q{i+1}: {questions[i]}")
        print(f"A{i+1}: {response['response']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔧 환경 변수 설정

### 필수 환경변수

```bash
# LLM API 키
OPENAI_API_KEY=your_openai_api_key_here
```

### 선택적 환경변수

```bash
# FastAPI 설정
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_RELOAD=true

# 인증 설정
ALLOWED_API_KEYS=key1,key2,key3
API_SECRET_KEY=your_secret_key
ADMIN_API_KEYS=admin_key1,admin_key2

# Rate Limiting
RATE_LIMIT_REQUESTS=100

# 로깅
LOG_LEVEL=info
```

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest tests/

# 특정 테스트 실행
pytest tests/test_main.py -v

# 커버리지 포함 테스트
pytest tests/ --cov=. --cov-report=html
```

## 🐳 Docker 배포

### 단일 컨테이너 실행

```bash
# Docker 이미지 빌드
docker build -t pyhub-llm-fastapi .

# 컨테이너 실행
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  pyhub-llm-fastapi
```

### Docker Compose 사용

```bash
# 전체 스택 실행 (FastAPI + Redis)
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 정리
docker-compose down
```

## 📊 성능 고려사항

### 1. LLM 인스턴스 캐싱
- 모델별로 LLM 인스턴스를 캐시하여 초기화 오버헤드 감소
- 메모리 사용량과 성능의 균형 고려

### 2. 배치 처리 최적화
- `max_parallel` 파라미터로 동시 요청 수 조절
- API 제공자의 Rate Limit 고려

### 3. Rate Limiting
- 기본값: 분당 100 요청
- 프로덕션 환경에서는 Redis 기반 Rate Limiter 권장

### 4. 세션 관리
- 현재는 인메모리 저장
- 프로덕션에서는 Redis나 데이터베이스 사용 권장

## 🔐 보안 고려사항

### 1. API 키 관리
- 환경변수로 API 키 관리
- 로그에 API 키 노출 방지

### 2. 인증 및 권한
- Bearer Token 방식의 API 키 인증
- 관리자 권한 분리

### 3. Rate Limiting
- DoS 공격 방지
- 사용자별 요청 제한

### 4. CORS 설정
- 프로덕션에서는 구체적인 도메인 설정 필요

## 🚀 프로덕션 배포 가이드

### 1. 환경 설정
```bash
# 프로덕션 환경변수
export FASTAPI_RELOAD=false
export LOG_LEVEL=warning
export RATE_LIMIT_REQUESTS=1000
```

### 2. 리버스 프록시 (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. 프로세스 관리 (Gunicorn)
```bash
gunicorn advanced:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## 🤝 기여하기

이 예제를 개선하거나 새로운 기능을 추가하고 싶다면:

1. 이슈를 생성하여 아이디어를 공유해주세요
2. Pull Request를 통해 코드를 기여해주세요
3. 문서 개선도 언제나 환영합니다

## 📝 라이선스

이 예제는 pyhub-llm과 동일한 라이선스를 따릅니다.

---

더 자세한 정보는 [pyhub-llm 공식 문서](https://github.com/pyhub-kr/pyhub-llm)를 참조해주세요.