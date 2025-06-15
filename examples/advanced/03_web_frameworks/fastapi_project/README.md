# FastAPI + pyhub-llm 통합 예제

FastAPI를 사용한 AI 챗봇 API 서버 예제입니다.

## 기능

- 💬 일반 채팅 API
- 🌊 스트리밍 채팅 API
- 📊 텍스트 분석 (감정, 요약, 키워드)
- 🖼️ 이미지 분석
- 🔢 텍스트 임베딩 생성
- 📝 대화 내역 관리
- 🔧 백그라운드 작업

## 설치

```bash
pip install -r requirements.txt
```

## 환경 설정

```bash
export OPENAI_API_KEY="your-api-key"
```

## 실행

```bash
# 개발 모드
uvicorn main:app --reload

# 프로덕션 모드
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API 문서

서버 실행 후 다음 주소에서 자동 생성된 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 사용 예시

### 채팅
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "안녕하세요!",
    "model": "gpt-4o-mini"
  }'
```

### 스트리밍 채팅
```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "파이썬에 대해 설명해주세요",
    "stream": true
  }'
```

### 텍스트 분석
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "오늘 정말 기분이 좋아요!",
    "analysis_type": "sentiment"
  }'
```

### 이미지 분석
```bash
curl -X POST "http://localhost:8000/analyze/image" \
  -F "file=@image.jpg" \
  -F 'request={"question":"이 이미지에 무엇이 있나요?"}'
```

## 주요 엔드포인트

- `GET /` - API 정보
- `POST /chat` - 일반 채팅
- `POST /chat/stream` - 스트리밍 채팅
- `POST /analyze` - 텍스트 분석
- `POST /analyze/image` - 이미지 분석
- `POST /embeddings` - 임베딩 생성
- `GET /conversations/{id}` - 대화 조회
- `DELETE /conversations/{id}` - 대화 삭제
- `POST /tasks/create` - 백그라운드 작업
- `GET /health` - 헬스체크