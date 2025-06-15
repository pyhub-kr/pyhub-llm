# Django + pyhub-llm 통합 예제

Django를 사용한 AI 챗봇 웹 애플리케이션 예제입니다.

## 기능

- 💬 대화 관리 (생성, 조회, 삭제)
- 📝 메시지 저장 및 조회
- 🌊 스트리밍 채팅
- 📊 텍스트 분석 (감정, 요약, 키워드)
- 🖼️ 이미지 분석 및 저장
- 📈 사용 통계
- 🔐 사용자 인증 지원

## 설치

```bash
# 패키지 설치
pip install -r requirements.txt

# 데이터베이스 마이그레이션
python manage.py makemigrations
python manage.py migrate

# 관리자 계정 생성 (선택사항)
python manage.py createsuperuser

# 정적 파일 수집 (프로덕션)
python manage.py collectstatic
```

## 환경 설정

```bash
export OPENAI_API_KEY="your-api-key"
```

## 실행

```bash
# 개발 서버 실행
python manage.py runserver

# 프로덕션 (gunicorn 사용)
pip install gunicorn
gunicorn chatbot.wsgi:application --bind 0.0.0.0:8000
```

## API 엔드포인트

### 대화 관리
- `GET /api/conversations/` - 대화 목록
- `POST /api/conversations/` - 새 대화 생성
- `GET /api/conversations/{id}/` - 대화 상세
- `DELETE /api/conversations/{id}/` - 대화 삭제
- `POST /api/conversations/{id}/send_message/` - 메시지 전송
- `GET /api/conversations/{id}/messages/` - 메시지 목록

### 분석
- `POST /api/analyze/text/` - 텍스트 분석
- `POST /api/image-analysis/` - 이미지 분석

### 스트리밍
- `POST /api/chat/stream/` - 스트리밍 채팅

### 통계
- `GET /api/stats/` - 사용 통계

## 사용 예시

### 대화 생성 및 메시지 전송
```python
import requests

# 대화 생성
response = requests.post('http://localhost:8000/api/conversations/', 
    json={'title': '새 대화'})
conversation = response.json()

# 메시지 전송
response = requests.post(
    f'http://localhost:8000/api/conversations/{conversation["id"]}/send_message/',
    json={'message': '안녕하세요!', 'model': 'gpt-4o-mini'}
)
print(response.json())
```

### 텍스트 분석
```python
response = requests.post('http://localhost:8000/api/analyze/text/',
    json={
        'text': '오늘 날씨가 정말 좋네요!',
        'analysis_type': 'sentiment'
    })
print(response.json())
```

### 이미지 분석
```python
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/api/image-analysis/',
        files={'image': f},
        data={'question': '이 이미지에 무엇이 있나요?'})
print(response.json())
```

## 관리자 페이지

Django 관리자 페이지에서 데이터를 관리할 수 있습니다:
- http://localhost:8000/admin/

## 프로젝트 구조

```text
django_project/
├── chatbot/          # 프로젝트 설정
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── chat/             # 채팅 앱
│   ├── models.py     # 데이터 모델
│   ├── views.py      # 뷰 로직
│   ├── serializers.py # API 시리얼라이저
│   └── urls.py       # URL 라우팅
├── manage.py         # Django 관리 스크립트
└── requirements.txt  # 패키지 의존성
```