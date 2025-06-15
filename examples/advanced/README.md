# 고급 예제 (Advanced Examples)

pyhub-llm의 고급 기능을 활용한 실전 예제들입니다.

## 🎯 학습 목표

- 임베딩과 유사도 검색 구현
- LLM 체이닝과 파이프라인 구축
- 웹 프레임워크 통합 (FastAPI, Django, Streamlit)
- 프로덕션 레벨 에러 처리 패턴
- 실전 애플리케이션 개발

## 📚 예제 목록

### 1. 임베딩 (01_embeddings.py)
- 텍스트 임베딩 생성
- 유사도 계산 및 검색
- 문서 클러스터링
- 의미 기반 검색 시스템

### 2. 체이닝 (02_chaining.py)
- 순차적 LLM 체인
- 병렬 처리 체인
- 조건부 체인
- 맵-리듀스 패턴

### 3. 웹 프레임워크 통합 (03_web_frameworks/)

#### FastAPI 프로젝트
- RESTful API 엔드포인트
- 스트리밍 응답
- 비동기 처리
- WebSocket 지원

#### Django 프로젝트
- 모델 기반 대화 관리
- REST API (DRF)
- 관리자 인터페이스
- 이미지 분석 API

#### Streamlit 프로젝트
- 대화형 챗봇 UI
- 실시간 스트리밍
- 텍스트/이미지 분석
- 다양한 도구 통합

### 4. 고급 에러 처리 (04_advanced_error_handling.py)
- 중앙집중식 에러 핸들러
- 재시도 전략 (지수 백오프)
- 서킷 브레이커 패턴
- 속도 제한 (Rate Limiting)
- 폴백 시스템
- 프로덕션 래퍼

### 5. 실전 예제 (05_practical_examples/)

#### 고급 챗봇 (chatbot.py)
- 컨텍스트 관리
- 도구 사용
- 대화 내보내기
- 사용자 선호도 기억

#### 문서 요약기 (document_summarizer.py)
- 다양한 파일 형식 지원 (PDF, DOCX, HTML)
- 섹션별 요약
- 핵심 포인트 추출
- 문서 구조 분석

#### 코드 리뷰어 (code_reviewer.py)
- Python 코드 분석
- 스타일/보안/성능 검사
- AI 기반 리뷰
- 품질 점수 및 개선 제안

#### 고급 번역기 (translator.py)
- 컨텍스트 인식 번역
- 다국어 지원
- 용어집 관리
- 번역 품질 검사

#### Q&A 시스템 (qa_system.py)
- 지식 베이스 관리
- 임베딩 기반 검색
- 대화형 질문 답변
- 답변 신뢰도 평가

## 🚀 실행 방법

### 기본 설정
```bash
# 환경 변수 설정
export OPENAI_API_KEY="your-api-key"

# 패키지 설치
pip install -r requirements.txt
```

### 개별 예제 실행
```bash
# 임베딩 예제
python 01_embeddings.py

# 체이닝 예제
python 02_chaining.py

# 에러 처리 예제
python 04_advanced_error_handling.py
```

### 웹 프레임워크 실행

#### FastAPI
```bash
cd 03_web_frameworks/fastapi_project
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Django
```bash
cd 03_web_frameworks/django_project
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

#### Streamlit
```bash
cd 03_web_frameworks/streamlit_project
pip install -r requirements.txt
streamlit run app.py
```

### 실전 예제 실행
```bash
cd 05_practical_examples

# 챗봇
python chatbot.py

# 문서 요약기
python document_summarizer.py

# 코드 리뷰어
python code_reviewer.py

# 번역기
python translator.py

# Q&A 시스템
python qa_system.py
```

## 💡 주요 개념

### 임베딩
- 텍스트를 벡터로 변환하여 의미적 유사도 계산
- 검색, 분류, 클러스터링에 활용

### LLM 체이닝
- 여러 LLM 호출을 연결하여 복잡한 작업 수행
- 파이프라인 패턴으로 모듈화

### 프로덕션 패턴
- 에러 처리, 재시도, 캐싱, 모니터링
- 확장 가능하고 안정적인 시스템 구축

## 📌 활용 시나리오

1. **기업 챗봇**: 고객 지원, 내부 문서 검색
2. **콘텐츠 관리**: 문서 요약, 번역, 분류
3. **개발 도구**: 코드 리뷰, 문서화, 테스트 생성
4. **교육 플랫폼**: Q&A 시스템, 학습 도우미
5. **연구 도구**: 논문 분석, 데이터 처리

## 🔍 추가 학습

- [OpenAI Embeddings 문서](https://platform.openai.com/docs/guides/embeddings)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [Streamlit 문서](https://docs.streamlit.io/)

## ⚠️ 주의사항

- API 사용량과 비용을 모니터링하세요
- 프로덕션 환경에서는 적절한 보안 설정을 추가하세요
- 대용량 처리 시 배치 처리와 캐싱을 활용하세요
- 에러 처리와 로깅을 철저히 구현하세요