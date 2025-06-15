# Streamlit + pyhub-llm 통합 예제

Streamlit을 사용한 대화형 AI 챗봇 웹 애플리케이션입니다.

## 기능

### 💬 채팅
- 실시간 스트리밍 응답
- 대화 내역 저장 및 관리
- 여러 대화 세션 지원
- 대화 내보내기 (JSON)

### 📊 텍스트 분석
- 감정 분석
- 텍스트 요약
- 키워드 추출
- 개체명 인식

### 🖼️ 이미지 분석
- 이미지 업로드 및 분석
- 커스텀 질문 지원

### 🔧 도구
- 번역기
- 코드 생성
- SQL 쿼리 생성
- 정규식 생성

### ⚙️ 설정
- 모델 선택 (GPT-4, GPT-3.5 등)
- Temperature 조절
- Max Tokens 설정
- 시스템 프롬프트 커스터마이징

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
# 기본 실행
streamlit run app.py

# 포트 지정
streamlit run app.py --server.port 8080

# 외부 접속 허용
streamlit run app.py --server.address 0.0.0.0
```

## 사용법

1. **채팅 탭**
   - 메시지 입력창에 질문 입력
   - Enter 키로 전송
   - 스트리밍 모드로 실시간 응답 확인

2. **분석 탭**
   - 텍스트 입력
   - 원하는 분석 옵션 선택
   - "분석 시작" 버튼 클릭

3. **이미지 탭**
   - 이미지 파일 업로드
   - 질문 입력 (선택사항)
   - "이미지 분석" 버튼 클릭

4. **도구 탭**
   - 원하는 도구 선택
   - 필요한 정보 입력
   - 해당 버튼 클릭

## 주요 기능 설명

### 대화 관리
- 여러 대화를 동시에 관리
- 대화별 제목 설정
- 대화 삭제 기능
- JSON 형식으로 내보내기

### 토큰 통계
- 실시간 토큰 사용량 추적
- 대화별 토큰 사용 기록
- 전체 통계 표시

### 스트리밍 응답
- 실시간으로 AI 응답 표시
- 타이핑 효과로 자연스러운 UX
- 응답 중단 가능

## 커스터마이징

### 테마 변경
`.streamlit/config.toml` 파일 생성:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### 페이지 아이콘 변경
`app.py`의 `st.set_page_config`에서 `page_icon` 파라미터 수정

## 배포

### Streamlit Cloud
1. GitHub에 코드 푸시
2. [share.streamlit.io](https://share.streamlit.io) 접속
3. 저장소 연결 및 배포

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 트러블슈팅

### API 키 오류
환경 변수가 제대로 설정되었는지 확인:
```bash
echo $OPENAI_API_KEY
```

### 메모리 사용량
대화가 길어지면 메모리 사용량이 증가할 수 있습니다.
주기적으로 오래된 대화를 삭제하세요.

### 속도 개선
- 작은 모델 사용 (gpt-3.5-turbo)
- 스트리밍 모드 활성화
- Max Tokens 제한