# 초급 예제

이 디렉토리는 pyhub-llm의 기본적인 사용법을 보여주는 초급 예제들을 포함합니다.

## 예제 목록

1. **01_hello_world.py** - 첫 번째 LLM 프로그램
   - LLM 인스턴스 생성
   - 간단한 질문과 응답
   - 토큰 사용량 확인

2. **02_streaming.py** - 스트리밍 응답
   - 실시간으로 응답 받기
   - 진행 상태 표시
   - 스트리밍 데이터 처리

3. **03_conversation.py** - 대화 관리
   - 대화 내역 유지
   - 시스템 메시지 설정
   - 대화형 챗봇 구현

4. **04_file_processing.py** - 파일 처리
   - 텍스트 파일 읽기 및 요약
   - 이미지 파일 분석
   - 여러 파일 동시 처리

## 필수 요구사항

### 환경 변수 설정
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 패키지 설치
```bash
pip install pyhub-llm
```

## 실행 방법

각 예제를 직접 실행할 수 있습니다:

```bash
# Hello World 예제 실행
python 01_hello_world.py

# 스트리밍 예제 실행
python 02_streaming.py

# 대화 관리 예제 실행
python 03_conversation.py

# 파일 처리 예제 실행
python 04_file_processing.py
```

## 학습 순서

1. **01_hello_world.py**부터 시작하여 기본 개념을 익히세요
2. **02_streaming.py**로 실시간 응답 처리를 배우세요
3. **03_conversation.py**로 대화 관리 방법을 학습하세요
4. **04_file_processing.py**로 파일 처리 기능을 익히세요

## 문제 해결

### API 키 오류
```text
⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.
```
위 메시지가 나오면 OpenAI API 키를 환경 변수로 설정해야 합니다.

### 파일 찾기 오류
파일 처리 예제에서 이미지를 분석하려면 `sample_image.jpg` 또는 `sample_image.png` 파일을 준비해주세요.

## 다음 단계

초급 예제를 마스터했다면 [중급 예제](../intermediate/README.md)로 진행하세요!