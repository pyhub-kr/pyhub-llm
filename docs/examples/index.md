# 🚀 pyhub-llm 예제 가이드

AI와 대화하는 프로그램을 만들어보고 싶으신가요? 이 가이드는 Python 기초만 알고 있는 분들을 위해 준비했습니다.

## 📚 이 가이드를 읽기 전에 알아야 할 것

- Python 기초 (변수, 함수, 반복문, 조건문)
- 텍스트 에디터 사용법 (VS Code, PyCharm 등)
- 터미널/명령 프롬프트 기본 사용법

**몰라도 되는 것들:**
- ❌ langchain, OpenAI API에 대한 지식
- ❌ 머신러닝이나 AI에 대한 전문 지식
- ❌ 웹 개발 경험

## 🗺️ 학습 경로

### 🌱 초보자 코스 (1-3일)
1. **[시작하기 전에](00-before-you-start/)** - 준비물과 기본 개념
2. **[첫 AI 대화](01-hello-llm/)** - Hello World부터 시작하기
3. **[일상 작업 자동화](02-everyday-tasks/)** - 번역, 요약 등 실용적인 예제

### 🌿 중급자 코스 (1주일)
4. **[대화 이어가기](03-conversations/)** - 간단한 챗봇 만들기
5. **[파일 다루기](04-working-with-files/)** - 이미지와 문서 처리
6. **[정형화된 데이터](05-structured-data/)** - 체계적인 응답 받기

### 🌳 고급자 코스 (2주일)
7. **[다양한 AI 모델](06-multiple-providers/)** - OpenAI, Claude, Ollama 활용
8. **[비용 절약하기](07-saving-money/)** - 효율적인 AI 사용법
9. **[실전 프로젝트](08-real-projects/)** - 완성된 애플리케이션 만들기

## 💡 이 가이드의 특징

### 📝 모든 코드에 한글 주석
```python
# AI 도우미를 만듭니다
assistant = LLM.create("gpt-4o-mini")

# 질문을 합니다
response = assistant.ask("안녕하세요!")

# 답변을 출력합니다
print(response.text)
```

### 🎯 실행 가능한 완전한 예제
모든 예제는 복사-붙여넣기만으로 바로 실행할 수 있습니다.

### 🔍 예상 결과 포함
```
실행 결과:
안녕하세요! 무엇을 도와드릴까요?
```

### 💰 비용 정보
각 예제마다 대략적인 API 사용 비용을 표시합니다.
```
💰 예상 비용: 약 0.01원 (10토큰)
```

## 🚦 시작하기

1. **[Python 설치 확인하기](00-before-you-start/python-basics.md)**
2. **[API 키 받기](00-before-you-start/api-keys-explained.md)**
3. **[첫 번째 예제 실행하기](01-hello-llm/first-chat.md)**

## ❓ 도움이 필요하신가요?

- **[자주 묻는 질문](troubleshooting/)**
- **[일반적인 오류 해결](troubleshooting/common-errors.md)**
- **[커뮤니티 지원](09-next-steps/community-resources.md)**

---

🎉 **환영합니다!** AI 프로그래밍의 세계로 첫 발을 내딛으신 것을 축하합니다. 이제 시작해볼까요?