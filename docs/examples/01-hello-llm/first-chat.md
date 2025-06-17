# 🎈 첫 번째 AI 대화 - Hello World!

드디어 AI와 첫 대화를 나눌 시간입니다! 차근차근 따라해보세요.

## 🎯 이번에 만들 것

```
나: 안녕! 나는 파이썬을 배우는 학생이야.
AI: 안녕하세요! 파이썬을 배우고 계시는군요. 정말 좋은 선택이십니다...
```

## 📦 Step 1: pyhub-llm 설치하기

터미널을 열고 다음 명령어를 입력하세요:

```bash
# OpenAI 사용하려면
pip install "pyhub-llm[openai]"

# 또는 모든 AI 제공자 설치
pip install "pyhub-llm[all]"
```

### 설치 확인
```python
# Python 실행
python

# 설치 확인
import pyhub.llm
print("설치 성공! 🎉")
```

## 🔑 Step 2: API 키 설정하기

### 방법 1: 환경 변수로 설정 (추천)
```bash
# Mac/Linux
export OPENAI_API_KEY="sk-..."

# Windows
set OPENAI_API_KEY=sk-...
```

### 방법 2: .env 파일 만들기
1. 프로젝트 폴더에 `.env` 파일 생성
2. 다음 내용 입력:
```
OPENAI_API_KEY=sk-...
```

## 💬 Step 3: 첫 번째 코드 작성하기

`my_first_ai.py` 파일을 만들고 다음 코드를 입력하세요:

```python
# AI와 대화하는 첫 번째 프로그램
# 파일명: my_first_ai.py

# 1. 필요한 도구 가져오기
from pyhub.llm import LLM

# 2. AI 도우미 만들기
print("AI 도우미를 준비하고 있습니다...")
ai = LLM.create("gpt-4o-mini")  # GPT-4 미니 모델 사용

# 3. AI에게 인사하기
print("AI에게 질문하는 중...")
response = ai.ask("안녕! 나는 파이썬을 배우는 학생이야.")

# 4. AI의 답변 출력하기
print("\n🤖 AI의 답변:")
print(response.text)

# 5. 사용한 토큰과 비용 확인
print(f"\n📊 사용 정보:")
print(f"- 사용한 토큰: {response.usage.total_tokens}개")
print(f"- 예상 비용: 약 {response.usage.total_tokens * 0.002:.2f}원")
```

## 🏃 Step 4: 실행하기

```bash
python my_first_ai.py
```

### 예상 결과
```
AI 도우미를 준비하고 있습니다...
AI에게 질문하는 중...

🤖 AI의 답변:
안녕하세요! 파이썬을 배우고 계시는군요. 정말 좋은 선택이십니다! 
파이썬은 배우기 쉽고 강력한 프로그래밍 언어예요. 
궁금한 점이 있으면 언제든지 물어보세요. 함께 공부해요! 😊

📊 사용 정보:
- 사용한 토큰: 95개
- 예상 비용: 약 0.19원
```

## 🔍 코드 한 줄씩 이해하기

### 1. 도구 가져오기
```python
from pyhub.llm import LLM
```
- `pyhub.llm`: 우리가 설치한 패키지
- `LLM`: AI와 대화하는 도구

### 2. AI 만들기
```python
ai = LLM.create("gpt-4o-mini")
```
- `LLM.create()`: AI 도우미를 만드는 함수
- `"gpt-4o-mini"`: 사용할 AI 모델 이름
- `ai`: 만든 AI를 저장할 변수

### 3. 질문하기
```python
response = ai.ask("안녕! 나는 파이썬을 배우는 학생이야.")
```
- `ai.ask()`: AI에게 질문하는 함수
- `response`: AI의 답변을 저장할 변수

### 4. 답변 출력
```python
print(response.text)
```
- `response.text`: AI가 답한 텍스트
- `print()`: 화면에 출력

## 🎮 연습해보기

### 연습 1: 다른 질문해보기
```python
# 다양한 질문 시도
questions = [
    "오늘 점심 메뉴 추천해줘",
    "파이썬으로 뭘 만들 수 있어?",
    "1부터 10까지 더하면 얼마야?"
]

for question in questions:
    response = ai.ask(question)
    print(f"Q: {question}")
    print(f"A: {response.text}\n")
```

### 연습 2: 대화 이어가기
```python
# 연속 대화
response1 = ai.ask("내 이름은 철수야")
print("AI:", response1.text)

response2 = ai.ask("내 이름이 뭐라고 했지?")
print("AI:", response2.text)  # AI는 기억하지 못합니다!
```

### 연습 3: 다른 모델 사용해보기
```python
# 다른 모델들 시도
models = ["gpt-4o-mini", "gpt-3.5-turbo"]

for model_name in models:
    ai = LLM.create(model_name)
    response = ai.ask("너는 어떤 AI야?")
    print(f"{model_name}: {response.text[:50]}...")
```

## 🚨 자주 발생하는 오류와 해결법

### 1. "API key not found"
```python
# 해결법: API 키 직접 전달
ai = LLM.create("gpt-4o-mini", api_key="sk-...")
```

### 2. "Rate limit exceeded"
```python
# 해결법: 잠시 기다리기
import time
time.sleep(1)  # 1초 대기
```

### 3. "Model not found"
```python
# 해결법: 올바른 모델명 사용
# ❌ ai = LLM.create("gpt-4-mini")
# ✅ ai = LLM.create("gpt-4o-mini")
```

## 💡 팁과 트릭

### 1. 긴 답변 받기
```python
response = ai.ask(
    "파이썬의 장점을 5가지 알려줘",
    max_tokens=500  # 더 긴 답변 허용
)
```

### 2. 간단한 답변 받기
```python
response = ai.ask(
    "1+1은?",
    system="짧고 간단하게 답해주세요."
)
```

### 3. 답변 형식 지정하기
```python
response = ai.ask(
    "좋아하는 과일 3개를 알려줘",
    system="번호를 매겨서 답해주세요."
)
```

## ✅ 배운 내용 정리

이제 여러분은:
- ✅ pyhub-llm 설치할 수 있습니다
- ✅ AI 도우미를 만들 수 있습니다
- ✅ 질문하고 답변받을 수 있습니다
- ✅ 토큰과 비용을 확인할 수 있습니다

## 🚀 다음 단계

기본적인 대화를 성공했다면, [응답 이해하기](understanding-responses.md)로 넘어가서 AI의 답변을 더 깊이 이해해봅시다!