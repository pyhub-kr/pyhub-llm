# 🔍 AI 응답 이해하기

AI가 주는 답변에는 텍스트 외에도 많은 정보가 담겨있습니다. 자세히 알아봅시다!

## 📦 응답 객체의 구조

AI의 응답은 단순한 텍스트가 아니라 여러 정보를 담은 "상자"입니다.

```python
from pyhub.llm import LLM

# AI에게 질문
ai = LLM.create("gpt-4o-mini")
response = ai.ask("안녕하세요!")

# response 안에는 뭐가 있을까?
print(type(response))  # <class 'CompletionResponse'>
```

### 응답에 포함된 정보들

```python
# 1. 답변 텍스트
print("답변:", response.text)

# 2. 사용량 정보
print("입력 토큰:", response.usage.prompt_tokens)
print("출력 토큰:", response.usage.completion_tokens)
print("총 토큰:", response.usage.total_tokens)

# 3. 모델 정보
print("사용한 모델:", response.model)

# 4. 원본 응답 (고급)
print("원본 데이터:", response.raw)
```

## 💰 토큰과 비용 이해하기

### 토큰이란?
토큰은 AI가 텍스트를 이해하는 단위입니다.

```python
# 토큰 예시 시각화
examples = [
    "안녕",        # 약 2토큰
    "Hello",       # 약 1토큰
    "안녕하세요",   # 약 5토큰
    "I love you",  # 약 3토큰
]

for text in examples:
    response = ai.ask(f"'{text}'는 몇 글자야?")
    print(f"'{text}' = {response.usage.prompt_tokens}토큰")
```

### 토큰 계산 규칙
- **한글**: 1글자 ≈ 2-3토큰
- **영어**: 1단어 ≈ 1-2토큰
- **숫자**: 1자리 ≈ 1토큰
- **특수문자**: 1개 ≈ 1토큰

### 비용 계산하기

```python
def calculate_cost(response, model="gpt-4o-mini"):
    """응답의 예상 비용을 계산합니다"""
    
    # 모델별 1,000토큰당 가격 (원화 기준)
    prices = {
        "gpt-4o-mini": {
            "input": 0.2,    # 입력 1,000토큰당 0.2원
            "output": 0.8    # 출력 1,000토큰당 0.8원
        },
        "gpt-4o": {
            "input": 5,      # 입력 1,000토큰당 5원
            "output": 15     # 출력 1,000토큰당 15원
        }
    }
    
    price = prices.get(model, prices["gpt-4o-mini"])
    
    # 비용 계산
    input_cost = (response.usage.prompt_tokens / 1000) * price["input"]
    output_cost = (response.usage.completion_tokens / 1000) * price["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# 사용 예시
response = ai.ask("파이썬으로 계산기 만드는 법 알려줘")
cost = calculate_cost(response)

print(f"💰 비용 분석:")
print(f"- 질문 비용: {cost['input_cost']:.3f}원")
print(f"- 답변 비용: {cost['output_cost']:.3f}원")
print(f"- 총 비용: {cost['total_cost']:.3f}원")
```

## 🎯 좋은 질문하는 방법

### 1. 명확하고 구체적으로

```python
# ❌ 나쁜 예
response = ai.ask("코드 써줘")

# ✅ 좋은 예
response = ai.ask("""
Python으로 간단한 계산기를 만들어줘.
- 더하기, 빼기, 곱하기, 나누기 기능
- 사용자 입력 받기
- 에러 처리 포함
""")
```

### 2. 맥락 제공하기

```python
# ❌ 나쁜 예
response = ai.ask("이거 고쳐줘")

# ✅ 좋은 예
response = ai.ask("""
다음 Python 코드에서 ZeroDivisionError가 발생합니다.
어떻게 고칠 수 있을까요?

def divide(a, b):
    return a / b
    
result = divide(10, 0)
""")
```

### 3. 원하는 형식 지정하기

```python
# 답변 형식을 지정하면 더 유용한 답변을 받을 수 있습니다
response = ai.ask("""
한국의 주요 도시 5개를 알려줘.
다음 형식으로 답해줘:
1. 도시명 - 인구수 - 특징
""")

print(response.text)
```

## 📊 응답 품질 개선하기

### 1. 시스템 프롬프트 활용

```python
# AI에게 역할 부여하기
response = ai.ask(
    "for 루프 설명해줘",
    system="당신은 초등학생도 이해할 수 있게 설명하는 친절한 선생님입니다."
)
```

### 2. 온도(Temperature) 조절

```python
# 온도: 0 = 일관된 답변, 1 = 창의적인 답변

# 사실적인 정보 (낮은 온도)
response = ai.ask(
    "한국의 수도는?",
    temperature=0.1
)

# 창의적인 답변 (높은 온도)
response = ai.ask(
    "외계인이 지구에 온다면 어떤 일이 일어날까?",
    temperature=0.9
)
```

### 3. 최대 토큰 설정

```python
# 짧은 답변
response = ai.ask(
    "인공지능이 뭐야?",
    max_tokens=50  # 약 20-30 글자
)

# 긴 답변
response = ai.ask(
    "인공지능의 역사를 자세히 설명해줘",
    max_tokens=1000  # 약 400-500 글자
)
```

## 🔧 디버깅과 문제 해결

### 응답 상태 확인하기

```python
def analyze_response(response):
    """응답을 분석하고 문제를 진단합니다"""
    
    print("🔍 응답 분석:")
    print(f"- 모델: {response.model}")
    print(f"- 답변 길이: {len(response.text)}자")
    print(f"- 토큰 사용량: {response.usage.total_tokens}")
    
    # 비용 계산
    cost = response.usage.total_tokens * 0.002
    print(f"- 예상 비용: {cost:.2f}원")
    
    # 답변 품질 체크
    if len(response.text) < 10:
        print("⚠️ 답변이 너무 짧습니다. 질문을 더 구체적으로 해보세요.")
    
    if response.usage.total_tokens > 1000:
        print("⚠️ 토큰을 많이 사용했습니다. 질문을 간단하게 해보세요.")
    
    return response

# 사용 예시
response = ai.ask("안녕")
analyze_response(response)
```

## 📝 실전 예제: 스마트 질문 도우미

```python
class SmartAI:
    """똑똑하게 질문하고 비용을 관리하는 AI 도우미"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.ai = LLM.create(model)
        self.total_cost = 0
        self.history = []
    
    def ask(self, question, save_money=True):
        """질문하고 비용을 추적합니다"""
        
        # 비용 절약 모드
        if save_money:
            # 짧은 답변 요청
            full_question = f"{question}\n(한 문단으로 간단히 답해주세요)"
            response = self.ai.ask(full_question, max_tokens=200)
        else:
            response = self.ai.ask(question)
        
        # 비용 계산
        cost = response.usage.total_tokens * 0.002
        self.total_cost += cost
        
        # 기록 저장
        self.history.append({
            "question": question,
            "answer": response.text,
            "tokens": response.usage.total_tokens,
            "cost": cost
        })
        
        return response
    
    def show_summary(self):
        """사용 요약을 보여줍니다"""
        print("\n📊 사용 요약:")
        print(f"- 총 질문 수: {len(self.history)}")
        print(f"- 총 비용: {self.total_cost:.2f}원")
        print(f"- 평균 비용: {self.total_cost/len(self.history):.2f}원")
        
        # 가장 비싼 질문
        expensive = max(self.history, key=lambda x: x['cost'])
        print(f"\n💸 가장 비싼 질문: {expensive['cost']:.2f}원")
        print(f"   '{expensive['question'][:30]}...'")

# 사용해보기
smart_ai = SmartAI()

# 여러 질문하기
questions = [
    "파이썬이 뭐야?",
    "리스트와 튜플의 차이점은?",
    "웹 스크래핑하는 방법 알려줘"
]

for q in questions:
    response = smart_ai.ask(q)
    print(f"\nQ: {q}")
    print(f"A: {response.text[:100]}...")

# 요약 보기
smart_ai.show_summary()
```

## ✅ 핵심 정리

1. **응답 객체**는 텍스트 외에도 많은 정보 포함
2. **토큰**은 AI가 텍스트를 처리하는 단위
3. **좋은 질문**이 좋은 답변을 만듭니다
4. **비용 관리**는 처음부터 습관화하세요

## 🚀 다음 단계

이제 AI 응답을 완벽히 이해했으니, [일상 작업 자동화](../02-everyday-tasks/)로 넘어가서 실용적인 예제들을 만들어봅시다!