# 대화 관리

pyhub-llm의 대화 히스토리 관리와 컨텍스트 제어 기능을 알아봅니다.

## 대화 히스토리 자동 관리

### 기본 대화 이어가기

pyhub-llm은 자동으로 대화 히스토리를 관리합니다:

```python
from pyhub.llm import LLM

llm = LLM.create("gpt-4o-mini")

# 첫 번째 대화
llm.ask("제 이름은 김철수입니다. 파이썬을 배우고 있어요.")

# 이전 대화를 기억하고 답변
reply = llm.ask("제가 누구라고 했죠?")
print(reply.text)  # "김철수님이라고 하셨습니다."

reply = llm.ask("제가 무엇을 배우고 있다고 했나요?")
print(reply.text)  # "파이썬을 배우고 계시다고 하셨습니다."
```

### 대화 히스토리 확인

```python
# 전체 대화 히스토리 확인
print(f"총 메시지 수: {len(llm.history)}")

# 대화 내용 출력
for msg in llm.history:
    role = msg["role"]
    content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
    print(f"{role}: {content}")
```

### 대화 초기화

```python
# 전체 대화 초기화
llm.clear()
print(f"대화 초기화 후 메시지 수: {len(llm.history)}")  # 0

# 시스템 프롬프트는 유지됨
llm = LLM.create("gpt-4o-mini", system_prompt="당신은 친절한 상담사입니다.")
llm.ask("안녕하세요")
llm.clear()  # 시스템 프롬프트는 유지, 대화만 초기화
```

## Stateless 모드

반복적인 독립 작업에는 Stateless 모드를 사용하여 성능을 최적화할 수 있습니다:

### 기본 사용법

```python
# Stateless 모드 - 대화 히스토리 저장 안 함
stateless_llm = LLM.create("gpt-4o-mini", stateless=True)

# 각 요청이 독립적으로 처리됨
stateless_llm.ask("제 이름은 김철수입니다.")
reply = stateless_llm.ask("제 이름이 뭐라고 했죠?")
print(reply.text)  # 이전 대화를 기억하지 못함
```

### 대량 처리에 적합

```python
# 감정 분석기 - 각 분석이 독립적
analyzer = LLM.create("gpt-4o-mini", stateless=True)

reviews = [
    "이 제품 정말 좋아요! 강력 추천합니다.",
    "배송이 너무 늦어서 실망했습니다.",
    "가격대비 괜찮은 것 같아요.",
    "다시는 구매하지 않을 것 같습니다."
]

# 각 리뷰를 독립적으로 분석
for review in reviews:
    reply = analyzer.ask(
        f"다음 리뷰의 감정을 분석하세요: '{review}'",
        choices=["긍정", "부정", "중립"]
    )
    print(f"{review[:20]}... → {reply.choice}")
```

### Stateless vs 일반 모드 비교

```python
import time

# 성능 비교
texts = ["텍스트 " + str(i) for i in range(100)]

# 일반 모드 (히스토리 누적)
normal_llm = LLM.create("gpt-4o-mini")
start = time.time()
for text in texts[:10]:  # 처음 10개만
    normal_llm.ask(f"다음을 요약하세요: {text}")
normal_time = time.time() - start

# Stateless 모드 (히스토리 없음)
stateless_llm = LLM.create("gpt-4o-mini", stateless=True)
start = time.time()
for text in texts[:10]:  # 처음 10개만
    stateless_llm.ask(f"다음을 요약하세요: {text}")
stateless_time = time.time() - start

print(f"일반 모드: {normal_time:.2f}초")
print(f"Stateless 모드: {stateless_time:.2f}초")
print(f"성능 향상: {(normal_time - stateless_time) / normal_time * 100:.1f}%")
```

## 컨텍스트 관리

### 수동 히스토리 제어

```python
# 특정 시점의 대화만 사용
llm = LLM.create("gpt-4o-mini")

# 여러 대화 진행
llm.ask("파이썬에 대해 알려주세요")
llm.ask("장점은 무엇인가요?")
checkpoint = len(llm.history)  # 체크포인트 저장

llm.ask("자바스크립트는 어떤가요?")
llm.ask("차이점은 무엇인가요?")

# 체크포인트로 롤백
llm.history = llm.history[:checkpoint]
reply = llm.ask("단점은 무엇인가요?")  # 파이썬의 단점을 물어봄
```

### 컨텍스트 윈도우 관리

```python
class ContextWindowManager:
    def __init__(self, llm, max_messages=10):
        self.llm = llm
        self.max_messages = max_messages
    
    def ask(self, prompt, **kwargs):
        # 최대 메시지 수 유지
        if len(self.llm.history) > self.max_messages:
            # 시스템 메시지 + 최근 메시지만 유지
            system_msgs = [m for m in self.llm.history if m["role"] == "system"]
            recent_msgs = self.llm.history[-(self.max_messages - len(system_msgs)):]
            self.llm.history = system_msgs + recent_msgs
        
        return self.llm.ask(prompt, **kwargs)

# 사용
llm = LLM.create("gpt-4o-mini")
manager = ContextWindowManager(llm, max_messages=6)

# 긴 대화를 해도 최근 6개 메시지만 유지
for i in range(20):
    manager.ask(f"질문 {i+1}")
    print(f"현재 히스토리 크기: {len(llm.history)}")
```

## 대화 저장과 복원

### JSON으로 저장/로드

```python
import json

def save_conversation(llm, filename):
    """대화를 JSON 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'model': llm.model,
            'history': llm.history,
            'timestamp': time.time()
        }, f, ensure_ascii=False, indent=2)

def load_conversation(filename, model=None):
    """JSON 파일에서 대화 로드"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    llm = LLM.create(model or data['model'])
    llm.history = data['history']
    return llm

# 사용 예시
llm = LLM.create("gpt-4o-mini")
llm.ask("안녕하세요")
llm.ask("파이썬에 대해 알려주세요")

# 저장
save_conversation(llm, "conversation.json")

# 나중에 로드
restored_llm = load_conversation("conversation.json")
reply = restored_llm.ask("방금 무엇에 대해 얘기했나요?")
print(reply.text)  # 파이썬에 대한 대화를 기억
```

### 대화 요약 및 압축

```python
def summarize_conversation(llm, keep_recent=4):
    """긴 대화를 요약하여 압축"""
    if len(llm.history) <= keep_recent + 1:  # +1 for system prompt
        return
    
    # 요약할 메시지들
    to_summarize = llm.history[1:-keep_recent]  # 시스템 제외, 최근 제외
    
    # 요약 생성
    summary_prompt = "다음 대화를 핵심만 간단히 요약해주세요:\n\n"
    for msg in to_summarize:
        summary_prompt += f"{msg['role']}: {msg['content']}\n"
    
    # Stateless로 요약 (현재 대화에 영향 없음)
    summarizer = LLM.create(llm.model, stateless=True)
    summary = summarizer.ask(summary_prompt).text
    
    # 히스토리 재구성
    system_msgs = [m for m in llm.history if m["role"] == "system"]
    summary_msg = {"role": "assistant", "content": f"[이전 대화 요약]\n{summary}"}
    recent_msgs = llm.history[-keep_recent:]
    
    llm.history = system_msgs + [summary_msg] + recent_msgs

# 사용
llm = LLM.create("gpt-4o-mini")

# 긴 대화 시뮬레이션
for i in range(20):
    llm.ask(f"질문 {i+1}에 대해 설명해주세요")

print(f"압축 전 메시지 수: {len(llm.history)}")
summarize_conversation(llm)
print(f"압축 후 메시지 수: {len(llm.history)}")
```

## 멀티턴 대화 패턴

### 단계별 정보 수집

```python
def collect_user_info(llm):
    """대화형으로 사용자 정보 수집"""
    info = {}
    
    # 이름
    reply = llm.ask("안녕하세요! 성함이 어떻게 되시나요?")
    info['name'] = llm.ask("감사합니다. 성함을 다시 한 번 말씀해 주시겠어요?").text
    
    # 관심사
    llm.ask("어떤 프로그래밍 언어에 관심이 있으신가요?")
    interests = llm.ask(
        "관심있는 언어를 하나 선택해주세요",
        choices=["Python", "JavaScript", "Java", "Go", "Rust"]
    ).choice
    info['interest'] = interests
    
    # 경험
    experience = llm.ask(
        f"{interests}를 얼마나 사용해보셨나요?",
        choices=["처음", "1년 미만", "1-3년", "3년 이상"]
    ).choice
    info['experience'] = experience
    
    # 요약
    summary = llm.ask(
        f"제가 이해한 바로는 {info['name']}님은 {interests}에 관심이 있고 "
        f"{experience}의 경험이 있으시군요. 맞나요?"
    )
    
    return info

# 사용
llm = LLM.create("gpt-4o-mini", 
    system_prompt="당신은 친절한 프로그래밍 교육 상담사입니다."
)
user_info = collect_user_info(llm)
print(user_info)
```

### 컨텍스트 기반 추천

```python
class ContextualRecommender:
    def __init__(self, model="gpt-4o-mini"):
        self.llm = LLM.create(model)
        self.preferences = []
    
    def add_preference(self, item, liked: bool):
        """사용자 선호도 추가"""
        action = "좋아한다" if liked else "싫어한다"
        self.llm.ask(f"사용자가 {item}을/를 {action}고 했습니다.")
        self.preferences.append((item, liked))
    
    def get_recommendation(self):
        """컨텍스트 기반 추천"""
        if not self.preferences:
            return "선호도 정보가 없습니다."
        
        return self.llm.ask(
            "지금까지의 선호도를 바탕으로 사용자가 좋아할 만한 것을 추천해주세요."
        ).text
    
    def explain_recommendation(self):
        """추천 이유 설명"""
        return self.llm.ask(
            "왜 그것을 추천했는지 이유를 설명해주세요."
        ).text

# 사용
recommender = ContextualRecommender()

# 선호도 입력
recommender.add_preference("파이썬", True)
recommender.add_preference("복잡한 문법", False)
recommender.add_preference("데이터 분석", True)
recommender.add_preference("웹 개발", True)

# 추천 받기
recommendation = recommender.get_recommendation()
print(f"추천: {recommendation}")

explanation = recommender.explain_recommendation()
print(f"이유: {explanation}")
```

## 대화 분기 처리

### 조건부 대화 흐름

```python
class ConversationFlow:
    def __init__(self, model="gpt-4o-mini"):
        self.llm = LLM.create(model)
        self.state = "start"
    
    def handle_conversation(self, user_input=None):
        if self.state == "start":
            reply = self.llm.ask(
                "프로그래밍 학습에 대해 궁금한 점이 있나요?",
                choices=["언어 선택", "학습 방법", "프로젝트 아이디어", "기타"]
            )
            
            if reply.choice == "언어 선택":
                self.state = "language_selection"
            elif reply.choice == "학습 방법":
                self.state = "learning_method"
            elif reply.choice == "프로젝트 아이디어":
                self.state = "project_ideas"
            else:
                self.state = "other"
            
            return self.handle_conversation()
        
        elif self.state == "language_selection":
            level = self.llm.ask(
                "프로그래밍 경험이 어느 정도 되시나요?",
                choices=["완전 초보", "기초 수준", "중급", "고급"]
            ).choice
            
            if level in ["완전 초보", "기초 수준"]:
                return self.llm.ask(
                    "초보자에게는 Python이나 JavaScript를 추천합니다. "
                    "어떤 분야에 관심이 있으신가요?"
                ).text
            else:
                return self.llm.ask(
                    "경험이 있으시군요! 어떤 분야의 언어를 찾고 계신가요?"
                ).text
        
        # ... 다른 상태들 처리

# 사용
flow = ConversationFlow()
response = flow.handle_conversation()
print(response)
```

## 성능 최적화 팁

### 1. 적절한 모드 선택

```python
# 대화형 애플리케이션
chatbot = LLM.create("gpt-4o-mini")  # 기본 모드

# 독립적 작업 처리
processor = LLM.create("gpt-4o-mini", stateless=True)  # Stateless 모드

# 하이브리드 접근
class HybridAssistant:
    def __init__(self):
        self.conversation_llm = LLM.create("gpt-4o-mini")
        self.utility_llm = LLM.create("gpt-4o-mini", stateless=True)
    
    def chat(self, message):
        """대화형 응답"""
        return self.conversation_llm.ask(message).text
    
    def analyze(self, text):
        """독립적 분석"""
        return self.utility_llm.ask(f"분석: {text}").text
```

### 2. 메모리 효율적 관리

```python
# 주기적 정리
def cleanup_old_conversations(llm, max_age_messages=50):
    if len(llm.history) > max_age_messages:
        # 오래된 메시지 제거
        llm.history = llm.history[-max_age_messages:]

# 중요 정보만 보존
def preserve_important_only(llm):
    important_keywords = ["이름", "목표", "중요", "기억"]
    
    preserved = []
    for msg in llm.history:
        if any(keyword in msg["content"] for keyword in important_keywords):
            preserved.append(msg)
    
    # 시스템 메시지 + 중요 메시지 + 최근 메시지
    system_msgs = [m for m in llm.history if m["role"] == "system"]
    recent_msgs = llm.history[-4:]  # 최근 4개
    
    llm.history = system_msgs + preserved + recent_msgs
```

## 다음 단계

- [프로바이더](providers.md) - 각 LLM 프로바이더의 특징
- [구조화된 출력](structured-output.md) - 정형화된 데이터 처리
- [고급 기능](advanced.md) - 스트리밍, 비동기 등 고급 패턴