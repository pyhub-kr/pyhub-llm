# 🧠 AI가 대화를 기억하는 방법

AI가 어떻게 이전 대화를 기억하고 맥락을 이해하는지 알아봅시다!

## 🎯 대화 기억의 원리

### AI는 사실 기억하지 못합니다!

```python
from pyhub.llm import LLM

ai = LLM.create("gpt-4o-mini")

# 첫 번째 대화
response1 = ai.ask("내 이름은 김철수입니다")
print(response1.text)  # "안녕하세요, 김철수님!"

# 두 번째 대화 - AI는 이전 대화를 모릅니다!
response2 = ai.ask("내 이름이 뭐라고 했죠?")
print(response2.text)  # "죄송하지만, 이름을 알려주지 않으셨습니다."
```

### 그럼 어떻게 기억하게 만들까요?

**비밀은 "대화 기록을 함께 보내기"입니다!**

```python
# 대화 기록을 만들어서 함께 보냅니다
conversation = """
사용자: 내 이름은 김철수입니다
AI: 안녕하세요, 김철수님!
사용자: 내 이름이 뭐라고 했죠?
"""

response = ai.ask(conversation + "\nAI:")
print(response.text)  # "김철수님이라고 말씀하셨습니다."
```

## 📝 대화 기록 관리하기

### Step 1: 간단한 대화 기록 시스템

```python
class SimpleChat:
    """간단한 대화 기록 관리 시스템"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.ai = LLM.create(model)
        self.history = []  # 대화 기록을 저장할 리스트
    
    def chat(self, user_message):
        """사용자 메시지를 받아 AI 응답을 생성합니다"""
        
        # 1. 사용자 메시지를 기록에 추가
        self.history.append(f"사용자: {user_message}")
        
        # 2. 전체 대화 기록을 만들기
        full_conversation = "\n".join(self.history)
        
        # 3. AI에게 전체 대화를 보내기
        prompt = f"{full_conversation}\nAI:"
        response = self.ai.ask(prompt)
        
        # 4. AI 응답을 기록에 추가
        self.history.append(f"AI: {response.text}")
        
        return response.text
    
    def show_history(self):
        """대화 기록을 보여줍니다"""
        print("\n=== 대화 기록 ===")
        for message in self.history:
            print(message)
        print("================\n")

# 사용 예시
chat = SimpleChat()

# 대화하기
print("🤖:", chat.chat("안녕! 나는 파이썬을 배우는 학생이야"))
print("🤖:", chat.chat("내가 뭘 배운다고 했지?"))
print("🤖:", chat.chat("파이썬으로 뭘 만들 수 있어?"))

# 대화 기록 확인
chat.show_history()
```

### Step 2: 메시지 형식 개선하기

```python
from datetime import datetime

class ImprovedChat:
    """개선된 대화 시스템"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.ai = LLM.create(model)
        self.messages = []  # 구조화된 메시지 저장
    
    def add_message(self, role, content):
        """메시지를 추가합니다"""
        self.messages.append({
            "role": role,  # "user" 또는 "assistant"
            "content": content,
            "timestamp": datetime.now()
        })
    
    def chat(self, user_message):
        """대화를 진행합니다"""
        # 사용자 메시지 추가
        self.add_message("user", user_message)
        
        # OpenAI 형식으로 변환
        formatted_messages = []
        for msg in self.messages:
            if msg["role"] == "user":
                formatted_messages.append(f"Human: {msg['content']}")
            else:
                formatted_messages.append(f"Assistant: {msg['content']}")
        
        # 대화 생성
        conversation = "\n".join(formatted_messages)
        prompt = f"{conversation}\nAssistant:"
        
        # AI 응답 받기
        response = self.ai.ask(prompt)
        
        # 응답 저장
        self.add_message("assistant", response.text)
        
        return response.text
    
    def get_context_summary(self):
        """대화 맥락을 요약합니다"""
        if len(self.messages) < 2:
            return "대화가 막 시작되었습니다."
        
        # 최근 5개 메시지만 사용
        recent = self.messages[-5:]
        summary_prompt = "다음 대화를 한 줄로 요약해주세요:\n"
        
        for msg in recent:
            summary_prompt += f"{msg['role']}: {msg['content']}\n"
        
        summary = self.ai.ask(summary_prompt)
        return summary.text

# 사용 예시
chat = ImprovedChat()

# 대화 진행
responses = [
    chat.chat("안녕! 오늘 날씨가 정말 좋네"),
    chat.chat("산책하기 좋은 곳 추천해줄 수 있어?"),
    chat.chat("서울에 있는 곳으로 부탁해")
]

for i, response in enumerate(responses, 1):
    print(f"응답 {i}: {response}\n")

# 대화 요약
print("📝 대화 요약:", chat.get_context_summary())
```

## 🎨 고급 메모리 관리

### 1. 메모리 크기 제한하기

```python
class MemoryLimitedChat:
    """메모리 크기를 제한하는 챗봇"""
    
    def __init__(self, model="gpt-4o-mini", max_messages=10):
        self.ai = LLM.create(model)
        self.messages = []
        self.max_messages = max_messages  # 최대 메시지 수
        
    def chat(self, user_message):
        """메모리 크기를 제한하면서 대화합니다"""
        # 메시지 추가
        self.messages.append({"role": "user", "content": user_message})
        
        # 메모리 크기 초과 시 오래된 메시지 제거
        if len(self.messages) > self.max_messages:
            # 첫 번째 시스템 메시지는 유지하고 나머지 제거
            self.messages = self.messages[-(self.max_messages):]
        
        # 대화 생성
        conversation = self._format_conversation()
        response = self.ai.ask(conversation)
        
        # 응답 저장
        self.messages.append({"role": "assistant", "content": response.text})
        
        return response.text
    
    def _format_conversation(self):
        """대화를 포맷팅합니다"""
        formatted = []
        for msg in self.messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted) + "\nAssistant:"
    
    def get_memory_usage(self):
        """현재 메모리 사용량을 확인합니다"""
        total_chars = sum(len(msg["content"]) for msg in self.messages)
        return {
            "messages": len(self.messages),
            "characters": total_chars,
            "estimated_tokens": total_chars // 4  # 대략적인 토큰 수
        }

# 사용 예시
chat = MemoryLimitedChat(max_messages=6)

# 긴 대화 진행
for i in range(10):
    response = chat.chat(f"이것은 {i+1}번째 메시지입니다")
    print(f"메시지 {i+1}: {response[:50]}...")
    
    # 메모리 상태 확인
    usage = chat.get_memory_usage()
    print(f"  → 메모리: {usage['messages']}개 메시지, 약 {usage['estimated_tokens']} 토큰\n")
```

### 2. 중요한 정보만 기억하기

```python
class SmartMemoryChat:
    """중요한 정보만 기억하는 똑똑한 챗봇"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.ai = LLM.create(model)
        self.short_term_memory = []  # 단기 기억
        self.long_term_memory = {}   # 장기 기억 (중요 정보)
        
    def chat(self, user_message):
        """중요한 정보를 추출하면서 대화합니다"""
        # 중요 정보 추출
        important_info = self._extract_important_info(user_message)
        if important_info:
            self.long_term_memory.update(important_info)
        
        # 단기 기억에 추가
        self.short_term_memory.append({"role": "user", "content": user_message})
        
        # 맥락 생성
        context = self._create_context()
        response = self.ai.ask(context)
        
        # 응답 저장
        self.short_term_memory.append({"role": "assistant", "content": response.text})
        
        # 단기 기억 정리 (최근 5개만 유지)
        if len(self.short_term_memory) > 10:
            self.short_term_memory = self.short_term_memory[-10:]
        
        return response.text
    
    def _extract_important_info(self, message):
        """메시지에서 중요한 정보를 추출합니다"""
        extract_prompt = f"""
        다음 메시지에서 기억해야 할 중요한 정보를 추출해주세요:
        "{message}"
        
        예시: 이름, 나이, 직업, 취미, 선호사항 등
        없으면 "없음"이라고 답하세요.
        있으면 "키: 값" 형식으로 답하세요.
        """
        
        result = self.ai.ask(extract_prompt)
        
        # 결과 파싱
        if "없음" in result.text:
            return None
        
        # 간단한 파싱 (실제로는 더 정교하게)
        info = {}
        lines = result.text.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
        
        return info if info else None
    
    def _create_context(self):
        """대화 맥락을 생성합니다"""
        context_parts = []
        
        # 장기 기억 추가
        if self.long_term_memory:
            context_parts.append("기억하고 있는 정보:")
            for key, value in self.long_term_memory.items():
                context_parts.append(f"- {key}: {value}")
            context_parts.append("")
        
        # 최근 대화 추가
        context_parts.append("최근 대화:")
        for msg in self.short_term_memory:
            role = "Human" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        context_parts.append("Assistant:")
        return "\n".join(context_parts)
    
    def show_memory(self):
        """현재 기억 상태를 보여줍니다"""
        print("\n=== 장기 기억 ===")
        for key, value in self.long_term_memory.items():
            print(f"{key}: {value}")
        
        print(f"\n=== 단기 기억 ===")
        print(f"최근 {len(self.short_term_memory)}개 메시지 저장 중")

# 사용 예시
smart_chat = SmartMemoryChat()

# 정보가 포함된 대화
conversations = [
    "안녕! 내 이름은 김철수야",
    "나는 25살이고 개발자로 일하고 있어",
    "오늘 날씨 어때?",
    "내가 몇 살이라고 했지?",
    "프로그래밍 관련 책 추천해줄 수 있어?"
]

for msg in conversations:
    print(f"\n👤 You: {msg}")
    response = smart_chat.chat(msg)
    print(f"🤖 Bot: {response}")

# 메모리 상태 확인
smart_chat.show_memory()
```

### 3. 대화 요약을 통한 메모리 압축

```python
class CompressedMemoryChat:
    """대화를 요약해서 메모리를 효율적으로 사용하는 챗봇"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.ai = LLM.create(model)
        self.current_conversation = []
        self.conversation_summary = ""
        self.summary_threshold = 10  # 10개 메시지마다 요약
        
    def chat(self, user_message):
        """대화를 압축하면서 진행합니다"""
        # 현재 대화에 추가
        self.current_conversation.append(f"User: {user_message}")
        
        # 임계값 도달 시 요약
        if len(self.current_conversation) >= self.summary_threshold:
            self._compress_conversation()
        
        # 전체 맥락 생성
        context = self._build_context()
        context += f"\nUser: {user_message}\nAssistant:"
        
        # 응답 생성
        response = self.ai.ask(context)
        self.current_conversation.append(f"Assistant: {response.text}")
        
        return response.text
    
    def _compress_conversation(self):
        """현재 대화를 요약합니다"""
        # 이전 요약과 현재 대화를 합쳐서 새로운 요약 생성
        compress_prompt = f"""
        이전 대화 요약:
        {self.conversation_summary}
        
        최근 대화:
        {chr(10).join(self.current_conversation)}
        
        위 내용을 모두 포함하여 간단히 요약해주세요.
        중요한 정보는 반드시 포함시켜주세요.
        """
        
        new_summary = self.ai.ask(compress_prompt)
        self.conversation_summary = new_summary.text
        
        # 최근 2개 메시지만 남기고 나머지 삭제
        self.current_conversation = self.current_conversation[-2:]
        
        print(f"\n💾 메모리 압축 완료! (요약 길이: {len(self.conversation_summary)}자)")
    
    def _build_context(self):
        """대화 맥락을 구성합니다"""
        parts = []
        
        # 요약된 내용이 있으면 추가
        if self.conversation_summary:
            parts.append(f"이전 대화 요약:\n{self.conversation_summary}\n")
        
        # 현재 대화 추가
        if self.current_conversation:
            parts.append("최근 대화:")
            parts.extend(self.current_conversation)
        
        return "\n".join(parts)
    
    def get_memory_stats(self):
        """메모리 통계를 반환합니다"""
        return {
            "summary_length": len(self.conversation_summary),
            "current_messages": len(self.current_conversation),
            "total_chars": len(self.conversation_summary) + 
                          sum(len(msg) for msg in self.current_conversation)
        }

# 사용 예시
compressed_chat = CompressedMemoryChat()

# 긴 대화 시뮬레이션
topics = [
    "파이썬 학습", "웹 개발", "데이터베이스", "API 설계",
    "프론트엔드", "백엔드", "클라우드", "DevOps",
    "머신러닝", "딥러닝", "자연어처리", "컴퓨터비전"
]

for i, topic in enumerate(topics):
    print(f"\n--- 대화 {i+1}: {topic} ---")
    
    # 각 주제에 대해 대화
    response = compressed_chat.chat(f"{topic}에 대해 알려줘")
    print(f"Bot: {response[:100]}...")
    
    # 메모리 상태 확인
    stats = compressed_chat.get_memory_stats()
    print(f"메모리: {stats['current_messages']}개 메시지, 총 {stats['total_chars']}자")
```

## 📊 메모리 전략 비교

| 전략 | 장점 | 단점 | 사용 시기 |
|-----|------|------|----------|
| **전체 기록** | 완벽한 맥락 | 비용 증가 | 짧은 대화 |
| **크기 제한** | 비용 절약 | 오래된 정보 손실 | 일반적인 대화 |
| **중요 정보 추출** | 효율적 | 구현 복잡 | 장기 대화 |
| **대화 압축** | 균형적 | 요약 시 정보 손실 | 매우 긴 대화 |

## 🔧 실전 팁

### 1. 시스템 메시지 활용

```python
def create_chat_with_personality(personality):
    """특정 성격을 가진 챗봇을 만듭니다"""
    chat = ImprovedChat()
    
    # 시스템 메시지로 성격 설정
    system_message = f"당신은 {personality} 챗봇입니다. 이 성격에 맞게 대화하세요."
    chat.add_message("system", system_message)
    
    return chat

# 다양한 성격의 챗봇
friendly_bot = create_chat_with_personality("친절하고 유머러스한")
professional_bot = create_chat_with_personality("전문적이고 정확한")
```

### 2. 대화 저장과 복원

```python
import json

def save_conversation(chat, filename):
    """대화를 파일로 저장합니다"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chat.messages, f, ensure_ascii=False, default=str, indent=2)

def load_conversation(chat, filename):
    """저장된 대화를 불러옵니다"""
    with open(filename, 'r', encoding='utf-8') as f:
        chat.messages = json.load(f)
    return chat
```

## ✅ 핵심 정리

1. **AI는 스스로 기억하지 못함** - 우리가 대화 기록을 관리해야 함
2. **메모리 관리가 중요** - 비용과 성능의 균형
3. **다양한 전략 존재** - 상황에 맞는 방법 선택
4. **구조화된 데이터** - 체계적인 메시지 관리

## 🚀 다음 단계

대화 기억 방법을 배웠으니, 이제 [챗봇 만들기](chatbot-basics.md)에서 실제로 대화형 AI를 구현해봅시다!