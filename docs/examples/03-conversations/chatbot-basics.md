# 🤖 간단한 챗봇 만들기

이제 본격적으로 대화형 챗봇을 만들어봅시다! 기본부터 고급 기능까지 단계별로 구현해보겠습니다.

## 🎯 만들 것

```
🤖 안녕하세요! 무엇을 도와드릴까요?
👤 파이썬 공부하는 방법을 알려줘
🤖 파이썬을 효과적으로 공부하는 방법을 알려드릴게요!
   1. 기초 문법부터 시작하세요...
```

## 📝 기본 챗봇 구현

### Step 1: 가장 간단한 챗봇

```python
# simple_chatbot.py
from pyhub.llm import LLM

class SimpleChatbot:
    """가장 기본적인 챗봇"""
    
    def __init__(self, name="도우미"):
        self.name = name
        self.ai = LLM.create("gpt-4o-mini")
        self.messages = []
        
    def greet(self):
        """인사말을 출력합니다"""
        greeting = f"""
🤖 안녕하세요! 저는 {self.name}입니다.
무엇이든 물어보세요. 도와드리겠습니다!
(종료하려면 'quit' 또는 '종료'를 입력하세요)
        """
        print(greeting)
        
    def chat(self, user_input):
        """사용자 입력을 받아 응답합니다"""
        # 대화 기록에 추가
        self.messages.append({"role": "user", "content": user_input})
        
        # 전체 대화 맥락 생성
        context = self._build_context()
        
        # AI 응답 생성
        response = self.ai.ask(context)
        ai_message = response.text
        
        # 응답을 대화 기록에 추가
        self.messages.append({"role": "assistant", "content": ai_message})
        
        return ai_message
    
    def _build_context(self):
        """대화 맥락을 구성합니다"""
        context = f"당신은 친절한 AI 어시스턴트 {self.name}입니다.\n\n"
        
        for msg in self.messages:
            if msg["role"] == "user":
                context += f"사용자: {msg['content']}\n"
            else:
                context += f"{self.name}: {msg['content']}\n"
                
        context += f"{self.name}:"
        return context
    
    def run(self):
        """챗봇을 실행합니다"""
        self.greet()
        
        while True:
            # 사용자 입력 받기
            user_input = input("\n👤 You: ").strip()
            
            # 종료 조건
            if user_input.lower() in ['quit', '종료', 'exit']:
                print(f"\n🤖 {self.name}: 안녕히 가세요! 좋은 하루 되세요! 👋")
                break
            
            # 응답 생성 및 출력
            response = self.chat(user_input)
            print(f"\n🤖 {self.name}: {response}")

# 실행
if __name__ == "__main__":
    bot = SimpleChatbot("파이썬 도우미")
    bot.run()
```

### Step 2: 기능이 추가된 챗봇

```python
class FeatureRichChatbot(SimpleChatbot):
    """다양한 기능을 가진 챗봇"""
    
    def __init__(self, name="스마트 봇"):
        super().__init__(name)
        self.commands = {
            "/help": self.show_help,
            "/clear": self.clear_history,
            "/save": self.save_conversation,
            "/stats": self.show_stats
        }
        
    def chat(self, user_input):
        """명령어를 처리하거나 일반 대화를 진행합니다"""
        # 명령어 확인
        if user_input.startswith("/"):
            return self.handle_command(user_input)
        
        # 일반 대화
        return super().chat(user_input)
    
    def handle_command(self, command):
        """명령어를 처리합니다"""
        cmd = command.split()[0]
        if cmd in self.commands:
            return self.commands[cmd]()
        else:
            return f"알 수 없는 명령어입니다. /help를 입력해 도움말을 확인하세요."
    
    def show_help(self):
        """도움말을 표시합니다"""
        return """
📋 사용 가능한 명령어:
/help - 이 도움말 표시
/clear - 대화 기록 초기화
/save - 대화 저장
/stats - 대화 통계 보기
        """
    
    def clear_history(self):
        """대화 기록을 초기화합니다"""
        self.messages = []
        return "대화 기록이 초기화되었습니다. 🗑️"
    
    def save_conversation(self):
        """대화를 파일로 저장합니다"""
        from datetime import datetime
        
        filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== {self.name} 대화 기록 ===\n")
            f.write(f"일시: {datetime.now()}\n\n")
            
            for msg in self.messages:
                role = "You" if msg["role"] == "user" else self.name
                f.write(f"{role}: {msg['content']}\n\n")
        
        return f"대화가 {filename}에 저장되었습니다. 💾"
    
    def show_stats(self):
        """대화 통계를 보여줍니다"""
        user_messages = [m for m in self.messages if m["role"] == "user"]
        bot_messages = [m for m in self.messages if m["role"] == "assistant"]
        
        total_user_chars = sum(len(m["content"]) for m in user_messages)
        total_bot_chars = sum(len(m["content"]) for m in bot_messages)
        
        return f"""
📊 대화 통계:
- 총 메시지: {len(self.messages)}개
- 사용자 메시지: {len(user_messages)}개 ({total_user_chars}자)
- 봇 메시지: {len(bot_messages)}개 ({total_bot_chars}자)
- 평균 응답 길이: {total_bot_chars // max(len(bot_messages), 1)}자
        """

# 사용 예시
bot = FeatureRichChatbot()
# bot.run()  # 실행하면 대화형 모드로 진입
```

## 🎨 특화된 챗봇 만들기

### 1. 학습 도우미 챗봇

```python
class StudyBuddyBot(SimpleChatbot):
    """학습을 도와주는 챗봇"""
    
    def __init__(self):
        super().__init__("학습 도우미")
        self.subject = None
        self.study_mode = None
        
    def greet(self):
        """학습 모드 선택"""
        print(f"""
📚 안녕하세요! {self.name}입니다.
어떤 공부를 도와드릴까요?

1. 개념 설명
2. 문제 풀이
3. 요약 정리
4. 퀴즈 만들기

원하는 번호를 선택하거나 자유롭게 질문하세요!
        """)
    
    def chat(self, user_input):
        """학습 모드에 따라 다르게 응답"""
        # 모드 선택
        if user_input in ["1", "2", "3", "4"]:
            modes = {
                "1": "개념 설명",
                "2": "문제 풀이", 
                "3": "요약 정리",
                "4": "퀴즈"
            }
            self.study_mode = modes[user_input]
            return f"{self.study_mode} 모드를 선택하셨습니다. 무엇을 공부하시나요?"
        
        # 학습 모드별 프롬프트 조정
        if self.study_mode:
            context_addon = f"\n현재 {self.study_mode} 모드입니다. 이에 맞게 답변해주세요."
        else:
            context_addon = ""
        
        # 기본 대화 + 학습 컨텍스트
        self.messages.append({"role": "user", "content": user_input})
        
        context = f"""당신은 친절한 학습 도우미입니다.
학생이 이해하기 쉽게 설명하고, 격려하며 도와주세요.{context_addon}

"""
        
        for msg in self.messages:
            if msg["role"] == "user":
                context += f"학생: {msg['content']}\n"
            else:
                context += f"도우미: {msg['content']}\n"
        
        context += "도우미:"
        
        response = self.ai.ask(context)
        self.messages.append({"role": "assistant", "content": response.text})
        
        return response.text
    
    def create_quiz(self, topic, num_questions=5):
        """주제에 대한 퀴즈를 생성합니다"""
        prompt = f"""
{topic}에 대한 퀴즈 {num_questions}개를 만들어주세요.

형식:
Q1. 질문
A1. 답변

Q2. 질문  
A2. 답변

난이도는 초급에서 중급 수준으로 만들어주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def explain_concept(self, concept, level="beginner"):
        """개념을 설명합니다"""
        levels = {
            "beginner": "초등학생도 이해할 수 있게",
            "intermediate": "중고등학생 수준으로",
            "advanced": "대학생 수준으로"
        }
        
        level_desc = levels.get(level, level)
        
        prompt = f"{concept}을 {level_desc} 설명해주세요. 예시를 포함해주세요."
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
study_bot = StudyBuddyBot()

# 개념 설명
print("📖 개념 설명:")
print(study_bot.explain_concept("재귀함수", "beginner"))

# 퀴즈 생성
print("\n📝 퀴즈:")
print(study_bot.create_quiz("파이썬 리스트", 3))
```

### 2. 고객 서비스 챗봇

```python
class CustomerServiceBot(SimpleChatbot):
    """고객 서비스 챗봇"""
    
    def __init__(self, company_name="우리 회사"):
        super().__init__(f"{company_name} 고객센터")
        self.company_name = company_name
        self.customer_info = {}
        self.ticket_number = None
        
    def greet(self):
        """고객 서비스 인사말"""
        print(f"""
🏢 {self.company_name} 고객센터입니다.
어떤 도움이 필요하신가요?

자주 묻는 질문:
1. 주문 조회
2. 환불/교환
3. 제품 문의
4. 기타 문의

번호를 선택하거나 직접 문의사항을 입력해주세요.
        """)
    
    def chat(self, user_input):
        """고객 응대"""
        # 초기 정보 수집
        if not self.customer_info.get("name"):
            self.customer_info["name"] = user_input
            return "감사합니다. 연락 가능한 전화번호를 알려주세요."
        
        if not self.customer_info.get("phone"):
            self.customer_info["phone"] = user_input
            return f"""
{self.customer_info['name']}님, 확인되었습니다.
어떤 문제로 연락주셨나요? 자세히 설명해주세요.
            """
        
        # 일반 상담 진행
        self.messages.append({"role": "user", "content": user_input})
        
        context = f"""당신은 {self.company_name}의 친절한 고객 서비스 상담원입니다.
고객 정보:
- 이름: {self.customer_info.get('name', '미확인')}
- 전화: {self.customer_info.get('phone', '미확인')}

공손하고 전문적으로 응대하며, 고객의 문제를 해결하도록 도와주세요.

대화 내용:
"""
        
        for msg in self.messages:
            if msg["role"] == "user":
                context += f"고객: {msg['content']}\n"
            else:
                context += f"상담원: {msg['content']}\n"
        
        context += "상담원:"
        
        response = self.ai.ask(context)
        self.messages.append({"role": "assistant", "content": response.text})
        
        # 티켓 생성 제안
        if len(self.messages) > 6 and not self.ticket_number:
            self.ticket_number = self._generate_ticket()
            response.text += f"\n\n📋 상담 내용을 접수했습니다. 접수번호: {self.ticket_number}"
        
        return response.text
    
    def _generate_ticket(self):
        """티켓 번호를 생성합니다"""
        from datetime import datetime
        import random
        
        date = datetime.now().strftime("%Y%m%d")
        rand = random.randint(1000, 9999)
        return f"TK{date}{rand}"
    
    def escalate_to_human(self):
        """상담원 연결"""
        return """
🙋 실제 상담원과 연결해드리겠습니다.
예상 대기 시간: 약 5분
연결되면 알림을 보내드리겠습니다.
대기 중에도 계속 문의사항을 입력하실 수 있습니다.
        """

# 사용 예시
cs_bot = CustomerServiceBot("테크샵")
# cs_bot.run()  # 실행하면 고객 서비스 모드로 진입
```

### 3. 창의적 대화 챗봇

```python
class CreativeChatbot(SimpleChatbot):
    """창의적인 대화를 하는 챗봇"""
    
    def __init__(self):
        super().__init__("창의봇")
        self.personality_traits = {
            "humor": 0.7,      # 유머 수준
            "creativity": 0.9,  # 창의성
            "formality": 0.3   # 격식
        }
        
    def chat(self, user_input):
        """창의적인 응답을 생성합니다"""
        self.messages.append({"role": "user", "content": user_input})
        
        # 성격 특성을 프롬프트에 반영
        personality_prompt = f"""
당신은 다음과 같은 성격을 가진 AI입니다:
- 유머 감각: {'높음' if self.personality_traits['humor'] > 0.6 else '보통'}
- 창의성: {'매우 높음' if self.personality_traits['creativity'] > 0.8 else '높음'}
- 격식: {'캐주얼' if self.personality_traits['formality'] < 0.5 else '격식있음'}

이런 성격에 맞게 재미있고 창의적으로 대답하세요.
가끔 은유나 비유를 사용하고, 상상력을 발휘하세요.
        """
        
        context = personality_prompt + "\n\n대화:\n"
        
        for msg in self.messages[-10:]:  # 최근 10개만
            if msg["role"] == "user":
                context += f"사용자: {msg['content']}\n"
            else:
                context += f"창의봇: {msg['content']}\n"
        
        context += "창의봇:"
        
        # 창의성 온도 설정
        response = self.ai.ask(
            context, 
            temperature=self.personality_traits['creativity']
        )
        
        self.messages.append({"role": "assistant", "content": response.text})
        return response.text
    
    def tell_story(self, theme):
        """주제에 맞는 짧은 이야기를 만듭니다"""
        prompt = f"""
'{theme}'를 주제로 짧고 재미있는 이야기를 만들어주세요.
200자 이내로 창의적이고 예상치 못한 전개를 포함하세요.
        """
        
        response = self.ai.ask(prompt, temperature=0.9)
        return f"📖 즉석 이야기: {theme}\n\n{response.text}"
    
    def play_word_game(self, start_word):
        """끝말잇기나 연상 게임을 합니다"""
        prompt = f"""
'{start_word}'로 시작하는 창의적인 단어 연상 게임을 해봅시다.
5개의 연관 단어를 제시하고 각각 재미있게 설명해주세요.
        """
        
        response = self.ai.ask(prompt, temperature=0.8)
        return response.text

# 사용 예시
creative_bot = CreativeChatbot()

# 이야기 만들기
print("📚 즉석 이야기:")
print(creative_bot.tell_story("AI와 고양이"))

# 단어 게임
print("\n🎮 단어 연상 게임:")
print(creative_bot.play_word_game("파이썬"))
```

## 🔧 챗봇 개선 기법

### 1. 응답 시간 표시

```python
import time

class TimedChatbot(SimpleChatbot):
    """응답 시간을 측정하는 챗봇"""
    
    def chat(self, user_input):
        start_time = time.time()
        
        # 타이핑 효과
        print("🤖 생각 중", end="", flush=True)
        for _ in range(3):
            time.sleep(0.5)
            print(".", end="", flush=True)
        print()
        
        # 실제 응답 생성
        response = super().chat(user_input)
        
        # 응답 시간 계산
        elapsed = time.time() - start_time
        print(f"⏱️ 응답 시간: {elapsed:.2f}초")
        
        return response
```

### 2. 감정 분석 챗봇

```python
class EmotionalChatbot(SimpleChatbot):
    """사용자의 감정을 인식하는 챗봇"""
    
    def analyze_emotion(self, text):
        """텍스트의 감정을 분석합니다"""
        prompt = f"""
다음 텍스트의 감정을 분석해주세요:
"{text}"

다음 중 하나로만 답하세요: 긍정적, 부정적, 중립적
        """
        
        response = self.ai.ask(prompt)
        return response.text.strip()
    
    def chat(self, user_input):
        # 감정 분석
        emotion = self.analyze_emotion(user_input)
        
        # 감정에 따른 이모티콘
        emotion_emoji = {
            "긍정적": "😊",
            "부정적": "😔", 
            "중립적": "😐"
        }
        
        # 감정 표시
        print(f"감지된 감정: {emotion} {emotion_emoji.get(emotion, '')}")
        
        # 감정을 고려한 응답
        self.messages.append({
            "role": "user",
            "content": user_input,
            "emotion": emotion
        })
        
        context = f"""
사용자의 현재 감정 상태는 '{emotion}'입니다.
이를 고려하여 공감하며 적절히 응답하세요.

대화:
"""
        
        for msg in self.messages[-5:]:
            if msg["role"] == "user":
                context += f"사용자: {msg['content']}\n"
            else:
                context += f"봇: {msg['content']}\n"
        
        context += "봇:"
        
        response = self.ai.ask(context)
        self.messages.append({"role": "assistant", "content": response.text})
        
        return response.text
```

### 3. 다국어 지원 챗봇

```python
class MultilingualChatbot(SimpleChatbot):
    """다국어를 지원하는 챗봇"""
    
    def __init__(self):
        super().__init__("글로벌 봇")
        self.language = "Korean"
        self.supported_languages = {
            "Korean": "한국어",
            "English": "영어",
            "Japanese": "일본어",
            "Chinese": "중국어"
        }
    
    def detect_language(self, text):
        """언어를 감지합니다"""
        prompt = f"""
다음 텍스트의 언어를 감지해주세요: "{text}"
Korean, English, Japanese, Chinese 중 하나로만 답하세요.
        """
        
        response = self.ai.ask(prompt)
        detected = response.text.strip()
        
        if detected in self.supported_languages:
            self.language = detected
        
        return detected
    
    def chat(self, user_input):
        # 언어 감지
        detected_lang = self.detect_language(user_input)
        
        # 언어별 응답
        self.messages.append({"role": "user", "content": user_input})
        
        context = f"""
{self.supported_languages[self.language]}로 응답하세요.
사용자가 사용한 언어와 같은 언어로 답변하세요.

대화:
"""
        
        for msg in self.messages[-5:]:
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            else:
                context += f"Bot: {msg['content']}\n"
        
        context += "Bot:"
        
        response = self.ai.ask(context)
        self.messages.append({"role": "assistant", "content": response.text})
        
        return f"[{self.supported_languages[self.language]}] {response.text}"
```

## 📊 챗봇 성능 모니터링

```python
class MonitoredChatbot(SimpleChatbot):
    """성능을 모니터링하는 챗봇"""
    
    def __init__(self):
        super().__init__("모니터봇")
        self.metrics = {
            "total_messages": 0,
            "total_tokens": 0,
            "response_times": [],
            "user_satisfaction": []
        }
    
    def chat(self, user_input):
        import time
        
        start = time.time()
        response = super().chat(user_input)
        elapsed = time.time() - start
        
        # 메트릭 업데이트
        self.metrics["total_messages"] += 2  # 사용자 + 봇
        self.metrics["response_times"].append(elapsed)
        
        # 토큰 추정 (실제로는 API 응답에서 가져와야 함)
        estimated_tokens = len(user_input + response) // 4
        self.metrics["total_tokens"] += estimated_tokens
        
        return response
    
    def ask_satisfaction(self):
        """만족도를 묻습니다"""
        print("\n이번 대화는 만족스러우셨나요? (1-5점)")
        try:
            score = int(input("점수: "))
            if 1 <= score <= 5:
                self.metrics["user_satisfaction"].append(score)
                return "피드백 감사합니다!"
        except:
            pass
        return "다음에 평가해주세요!"
    
    def show_metrics(self):
        """성능 지표를 보여줍니다"""
        avg_response = sum(self.metrics["response_times"]) / max(len(self.metrics["response_times"]), 1)
        avg_satisfaction = sum(self.metrics["user_satisfaction"]) / max(len(self.metrics["user_satisfaction"]), 1)
        
        return f"""
📊 챗봇 성능 지표:
- 총 메시지: {self.metrics['total_messages']}개
- 총 토큰 사용: {self.metrics['total_tokens']}개
- 평균 응답 시간: {avg_response:.2f}초
- 평균 만족도: {avg_satisfaction:.1f}/5.0
- 예상 비용: ${self.metrics['total_tokens'] * 0.000002:.4f}
        """
```

## ✅ 핵심 정리

1. **기본 구조** - 메시지 관리, 맥락 구성, 응답 생성
2. **특화 기능** - 목적에 맞는 챗봇 설계
3. **사용자 경험** - 명령어, 감정 인식, 다국어 지원
4. **성능 관리** - 메트릭 추적, 최적화

## 🚀 다음 단계

챗봇을 만들었으니, 이제 [대화 잘하는 법](conversation-tips.md)에서 더 나은 대화를 위한 팁을 알아봅시다!