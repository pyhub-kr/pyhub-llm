# 🤖 AI 비서 만들기

일상생활에서 자주 하는 질문들에 답하는 똑똑한 AI 비서를 만들어봅시다!

## 🎯 만들 것

```
나: 오늘 저녁 메뉴 추천해줘
AI: 오늘같이 쌀쌀한 날씨엔 따뜻한 김치찌개나 부대찌개는 어떠세요?

나: 10달러는 한국돈으로 얼마야?
AI: 현재 환율 기준으로 약 13,000원입니다.
```

## 📝 기본 AI 비서 만들기

### Step 1: 간단한 비서 클래스

```python
# ai_assistant.py
from pyhub.llm import LLM
from datetime import datetime

class AIAssistant:
    """일상 질문에 답하는 AI 비서"""
    
    def __init__(self, model="gpt-4o-mini"):
        # AI 모델 초기화
        self.ai = LLM.create(model)
        self.name = "도우미"  # 비서 이름
        
    def answer(self, question):
        """질문에 답변합니다"""
        # 현재 시간과 요일 정보 추가
        current_time = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
        weekday = ["월", "화", "수", "목", "금", "토", "일"][datetime.now().weekday()]
        
        # 컨텍스트와 함께 질문
        prompt = f"""
        현재 시각: {current_time} ({weekday}요일)
        질문: {question}
        
        친절하고 도움이 되는 답변을 해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용하기
assistant = AIAssistant()
print(assistant.answer("오늘 뭐 먹을까?"))
```

### Step 2: 특화된 기능 추가

```python
class SmartAssistant(AIAssistant):
    """더 똑똑한 AI 비서"""
    
    def recommend_food(self, meal_time="점심", preference=None):
        """식사 메뉴를 추천합니다"""
        prompt = f"{meal_time} 메뉴를 추천해주세요."
        
        if preference:
            prompt += f" {preference}을/를 선호합니다."
            
        # 계절 정보 추가
        month = datetime.now().month
        if month in [12, 1, 2]:
            prompt += " 겨울이라 따뜻한 음식이면 좋겠어요."
        elif month in [6, 7, 8]:
            prompt += " 여름이라 시원한 음식이면 좋겠어요."
            
        return self.answer(prompt)
    
    def calculate(self, expression):
        """계산을 도와줍니다"""
        prompt = f"다음을 계산해주세요: {expression}"
        return self.answer(prompt)
    
    def translate(self, text, target_language="영어"):
        """번역을 도와줍니다"""
        prompt = f"'{text}'를 {target_language}로 번역해주세요."
        return self.answer(prompt)
    
    def explain(self, topic):
        """어려운 개념을 쉽게 설명합니다"""
        prompt = f"{topic}에 대해 초등학생도 이해할 수 있게 설명해주세요."
        return self.answer(prompt)

# 사용 예시
assistant = SmartAssistant()

# 메뉴 추천
print("🍽️ 메뉴 추천:")
print(assistant.recommend_food("저녁", "한식"))

# 계산
print("\n🧮 계산:")
print(assistant.calculate("1달러가 1,300원일 때 50달러는 몇 원?"))

# 번역
print("\n🌏 번역:")
print(assistant.translate("좋은 하루 되세요", "일본어"))

# 설명
print("\n📚 설명:")
print(assistant.explain("광합성"))
```

## 🎨 대화형 AI 비서

### 인터랙티브 비서 만들기

```python
class InteractiveAssistant(SmartAssistant):
    """대화형 AI 비서"""
    
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.commands = {
            "메뉴": self.food_menu,
            "계산": self.calc_menu,
            "번역": self.trans_menu,
            "설명": self.explain_menu,
            "도움말": self.show_help
        }
    
    def start(self):
        """대화형 세션을 시작합니다"""
        print(f"안녕하세요! {self.name}입니다. 무엇을 도와드릴까요?")
        print("(종료하려면 '종료' 또는 'quit'을 입력하세요)")
        self.show_help()
        
        while True:
            user_input = input("\n👤 you: ").strip()
            
            if user_input.lower() in ['종료', 'quit', 'exit']:
                print(f"좋은 하루 되세요! 👋")
                break
            
            # 명령어 확인
            command_found = False
            for cmd, func in self.commands.items():
                if user_input.startswith(cmd):
                    func(user_input)
                    command_found = True
                    break
            
            # 일반 질문
            if not command_found:
                response = self.answer(user_input)
                print(f"🤖 {self.name}: {response}")
    
    def show_help(self):
        """도움말을 표시합니다"""
        print("\n📋 사용 가능한 명령어:")
        print("  메뉴 - 식사 메뉴 추천")
        print("  계산 - 수식이나 단위 변환")
        print("  번역 - 다국어 번역")
        print("  설명 - 어려운 개념 설명")
        print("  도움말 - 이 메시지 표시")
        print("  또는 자유롭게 질문하세요!")
    
    def food_menu(self, user_input):
        """메뉴 추천 처리"""
        print("🍽️ 메뉴 추천 모드")
        meal = input("어느 끼니인가요? (아침/점심/저녁): ")
        pref = input("선호하는 음식 종류가 있나요? (한식/중식/일식/양식 등): ")
        
        result = self.recommend_food(meal, pref)
        print(f"🤖 {self.name}: {result}")
    
    def calc_menu(self, user_input):
        """계산 처리"""
        print("🧮 계산 모드")
        expr = input("계산할 내용을 입력하세요: ")
        
        result = self.calculate(expr)
        print(f"🤖 {self.name}: {result}")
    
    def trans_menu(self, user_input):
        """번역 처리"""
        print("🌏 번역 모드")
        text = input("번역할 텍스트: ")
        lang = input("목표 언어 (영어/일본어/중국어 등): ")
        
        result = self.translate(text, lang)
        print(f"🤖 {self.name}: {result}")
    
    def explain_menu(self, user_input):
        """설명 처리"""
        print("📚 설명 모드")
        topic = input("설명이 필요한 주제: ")
        
        result = self.explain(topic)
        print(f"🤖 {self.name}: {result}")

# 실행하기
if __name__ == "__main__":
    assistant = InteractiveAssistant()
    assistant.start()
```

## 🔧 고급 기능 추가

### 1. 할 일 관리 기능

```python
class TaskAssistant(SmartAssistant):
    """할 일을 관리하는 AI 비서"""
    
    def __init__(self):
        super().__init__()
        self.tasks = []  # 할 일 목록
    
    def add_task(self, task):
        """할 일을 추가합니다"""
        self.tasks.append({
            "task": task,
            "created": datetime.now(),
            "completed": False
        })
        return f"✅ '{task}' 항목을 추가했습니다."
    
    def list_tasks(self):
        """할 일 목록을 보여줍니다"""
        if not self.tasks:
            return "📋 할 일이 없습니다!"
        
        result = "📋 할 일 목록:\n"
        for i, task in enumerate(self.tasks, 1):
            status = "✓" if task["completed"] else "○"
            result += f"{i}. [{status}] {task['task']}\n"
        
        return result
    
    def complete_task(self, task_num):
        """할 일을 완료 처리합니다"""
        if 1 <= task_num <= len(self.tasks):
            self.tasks[task_num-1]["completed"] = True
            return f"✅ 완료했습니다!"
        return "❌ 잘못된 번호입니다."
    
    def get_task_suggestion(self):
        """AI가 할 일을 제안합니다"""
        current_tasks = [t["task"] for t in self.tasks if not t["completed"]]
        
        prompt = f"""
        현재 할 일 목록: {current_tasks}
        시간: {datetime.now().strftime("%H시")}
        
        지금 해야 할 가장 중요한 일 1가지를 추천해주세요.
        """
        
        return self.answer(prompt)

# 사용 예시
task_assistant = TaskAssistant()

# 할 일 추가
print(task_assistant.add_task("Python 공부하기"))
print(task_assistant.add_task("운동 30분"))
print(task_assistant.add_task("책 읽기"))

# 목록 보기
print(task_assistant.list_tasks())

# AI 추천 받기
print("\n🤖 AI 추천:")
print(task_assistant.get_task_suggestion())
```

### 2. 일정 관리 기능

```python
class ScheduleAssistant(SmartAssistant):
    """일정을 관리하는 AI 비서"""
    
    def parse_schedule(self, text):
        """자연어로 된 일정을 파싱합니다"""
        prompt = f"""
        다음 텍스트에서 일정 정보를 추출해주세요:
        "{text}"
        
        다음 형식으로 답해주세요:
        - 날짜: YYYY-MM-DD
        - 시간: HH:MM
        - 제목: 일정 제목
        - 장소: 장소 (없으면 "없음")
        """
        
        return self.answer(prompt)
    
    def remind_schedule(self, schedules):
        """오늘의 일정을 알려줍니다"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
        오늘({today})의 일정입니다:
        {schedules}
        
        친절하게 오늘의 일정을 브리핑해주세요.
        준비할 것이나 주의사항도 함께 알려주세요.
        """
        
        return self.answer(prompt)

# 사용 예시
schedule_ai = ScheduleAssistant()

# 자연어 일정 파싱
text = "내일 오후 3시에 강남역에서 친구 만나기"
parsed = schedule_ai.parse_schedule(text)
print("📅 파싱된 일정:")
print(parsed)

# 일정 브리핑
schedules = """
- 09:00 팀 회의 (회의실 A)
- 12:00 점심 약속 (회사 근처 식당)
- 15:00 프로젝트 발표 (대회의실)
"""
briefing = schedule_ai.remind_schedule(schedules)
print("\n📢 오늘의 일정 브리핑:")
print(briefing)
```

## 💾 대화 기록 저장하기

```python
import json
from datetime import datetime

class MemoryAssistant(SmartAssistant):
    """대화를 기억하는 AI 비서"""
    
    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.load_history()
    
    def answer_with_memory(self, question):
        """대화 기록을 참고해서 답변합니다"""
        # 최근 대화 3개 가져오기
        recent = self.conversation_history[-3:] if self.conversation_history else []
        
        context = "이전 대화:\n"
        for conv in recent:
            context += f"Q: {conv['question']}\n"
            context += f"A: {conv['answer'][:50]}...\n\n"
        
        # 현재 질문 추가
        full_prompt = context + f"현재 질문: {question}"
        
        response = self.ai.ask(full_prompt)
        
        # 대화 저장
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": response.text
        })
        
        self.save_history()
        return response.text
    
    def save_history(self):
        """대화 기록을 파일로 저장합니다"""
        with open("conversation_history.json", "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
    
    def load_history(self):
        """저장된 대화 기록을 불러옵니다"""
        try:
            with open("conversation_history.json", "r", encoding="utf-8") as f:
                self.conversation_history = json.load(f)
        except FileNotFoundError:
            self.conversation_history = []
    
    def search_history(self, keyword):
        """대화 기록에서 키워드를 검색합니다"""
        results = []
        for conv in self.conversation_history:
            if keyword in conv["question"] or keyword in conv["answer"]:
                results.append(conv)
        
        return results

# 사용하기
memory_ai = MemoryAssistant()

# 대화하기
questions = [
    "파이썬을 배우고 싶어",
    "초보자가 보면 좋은 책 추천해줘",
    "아까 추천해준 책 중에 어떤 게 제일 좋아?"
]

for q in questions:
    print(f"\n👤 You: {q}")
    answer = memory_ai.answer_with_memory(q)
    print(f"🤖 AI: {answer[:100]}...")

# 대화 검색
print("\n🔍 '책' 관련 대화 검색:")
results = memory_ai.search_history("책")
for r in results:
    print(f"- {r['timestamp']}: {r['question']}")
```

## 🎯 실전 활용 예제

### 하루 일과 도우미

```python
def daily_assistant():
    """하루를 시작하고 마무리하는 AI 도우미"""
    assistant = SmartAssistant()
    
    # 아침 브리핑
    morning_prompt = """
    좋은 아침입니다! 오늘 하루를 시작하는 사람에게
    동기부여가 되는 메시지와 함께 다음을 포함해 브리핑해주세요:
    - 오늘의 날짜와 요일
    - 하루를 시작하는 팁 1가지
    - 긍정적인 한 마디
    """
    
    print("☀️ 아침 브리핑")
    print(assistant.answer(morning_prompt))
    
    # 저녁 정리
    evening_prompt = """
    수고하셨습니다! 하루를 마무리하는 사람에게
    다음을 포함한 메시지를 전해주세요:
    - 오늘 하루 돌아보기 유도
    - 내일을 위한 준비 팁
    - 편안한 밤 인사
    """
    
    print("\n🌙 저녁 마무리")
    print(assistant.answer(evening_prompt))

# 실행
daily_assistant()
```

## ✅ 핵심 정리

1. **AI 비서**는 일상 업무를 도와주는 똑똑한 도구
2. **클래스로 구조화**하면 기능 확장이 쉬움
3. **대화형 인터페이스**로 사용성 향상
4. **메모리 기능**으로 맥락 있는 대화 가능

## 🚀 다음 단계

AI 비서를 만들었으니, 이제 [텍스트 개선 도구](text-improver.md)를 만들어 글쓰기를 도와주는 AI를 만들어봅시다!