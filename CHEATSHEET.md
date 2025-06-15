# pyhub-llm CHEATSHEET

## 목차
- [설치](#설치)
- [기본 사용법](#기본-사용법)
- [스트리밍](#스트리밍)
- [구조화된 출력](#구조화된-출력)
- [분류 및 선택](#분류-및-선택)
- [비동기 처리](#비동기-처리)
- [캐싱](#캐싱)
- [대화 관리](#대화-관리)
- [파일 처리](#파일-처리)
- [임베딩](#임베딩)
- [템플릿 활용](#템플릿-활용)
- [History Backup](#history-backup)
- [MCP 통합](#mcp-통합)
- [웹 프레임워크 통합](#웹-프레임워크-통합)
- [도구/함수 호출](#도구함수-호출)
- [체이닝](#체이닝)
- [에러 처리](#에러-처리)
- [실용적인 예제](#실용적인-예제)

## 설치

```bash
# 전체 설치 (모든 프로바이더)
pip install "pyhub-llm[all]"

# 특정 프로바이더만 설치
pip install "pyhub-llm[openai]"      # OpenAI만
pip install "pyhub-llm[anthropic]"   # Anthropic만
pip install "pyhub-llm[google]"      # Google만
pip install "pyhub-llm[ollama]"      # Ollama만

# MCP 지원 포함
pip install "pyhub-llm[all,mcp]"
```

## 기본 사용법

### 환경변수 설정

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export UPSTAGE_API_KEY="..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

### OpenAI

```python
from pyhub.llm import LLM, OpenAILLM

# 팩토리 패턴 사용 (권장)
llm = LLM.create("gpt-4o-mini")
reply = llm.ask("안녕하세요!")
print(reply.text)

# 직접 생성
llm = OpenAILLM(model="gpt-4o-mini", temperature=0.7)
reply = llm.ask("파이썬의 장점을 3가지 알려주세요")
print(reply.text)
```

### Anthropic

```python
from pyhub.llm import LLM, AnthropicLLM

# Claude 사용
llm = LLM.create("claude-3-haiku-20240307")
reply = llm.ask("양자 컴퓨터를 쉽게 설명해주세요")
print(reply.text)

# 시스템 프롬프트 설정
llm = AnthropicLLM(
    model="claude-3-5-sonnet-20241022",
    system_prompt="당신은 친절한 교육 전문가입니다."
)
```

### Google

```python
from pyhub.llm import LLM, GoogleLLM

# Gemini 사용
llm = LLM.create("gemini-1.5-flash")
reply = llm.ask("오늘 날씨가 좋네요. 산책하기 좋은 장소를 추천해주세요.")
print(reply.text)

# 긴 컨텍스트 처리
llm = GoogleLLM(model="gemini-1.5-pro", max_tokens=8192)
```

### Ollama (로컬)

```python
from pyhub.llm import LLM, OllamaLLM

# Ollama는 API 키 불필요
llm = LLM.create("mistral")
reply = llm.ask("로컬에서 실행되는 LLM의 장점은?")
print(reply.text)

# 커스텀 서버 주소
llm = OllamaLLM(
    model="llama2",
    base_url="http://192.168.1.100:11434"
)
```

### Upstage

```python
from pyhub.llm import UpstageLLM

llm = UpstageLLM(model="solar-mini")
reply = llm.ask("한국어 자연어 처리의 어려운 점은?")
print(reply.text)
```

## 스트리밍

실시간으로 응답을 받아 처리합니다.

```python
from pyhub.llm import LLM

llm = LLM.create("gpt-4o-mini")

# 기본 스트리밍
for chunk in llm.ask("긴 이야기를 들려주세요", stream=True):
    print(chunk.text, end="", flush=True)
print()

# 스트리밍 중 처리
def process_stream(llm, prompt):
    full_text = ""
    for i, chunk in enumerate(llm.ask(prompt, stream=True)):
        full_text += chunk.text
        # 특정 단어가 나오면 중단
        if "종료" in chunk.text:
            break
        # 진행 상황 표시
        if i % 10 == 0:
            print(".", end="", flush=True)
    return full_text
```

## 구조화된 출력

### Pydantic 스키마 사용

```python
from pydantic import BaseModel, Field
from typing import List
from pyhub.llm import LLM

class BookInfo(BaseModel):
    title: str = Field(description="책 제목")
    author: str = Field(description="저자")
    year: int = Field(description="출판 연도")
    genres: List[str] = Field(description="장르 목록")
    summary: str = Field(description="간단한 줄거리")

llm = LLM.create("gpt-4o-mini")
reply = llm.ask(
    "해리포터와 마법사의 돌에 대해 알려주세요",
    schema=BookInfo
)

book: BookInfo = reply.structured_data
print(f"제목: {book.title}")
print(f"저자: {book.author}")
print(f"장르: {', '.join(book.genres)}")
```

### 복잡한 구조

```python
class Company(BaseModel):
    name: str
    founded: int
    headquarters: str

class ProductAnalysis(BaseModel):
    product_name: str
    manufacturer: Company
    pros: List[str]
    cons: List[str]
    rating: float = Field(ge=0, le=5)
    recommendation: bool

llm = LLM.create("gpt-4o-mini")
reply = llm.ask(
    "iPhone 15 Pro에 대한 분석을 해주세요",
    schema=ProductAnalysis
)

analysis: ProductAnalysis = reply.structured_data
print(f"제조사: {analysis.manufacturer.name}")
print(f"평점: {analysis.rating}/5.0")
```

### 다국어 응답

```python
class Translation(BaseModel):
    korean: str
    english: str
    japanese: str
    chinese: str

llm = LLM.create("gpt-4o-mini", system_prompt="다국어 번역 전문가")
reply = llm.ask("'인공지능'을 4개 언어로 번역해주세요", schema=Translation)

trans: Translation = reply.structured_data
print(f"한국어: {trans.korean}")
print(f"영어: {trans.english}")
print(f"일본어: {trans.japanese}")
print(f"중국어: {trans.chinese}")
```

## 분류 및 선택

### 감정 분석

```python
from pyhub.llm import LLM

llm = LLM.create("gpt-4o-mini")

# 단일 선택
emotions = ["기쁨", "슬픔", "분노", "공포", "놀람", "혐오"]
reply = llm.ask("오늘 승진했어요! 축하 파티도 했답니다.", choices=emotions)
print(f"감정: {reply.choice}")  # "기쁨"
print(f"인덱스: {reply.choice_index}")  # 0

# 여러 문장 일괄 처리
texts = [
    "프로젝트가 실패했습니다.",
    "복권에 당첨됐어요!",
    "또 야근이네요..."
]

for text in texts:
    reply = llm.ask(text, choices=emotions)
    print(f"{text} → {reply.choice}")
```

### 의도 분류

```python
class IntentClassifier:
    def __init__(self):
        self.llm = LLM.create("gpt-4o-mini")
        self.intents = [
            "질문",
            "요청",
            "불만",
            "칭찬",
            "정보제공",
            "기타"
        ]
    
    def classify(self, text: str) -> str:
        reply = self.llm.ask(text, choices=self.intents)
        return reply.choice

classifier = IntentClassifier()
print(classifier.classify("이 제품 환불 가능한가요?"))  # "질문"
print(classifier.classify("정말 최고의 서비스입니다!"))  # "칭찬"
```

### 다중 라벨 분류

```python
from pydantic import BaseModel
from typing import List

class TopicLabels(BaseModel):
    topics: List[str] = Field(description="해당하는 모든 주제")

llm = LLM.create("gpt-4o-mini", system_prompt="텍스트의 주제를 분류하는 전문가")

available_topics = ["정치", "경제", "사회", "문화", "스포츠", "IT", "과학", "건강"]

prompt = f"""
다음 텍스트에 해당하는 모든 주제를 선택하세요.
가능한 주제: {', '.join(available_topics)}

텍스트: 'AI 기술이 의료 분야에 혁명을 일으키고 있습니다. 특히 암 진단의 정확도가 크게 향상되었습니다.'
"""

reply = llm.ask(prompt, schema=TopicLabels)
print(f"분류된 주제: {', '.join(reply.structured_data.topics)}")  # "IT, 과학, 건강"
```

## 비동기 처리

### 기본 비동기 사용

```python
import asyncio
from pyhub.llm import LLM

async def main():
    llm = LLM.create("gpt-4o-mini")
    
    # 비동기 요청
    reply = await llm.ask_async("비동기 프로그래밍의 장점은?")
    print(reply.text)
    
    # 비동기 스트리밍
    async for chunk in llm.ask_async("긴 설명을 해주세요", stream=True):
        print(chunk.text, end="", flush=True)

# 실행
asyncio.run(main())
```

### 동시 요청 처리

```python
async def process_multiple_queries():
    llm = LLM.create("gpt-4o-mini")
    
    queries = [
        "Python의 장점은?",
        "JavaScript의 장점은?",
        "Rust의 장점은?"
    ]
    
    # 모든 요청을 동시에 처리
    tasks = [llm.ask_async(q) for q in queries]
    replies = await asyncio.gather(*tasks)
    
    for query, reply in zip(queries, replies):
        print(f"\nQ: {query}")
        print(f"A: {reply.text[:100]}...")

asyncio.run(process_multiple_queries())
```

### MCP와 함께 비동기 사용

```python
from pyhub.llm import LLM

async def main():
    # 간편한 문자열 설정로 MCP 서버와 함께 LLM 생성
    llm = await LLM.create_async(
        "gpt-4o-mini",
        mcp_servers="python calculator.py"  # 문자열로 간편 설정
    )
    
    # 또는 더 상세한 설정
    # from pyhub.llm.mcp import McpConfig
    # llm = await LLM.create_async(
    #     "gpt-4o-mini",
    #     mcp_servers=McpConfig(
    #         cmd="calculator-server",
    #         name="my-calc"
    #     )
    # )
    
    # MCP 도구 사용
    reply = await llm.ask_async("25 곱하기 17은?")
    print(reply.text)
    
    # 정리
    await llm.close_mcp()

asyncio.run(main())
```

## 캐싱

### 인메모리 캐싱

```python
from pyhub.llm import LLM
from pyhub.llm.cache import InMemoryCache

# 캐시 설정
cache = InMemoryCache(ttl=3600)  # 1시간 TTL
llm = LLM.create("gpt-4o-mini", cache=cache)

# 첫 번째 요청 (API 호출)
reply1 = llm.ask("파이썬의 역사를 간단히 설명해주세요")
print("첫 번째 요청 완료")

# 두 번째 요청 (캐시에서 반환)
reply2 = llm.ask("파이썬의 역사를 간단히 설명해주세요")
print("캐시된 응답:", reply1.text == reply2.text)  # True
```

### 파일 기반 캐싱

```python
from pyhub.llm.cache import FileCache
from pathlib import Path

# 파일 캐시 설정
cache_dir = Path("./llm_cache")
cache = FileCache(cache_dir=cache_dir, ttl=86400)  # 24시간 TTL

llm = LLM.create("gpt-4o-mini", cache=cache)

# 캐시 통계
print(f"캐시 히트율: {cache.hit_rate:.2%}")
print(f"캐시 크기: {cache.size_bytes / 1024 / 1024:.2f} MB")

# 캐시 정리
cache.clear_expired()  # 만료된 항목 삭제
# cache.clear()  # 전체 캐시 삭제
```

### 조건부 캐싱

```python
class SmartCache:
    def __init__(self, llm):
        self.llm = llm
        self.cache = InMemoryCache(ttl=3600)
        self.llm_with_cache = LLM.create(llm.model, cache=self.cache)
    
    def ask(self, prompt: str, use_cache: bool = True):
        """캐시 사용 여부를 동적으로 결정"""
        if use_cache and not self._is_dynamic_content(prompt):
            return self.llm_with_cache.ask(prompt)
        else:
            return self.llm.ask(prompt)
    
    def _is_dynamic_content(self, prompt: str) -> bool:
        """동적 컨텐츠 여부 판단"""
        dynamic_keywords = ["현재", "오늘", "지금", "실시간", "최신"]
        return any(keyword in prompt for keyword in dynamic_keywords)

# 사용 예
smart_llm = SmartCache(LLM.create("gpt-4o-mini"))
reply1 = smart_llm.ask("파이썬이란?")  # 캐시됨
reply2 = smart_llm.ask("현재 시각은?")  # 캐시 안됨
```

## 대화 관리

### 대화 히스토리 유지

```python
from pyhub.llm import LLM
from pyhub.llm.types import Message

class ChatBot:
    def __init__(self, model="gpt-4o-mini"):
        self.llm = LLM.create(model)
        self.history = []
    
    def chat(self, user_input: str) -> str:
        # 사용자 메시지 추가
        self.history.append(Message(role="user", content=user_input))
        
        # LLM에게 전체 히스토리 전달
        reply = self.llm.messages(self.history)
        
        # 응답을 히스토리에 추가
        self.history.append(Message(role="assistant", content=reply.text))
        
        return reply.text
    
    def reset(self):
        """대화 초기화"""
        self.history = []
    
    def save_history(self, filename: str):
        """대화 내역 저장"""
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([msg.dict() for msg in self.history], f, ensure_ascii=False, indent=2)

# 사용 예
bot = ChatBot()
print(bot.chat("안녕하세요! 저는 프로그래밍을 배우고 싶어요."))
print(bot.chat("파이썬부터 시작하면 좋을까요?"))
print(bot.chat("그럼 첫 번째로 뭘 배워야 할까요?"))
```

### 컨텍스트 윈도우 관리

```python
class ContextManagedChat:
    def __init__(self, model="gpt-4o-mini", max_messages=10):
        self.llm = LLM.create(model)
        self.history = []
        self.max_messages = max_messages
    
    def chat(self, user_input: str) -> str:
        # 컨텍스트 윈도우 크기 제한
        if len(self.history) >= self.max_messages * 2:
            # 시스템 메시지는 유지하고 오래된 대화 제거
            self.history = self.history[-self.max_messages * 2:]
        
        self.history.append(Message(role="user", content=user_input))
        reply = self.llm.messages(self.history)
        self.history.append(Message(role="assistant", content=reply.text))
        
        return reply.text
```

### 페르소나 기반 대화

```python
class PersonaChat:
    def __init__(self, persona: str, model="gpt-4o-mini"):
        self.llm = LLM.create(
            model,
            system_prompt=persona
        )
        self.history = []
    
    def chat(self, message: str) -> str:
        reply = self.llm.ask(message)
        return reply.text

# 다양한 페르소나
teacher = PersonaChat("당신은 친절하고 인내심 있는 프로그래밍 교사입니다.")
chef = PersonaChat("당신은 미슐랭 3스타 셰프입니다. 요리에 대한 열정이 넘칩니다.")
doctor = PersonaChat("당신은 의학 전문가입니다. 정확하고 신중하게 답변합니다.")

print(teacher.chat("재귀함수가 뭔가요?"))
print(chef.chat("파스타 면을 삶는 최적의 시간은?"))
print(doctor.chat("두통이 자주 있어요"))  # 주의: 실제 의료 조언이 아님
```

## 파일 처리

### 이미지 분석

```python
from pyhub.llm import LLM

llm = LLM.create("gpt-4o-mini")  # 비전 지원 모델

# 단일 이미지 분석
reply = llm.ask(
    "이 이미지에 무엇이 보이나요?",
    files=["photo.jpg"]
)
print(reply.text)

# 여러 이미지 비교
reply = llm.ask(
    "이 두 이미지의 차이점을 설명해주세요",
    files=["before.png", "after.png"]
)
print(reply.text)

# 이미지와 함께 구조화된 출력
from pydantic import BaseModel
from typing import List

class ImageAnalysis(BaseModel):
    objects: List[str] = Field(description="이미지에서 발견된 객체들")
    scene: str = Field(description="전체적인 장면 설명")
    colors: List[str] = Field(description="주요 색상들")
    mood: str = Field(description="이미지의 분위기")

reply = llm.ask(
    "이 이미지를 분석해주세요",
    files=["landscape.jpg"],
    schema=ImageAnalysis
)
analysis = reply.structured_data
print(f"발견된 객체: {', '.join(analysis.objects)}")
print(f"분위기: {analysis.mood}")
```

### PDF 문서 처리

```python
# PDF는 자동으로 이미지로 변환됨
llm = LLM.create("gpt-4o-mini")

# PDF 요약
reply = llm.ask(
    "이 PDF 문서의 주요 내용을 요약해주세요",
    files=["report.pdf"]
)
print(reply.text)

# 여러 페이지 PDF 처리
class PDFSummary(BaseModel):
    title: str
    main_topics: List[str]
    key_findings: List[str]
    recommendations: List[str]

reply = llm.ask(
    "이 연구 보고서를 분석해주세요",
    files=["research_paper.pdf"],
    schema=PDFSummary
)
```

### 이미지 생성 프롬프트

```python
class ImagePrompt(BaseModel):
    style: str = Field(description="그림 스타일")
    subject: str = Field(description="주요 대상")
    background: str = Field(description="배경 설명")
    mood: str = Field(description="분위기")
    details: List[str] = Field(description="추가 세부사항")

llm = LLM.create("gpt-4o-mini")

# 이미지를 보고 유사한 이미지 생성을 위한 프롬프트 생성
reply = llm.ask(
    "이 이미지와 유사한 이미지를 생성하기 위한 프롬프트를 만들어주세요",
    files=["reference_image.jpg"],
    schema=ImagePrompt
)

prompt_data = reply.structured_data
print(f"스타일: {prompt_data.style}")
print(f"프롬프트: A {prompt_data.style} image of {prompt_data.subject} with {prompt_data.background}")
```

## 임베딩

### 텍스트 임베딩 생성

```python
from pyhub.llm import LLM
import numpy as np

# 임베딩 모델 사용
llm = LLM.create("text-embedding-3-small")

# 단일 텍스트 임베딩
text = "인공지능은 인간의 지능을 모방한 기술입니다."
embedding = llm.embed(text)
print(f"임베딩 차원: {len(embedding.embeddings[0])}")  # 1536

# 여러 텍스트 임베딩
texts = [
    "파이썬은 프로그래밍 언어입니다.",
    "Python is a programming language.",
    "자바스크립트는 웹 개발에 사용됩니다."
]
embeddings = llm.embed(texts)
print(f"생성된 임베딩 수: {len(embeddings.embeddings)}")
```

### 유사도 계산

```python
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1: str, text2: str) -> float:
    """두 텍스트의 유사도 계산"""
    llm = LLM.create("text-embedding-3-small")
    
    embeddings = llm.embed([text1, text2])
    vec1 = np.array(embeddings.embeddings[0]).reshape(1, -1)
    vec2 = np.array(embeddings.embeddings[1]).reshape(1, -1)
    
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return similarity

# 사용 예
pairs = [
    ("고양이는 귀여운 동물입니다.", "강아지는 충실한 반려동물입니다."),
    ("파이썬으로 웹 개발하기", "Python web development"),
    ("오늘 날씨가 좋네요", "내일 비가 온대요")
]

for text1, text2 in pairs:
    sim = calculate_similarity(text1, text2)
    print(f"유사도: {sim:.3f} - '{text1}' vs '{text2}'")
```

### 문서 검색 시스템

```python
class DocumentSearch:
    def __init__(self, model="text-embedding-3-small"):
        self.llm = LLM.create(model)
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, docs: List[str]):
        """문서 추가 및 임베딩 생성"""
        self.documents.extend(docs)
        new_embeddings = self.llm.embed(docs).embeddings
        self.embeddings.extend(new_embeddings)
    
    def search(self, query: str, top_k: int = 3) -> List[tuple]:
        """쿼리와 가장 유사한 문서 검색"""
        query_embedding = self.llm.embed(query).embeddings[0]
        
        # 모든 문서와의 유사도 계산
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(
                [query_embedding], 
                [doc_embedding]
            )[0][0]
            similarities.append((sim, i, self.documents[i]))
        
        # 상위 k개 반환
        similarities.sort(reverse=True)
        return [(score, doc) for score, idx, doc in similarities[:top_k]]

# 사용 예
search = DocumentSearch()
search.add_documents([
    "파이썬은 배우기 쉬운 프로그래밍 언어입니다.",
    "자바스크립트는 웹 브라우저에서 실행됩니다.",
    "머신러닝은 데이터를 학습하는 기술입니다.",
    "인공지능은 인간의 지능을 모방합니다.",
    "데이터베이스는 데이터를 저장하고 관리합니다."
])

results = search.search("AI와 기계학습", top_k=2)
for score, doc in results:
    print(f"유사도 {score:.3f}: {doc}")
```

## 템플릿 활용

### Jinja2 템플릿

```python
from pyhub.llm import LLM
from pyhub.llm.templates import PromptTemplate

# 템플릿 정의
template = PromptTemplate("""
당신은 {{ role }}입니다.

사용자의 요청: {{ request }}

다음 조건을 고려하여 답변해주세요:
{% for condition in conditions %}
- {{ condition }}
{% endfor %}
""")

llm = LLM.create("gpt-4o-mini")

# 템플릿 사용
prompt = template.render(
    role="전문 요리사",
    request="파스타 만드는 법을 알려주세요",
    conditions=[
        "초보자도 쉽게 따라할 수 있도록",
        "재료는 마트에서 쉽게 구할 수 있는 것으로",
        "30분 이내에 완성 가능한 레시피"
    ]
)

reply = llm.ask(prompt)
print(reply.text)
```

### Few-shot 템플릿

```python
few_shot_template = PromptTemplate("""
다음 예시를 참고하여 작업을 수행하세요.

{% for example in examples %}
입력: {{ example.input }}
출력: {{ example.output }}

{% endfor %}
입력: {{ input }}
출력:""")

# 번역 예시
examples = [
    {"input": "Hello", "output": "안녕하세요"},
    {"input": "Thank you", "output": "감사합니다"},
    {"input": "Good morning", "output": "좋은 아침입니다"}
]

prompt = few_shot_template.render(
    examples=examples,
    input="How are you?"
)

reply = llm.ask(prompt)
print(reply.text)  # "어떻게 지내세요?" 또는 유사한 번역
```

### 동적 템플릿

```python
class DynamicPromptBuilder:
    def __init__(self):
        self.templates = {
            "technical": PromptTemplate("기술적 관점에서 {{ topic }}에 대해 설명하세요."),
            "simple": PromptTemplate("5살 아이도 이해할 수 있게 {{ topic }}을 설명하세요."),
            "business": PromptTemplate("비즈니스 관점에서 {{ topic }}의 가치를 설명하세요.")
        }
    
    def build(self, style: str, topic: str) -> str:
        template = self.templates.get(style, self.templates["simple"])
        return template.render(topic=topic)

builder = DynamicPromptBuilder()
llm = LLM.create("gpt-4o-mini")

# 같은 주제를 다른 스타일로
topic = "블록체인 기술"
for style in ["technical", "simple", "business"]:
    prompt = builder.build(style, topic)
    reply = llm.ask(prompt)
    print(f"\n[{style.upper()}]\n{reply.text[:200]}...")
```

## History Backup

대화 히스토리를 외부 저장소에 백업하고 복원하는 기능입니다. 메모리 기반 히스토리와 별도로 영구 저장소에 대화 내역을 보관할 수 있습니다.

### 기본 사용법 (InMemoryHistoryBackup)

```python
from pyhub.llm import LLM
from pyhub.llm.history import InMemoryHistoryBackup

# 메모리 기반 백업 (테스트용)
backup = InMemoryHistoryBackup(
    user_id="user123",
    session_id="session456"
)

# 백업이 활성화된 LLM 생성
llm = LLM.create("gpt-4o-mini", history_backup=backup)

# 대화 진행 (자동으로 백업됨)
llm.ask("Python의 장점은 무엇인가요?")
llm.ask("더 자세히 설명해주세요")

# 백업된 히스토리 확인
messages = backup.load_history()
for msg in messages:
    print(f"{msg.role}: {msg.content[:50]}...")

# 사용량 통계
usage = backup.get_usage_summary()
print(f"총 입력 토큰: {usage.input}")
print(f"총 출력 토큰: {usage.output}")
```

### SQLAlchemy 백업 (영구 저장)

```python
from pyhub.llm import LLM
from pyhub.llm.history import SQLAlchemyHistoryBackup
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 데이터베이스 설정
engine = create_engine("sqlite:///chat_history.db")
Session = sessionmaker(bind=engine)
session = Session()

# SQLAlchemy 백업 생성
backup = SQLAlchemyHistoryBackup(
    session=session,
    user_id="user123",
    session_id="session456"
)

# 테이블 자동 생성
from pyhub.llm.history.sqlalchemy_backup import Base
Base.metadata.create_all(engine)

# 백업이 활성화된 LLM 생성
llm = LLM.create("gpt-4o-mini", history_backup=backup)

# 대화 진행
llm.ask("데이터베이스 설계 원칙을 설명해주세요")
llm.ask("정규화에 대해 더 자세히 알려주세요")

# 세션 커밋 (영구 저장)
session.commit()
```

### 이전 대화 복원

```python
# 새로운 세션에서 이전 대화 불러오기
new_session = Session()
backup = SQLAlchemyHistoryBackup(
    session=new_session,
    user_id="user123",
    session_id="session456"
)

# 이전 대화가 자동으로 복원된 LLM
llm = LLM.create("gpt-4o-mini", history_backup=backup)

# 이전 대화 컨텍스트를 유지한 채 계속 대화
llm.ask("앞서 설명한 정규화의 단점은 무엇인가요?")
```

### 여러 세션 관리

```python
# 사용자별 여러 세션 관리
user_id = "user123"

# 세션 1: 프로그래밍 질문
session1_backup = SQLAlchemyHistoryBackup(
    session=session,
    user_id=user_id,
    session_id="programming_session"
)
llm1 = LLM.create("gpt-4o-mini", history_backup=session1_backup)
llm1.ask("Python과 JavaScript의 차이점은?")

# 세션 2: 수학 질문
session2_backup = SQLAlchemyHistoryBackup(
    session=session,
    user_id=user_id,
    session_id="math_session"
)
llm2 = LLM.create("gpt-4o-mini", history_backup=session2_backup)
llm2.ask("미적분학의 기본 정리를 설명해주세요")

# 각 세션은 독립적으로 관리됨
```

### Tool 사용 내역 자동 저장

```python
# 도구 호출 내역도 자동으로 백업됨
def get_weather(city: str) -> str:
    """도시의 날씨 정보를 가져옵니다."""
    return f"{city}의 날씨는 맑음, 25°C입니다."

def get_time(timezone: str = "UTC") -> str:
    """현재 시간을 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

llm = LLM.create("gpt-4o-mini", history_backup=backup)
reply = llm.ask(
    "서울의 날씨와 현재 시간을 알려주세요",
    tools=[get_weather, get_time]
)

# 백업된 메시지 확인
messages = backup.load_history()
assistant_msg = messages[-1]  # 마지막 어시스턴트 메시지

# tool_interactions 필드에 도구 사용 내역이 저장됨
if assistant_msg.tool_interactions:
    for interaction in assistant_msg.tool_interactions:
        print(f"도구: {interaction['tool']}")
        print(f"인자: {interaction['arguments']}")
        print(f"결과: {interaction.get('result', 'N/A')}")
```

### 사용자 정의 백업 구현

```python
from abc import ABC, abstractmethod
from pyhub.llm.history import HistoryBackup
from pyhub.llm.types import Message, Usage

class MongoDBHistoryBackup(HistoryBackup):
    """MongoDB를 사용한 히스토리 백업 예제"""
    
    def __init__(self, collection, user_id: str, session_id: str):
        self.collection = collection
        self.user_id = user_id
        self.session_id = session_id
    
    def save_exchange(
        self,
        user_msg: Message,
        assistant_msg: Message,
        usage: Optional[Usage] = None,
        model: Optional[str] = None
    ) -> None:
        """대화 교환을 MongoDB에 저장"""
        doc = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": datetime.utcnow(),
            "user_message": {
                "content": user_msg.content,
                "files": user_msg.files
            },
            "assistant_message": {
                "content": assistant_msg.content,
                "tool_interactions": assistant_msg.tool_interactions
            },
            "usage": {
                "input": usage.input if usage else 0,
                "output": usage.output if usage else 0
            },
            "model": model
        }
        self.collection.insert_one(doc)
    
    def load_history(self, limit: Optional[int] = None) -> list[Message]:
        """MongoDB에서 히스토리 로드"""
        query = {
            "user_id": self.user_id,
            "session_id": self.session_id
        }
        
        cursor = self.collection.find(query).sort("timestamp", 1)
        if limit:
            cursor = cursor.limit(limit // 2)  # 각 교환은 2개 메시지
        
        messages = []
        for doc in cursor:
            # 사용자 메시지
            messages.append(Message(
                role="user",
                content=doc["user_message"]["content"],
                files=doc["user_message"].get("files")
            ))
            
            # 어시스턴트 메시지
            messages.append(Message(
                role="assistant",
                content=doc["assistant_message"]["content"],
                tool_interactions=doc["assistant_message"].get("tool_interactions")
            ))
        
        return messages
    
    def get_usage_summary(self) -> Usage:
        """총 사용량 계산"""
        pipeline = [
            {"$match": {"user_id": self.user_id, "session_id": self.session_id}},
            {"$group": {
                "_id": None,
                "total_input": {"$sum": "$usage.input"},
                "total_output": {"$sum": "$usage.output"}
            }}
        ]
        
        result = list(self.collection.aggregate(pipeline))
        if result:
            return Usage(
                input=result[0]["total_input"],
                output=result[0]["total_output"]
            )
        return Usage(input=0, output=0)
    
    def clear(self) -> int:
        """히스토리 삭제"""
        result = self.collection.delete_many({
            "user_id": self.user_id,
            "session_id": self.session_id
        })
        return result.deleted_count * 2  # 각 문서는 2개 메시지
```

### 백업 실패 처리

```python
# 백업 실패 시에도 LLM은 정상 동작
import logging

class UnreliableBackup(HistoryBackup):
    """간헐적으로 실패하는 백업 (예제)"""
    
    def save_exchange(self, user_msg, assistant_msg, usage=None, model=None):
        import random
        if random.random() < 0.3:  # 30% 확률로 실패
            raise Exception("Backup service temporarily unavailable")
        # 실제 저장 로직...

# 백업 실패는 자동으로 처리됨
backup = UnreliableBackup()
llm = LLM.create("gpt-4o-mini", history_backup=backup)

# 백업이 실패해도 대화는 계속됨
reply = llm.ask("백업이 실패해도 괜찮나요?")
# 경고 로그만 출력되고 정상 동작
```

### 주요 메서드 설명

- `save_exchange()`: 사용자 메시지와 어시스턴트 응답을 한 쌍으로 저장
- `load_history(limit)`: 저장된 히스토리를 Message 리스트로 반환
- `get_usage_summary()`: 총 토큰 사용량 통계 반환
- `clear()`: 해당 세션의 모든 히스토리 삭제

> 💡 **팁**: 
> - 백업은 메모리 히스토리와 별개로 동작하며, 주로 영구 저장 용도로 사용됩니다
> - Tool 사용 내역은 `tool_interactions` 필드에 자동으로 저장됩니다
> - 백업 실패 시에도 LLM은 정상적으로 동작하며, 경고 로그만 출력됩니다

## MCP 통합

### 서버 이름 자동 감지

MCP 서버는 초기화 시 자체 정보(이름, 버전)를 제공합니다. pyhub-llm은 이를 활용하여 서버 이름을 자동으로 감지합니다:

```python
# name 없이 설정 - 서버가 "calculator-server"로 자동 제공
config = McpConfig(
    cmd="pyhub-llm mcp-server run calculator"
)

# 사용자가 원하면 name 오버라이드 가능
config = McpConfig(
    name="my_calc",  # 서버 이름을 "my_calc"로 변경
    cmd="pyhub-llm mcp-server run calculator"
)
```

**서버 이름 우선순위:**
1. 사용자가 지정한 `name` (최우선)
2. 서버가 제공하는 이름 (자동 감지)
3. 자동 생성된 이름 (transport_uuid 형태)

**중복 처리:**
- 동일한 이름의 서버가 여러 개인 경우 자동으로 suffix 추가 (`calculator-server_1`, `calculator-server_2`)
- 중복 시 경고 로그 출력

### 통합 설정 및 자동 감지

pyhub-llm 0.7.0부터는 모든 MCP transport 타입을 단일 `McpConfig` 클래스로 통합하고, transport 타입을 자동으로 감지합니다:

```python
from pyhub.llm.mcp import McpConfig, create_mcp_config

# 1. 기본 설정 - transport 자동 감지
stdio_config = McpConfig(
    cmd="python calculator.py"  # stdio transport로 자동 감지
)

http_config = McpConfig(
    url="http://localhost:8080/mcp"  # streamable_http transport로 자동 감지
)

ws_config = McpConfig(
    url="ws://localhost:8080/ws"  # websocket transport로 자동 감지
)

sse_config = McpConfig(
    url="http://localhost:8080/sse"  # sse transport로 자동 감지
)

# 2. 문자열로 간편 설정 - 팩토리 함수 사용
config1 = create_mcp_config("python server.py")  # stdio
config2 = create_mcp_config("http://localhost:8080")  # http
config3 = create_mcp_config("ws://localhost:8080")  # websocket

# 3. 딕셔너리로 설정
config4 = create_mcp_config({
    "cmd": "python server.py",
    "name": "my-server",
    "timeout": 60
})
```

**Transport 자동 감지 규칙:**
- `cmd` 또는 `command` 필드 → `stdio` transport
- `http://` 또는 `https://` URL → `streamable_http` transport
- `ws://` 또는 `wss://` URL → `websocket` transport
- URL에 `/sse` 포함 또는 `text/event-stream` 헤더 → `sse` transport

### 기본 MCP 사용

```python
from pyhub.llm import LLM
from pyhub.llm.mcp import McpConfig

# MCP 서버 설정
mcp_config = McpConfig(
    cmd="calculator-server"  # MCP 서버 실행 명령 (name은 서버가 자동 제공)
)

# LLM과 MCP 통합
llm = LLM.create("gpt-4o-mini", mcp_servers=mcp_config)

# 자동으로 MCP 초기화
await llm.initialize_mcp()

# MCP 도구가 자동으로 사용됨
reply = await llm.ask_async("1234 곱하기 5678은?")
print(reply.text)  # MCP 계산기를 사용한 정확한 답변

# 정리
await llm.close_mcp()
```

### MCP 연결 정책

```python
from pyhub.llm.mcp import MCPConnectionPolicy, MCPConnectionError

# OPTIONAL (기본값) - MCP 실패해도 계속
llm1 = await LLM.create_async(
    "gpt-4o-mini",
    mcp_servers=mcp_config
)

# REQUIRED - MCP 필수, 실패 시 예외
try:
    llm2 = await LLM.create_async(
        "gpt-4o-mini",
        mcp_servers=mcp_config,
        mcp_policy=MCPConnectionPolicy.REQUIRED
    )
except MCPConnectionError as e:
    print(f"MCP 연결 실패: {e}")
    print(f"실패한 서버: {e.failed_servers}")

# WARN - 실패 시 경고만
llm3 = await LLM.create_async(
    "gpt-4o-mini",
    mcp_servers=mcp_config,
    mcp_policy=MCPConnectionPolicy.WARN
)
```

### 여러 MCP 서버 통합

```python
from pyhub.llm.mcp import McpConfig, create_mcp_config

# 방법 1: 문자열 리스트로 간편 설정
servers = [
    "python calculator.py",  # stdio transport 자동 감지
    "http://localhost:8080/mcp"  # http transport 자동 감지
]

# 방법 2: 딕셔너리 리스트로 상세 설정
servers = [
    {"cmd": "calculator-server", "name": "calc"},
    {"url": "http://localhost:8080/mcp", "name": "web-api"}
]

# 방법 3: McpConfig 객체로 상세 설정
servers = [
    McpConfig(
        cmd="calculator-server",
        timeout=60
    ),
    McpConfig(
        url="http://localhost:8080/mcp",
        headers={"Authorization": "Bearer token"}
    )
]

# 여러 서버와 함께 LLM 생성
llm = await LLM.create_async("gpt-4o-mini", mcp_servers=servers)

# 모든 도구가 통합되어 사용 가능
reply = await llm.ask_async(
    "서울의 현재 온도를 섭씨에서 화씨로 변환해주세요"
)
print(reply.text)  # 날씨 API + 계산기 도구 모두 사용
```

### MCP 도구 직접 제어

```python
# MCP 도구 목록 확인
if llm._mcp_tools:
    print("사용 가능한 MCP 도구:")
    for tool in llm._mcp_tools:
        print(f"- {tool.name}: {tool.description}")

# 특정 도구만 사용하도록 필터링
filtered_config = McpConfig(
    cmd="calculator-server",
    filter_tools=["add", "multiply"]  # 덧셈과 곱셈만 사용
)
```

## 웹 프레임워크 통합

### FastAPI 통합

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pyhub.llm import LLM
import asyncio

app = FastAPI()

# 전역 LLM 인스턴스
llm = LLM.create("gpt-4o-mini")

class ChatRequest(BaseModel):
    message: str
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    tokens_used: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """일반 채팅 엔드포인트"""
    try:
        reply = await llm.ask_async(request.message)
        return ChatResponse(
            response=reply.text,
            tokens_used=reply.usage.total if reply.usage else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """스트리밍 채팅 엔드포인트"""
    async def generate():
        async for chunk in llm.ask_async(request.message, stream=True):
            yield f"data: {chunk.text}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# 구조화된 출력 엔드포인트
class AnalysisRequest(BaseModel):
    text: str

class SentimentAnalysis(BaseModel):
    sentiment: str
    confidence: float
    keywords: List[str]

@app.post("/analyze", response_model=SentimentAnalysis)
async def analyze_sentiment(request: AnalysisRequest):
    """감정 분석 엔드포인트"""
    reply = await llm.ask_async(
        f"다음 텍스트의 감정을 분석하세요: {request.text}",
        schema=SentimentAnalysis
    )
    return reply.structured_data

# 이미지 처리 엔드포인트
from fastapi import UploadFile, File

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = "이 이미지를 설명해주세요"
):
    """이미지 분석 엔드포인트"""
    contents = await file.read()
    
    # 임시 파일로 저장
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        reply = await llm.ask_async(question, files=[tmp_path])
        return {"description": reply.text}
    finally:
        import os
        os.unlink(tmp_path)

# 백그라운드 작업
from fastapi import BackgroundTasks

async def process_long_task(task_id: str, prompt: str):
    """긴 작업을 백그라운드에서 처리"""
    reply = await llm.ask_async(prompt)
    # 결과를 DB나 캐시에 저장
    # save_result(task_id, reply.text)

@app.post("/long-task")
async def create_long_task(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """백그라운드 작업 생성"""
    task_id = str(uuid.uuid4())
    background_tasks.add_task(
        process_long_task,
        task_id,
        request.message
    )
    return {"task_id": task_id, "status": "processing"}
```

### Django 통합

```python
# views.py
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.core.cache import cache
from pyhub.llm import LLM
import json

# 전역 LLM 인스턴스 (settings에서 관리 권장)
llm = LLM.create("gpt-4o-mini")

@method_decorator(csrf_exempt, name='dispatch')
class ChatView(View):
    """채팅 API 뷰"""
    
    def post(self, request):
        data = json.loads(request.body)
        message = data.get('message', '')
        
        # 캐시 확인
        cache_key = f"chat:{hash(message)}"
        cached_response = cache.get(cache_key)
        if cached_response:
            return JsonResponse({'response': cached_response, 'cached': True})
        
        # LLM 호출
        reply = llm.ask(message)
        
        # 캐시 저장 (1시간)
        cache.set(cache_key, reply.text, 3600)
        
        return JsonResponse({
            'response': reply.text,
            'cached': False,
            'tokens': reply.usage.total if reply.usage else 0
        })

def chat_stream_view(request):
    """스트리밍 채팅 뷰"""
    message = request.GET.get('message', '')
    
    def generate():
        for chunk in llm.ask(message, stream=True):
            yield f"data: {json.dumps({'text': chunk.text})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingHttpResponse(
        generate(),
        content_type='text/event-stream'
    )

# models.py
from django.db import models
from django.contrib.auth.models import User

class ChatHistory(models.Model):
    """채팅 히스토리 모델"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    tokens_used = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-created_at']

# 채팅 히스토리와 함께 사용
class ChatWithHistoryView(View):
    def post(self, request):
        user = request.user
        data = json.loads(request.body)
        message = data.get('message', '')
        
        # 이전 대화 내역 가져오기
        history = ChatHistory.objects.filter(user=user).order_by('-created_at')[:5]
        
        # 컨텍스트 구성
        messages = []
        for h in reversed(history):
            messages.append(Message(role="user", content=h.message))
            messages.append(Message(role="assistant", content=h.response))
        messages.append(Message(role="user", content=message))
        
        # LLM 호출
        reply = llm.messages(messages)
        
        # 히스토리 저장
        ChatHistory.objects.create(
            user=user,
            message=message,
            response=reply.text,
            tokens_used=reply.usage.total if reply.usage else 0
        )
        
        return JsonResponse({'response': reply.text})

# 관리자 명령어 (management/commands/chat_stats.py)
from django.core.management.base import BaseCommand
from django.db.models import Sum, Count

class Command(BaseCommand):
    help = '채팅 통계 표시'
    
    def handle(self, *args, **options):
        stats = ChatHistory.objects.aggregate(
            total_chats=Count('id'),
            total_tokens=Sum('tokens_used')
        )
        
        self.stdout.write(
            self.style.SUCCESS(
                f"총 대화 수: {stats['total_chats']}\n"
                f"총 토큰 사용량: {stats['total_tokens']}"
            )
        )
```

### Streamlit 통합

```python
import streamlit as st
from pyhub.llm import LLM
from pyhub.llm.types import Message
import time

# 페이지 설정
st.set_page_config(
    page_title="AI 챗봇",
    page_icon="🤖",
    layout="wide"
)

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'llm' not in st.session_state:
    st.session_state.llm = LLM.create("gpt-4o-mini")

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    
    # 모델 선택
    model = st.selectbox(
        "모델 선택",
        ["gpt-4o-mini", "gpt-4o", "claude-3-haiku-20240307"]
    )
    
    # 온도 설정
    temperature = st.slider("창의성 (Temperature)", 0.0, 2.0, 0.7)
    
    # 시스템 프롬프트
    system_prompt = st.text_area(
        "시스템 프롬프트",
        value="당신은 도움이 되는 AI 어시스턴트입니다."
    )
    
    # 설정 적용
    if st.button("설정 적용"):
        st.session_state.llm = LLM.create(
            model,
            temperature=temperature,
            system_prompt=system_prompt
        )
        st.success("설정이 적용되었습니다!")
    
    # 대화 초기화
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# 메인 채팅 인터페이스
st.title("🤖 AI 챗봇")

# 대화 내역 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 스트리밍 응답
        for chunk in st.session_state.llm.ask(prompt, stream=True):
            full_response += chunk.text
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 추가 기능들
with st.expander("고급 기능"):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("대화 내보내기"):
            import json
            conversation = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
            st.download_button(
                label="JSON으로 다운로드",
                data=conversation,
                file_name="conversation.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("토큰 사용량 확인"):
            # 실제 구현 시 토큰 카운팅 로직 추가
            st.info("이 기능은 구현 중입니다.")

# 파일 업로드 (이미지 분석)
uploaded_file = st.file_uploader(
    "이미지를 업로드하세요",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="업로드된 이미지", use_column_width=True)
    
    if st.button("이미지 분석"):
        with st.spinner("분석 중..."):
            # 임시 파일로 저장
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            # 분석
            reply = st.session_state.llm.ask(
                "이 이미지를 자세히 설명해주세요.",
                files=[tmp_path]
            )
            
            st.write("### 이미지 분석 결과")
            st.write(reply.text)
            
            # 임시 파일 삭제
            import os
            os.unlink(tmp_path)
```

## 도구/함수 호출

### 기본 함수 정의

```python
from pyhub.llm import LLM
from typing import Dict, Any
import json

# 함수 정의
def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """현재 날씨 정보를 가져옵니다."""
    # 실제로는 API 호출
    return {
        "location": location,
        "temperature": 25,
        "unit": unit,
        "condition": "맑음"
    }

def calculate(expression: str) -> float:
    """수학 표현식을 계산합니다."""
    # 안전한 eval 사용
    import ast
    import operator as op
    
    ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow
    }
    
    def eval_expr(expr):
        return eval(expr, {"__builtins__": {}}, {})
    
    try:
        return eval_expr(expression)
    except:
        return "계산할 수 없는 표현식입니다."

# 도구 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 위치의 현재 날씨 정보를 가져옵니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "도시 이름 (예: 서울, 뉴욕)"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "온도 단위"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "수학 표현식을 계산합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "계산할 수학 표현식 (예: 2+2, 10*5)"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# LLM과 함께 사용
llm = LLM.create("gpt-4o-mini")

# 도구와 함께 질문
response = llm.ask_with_tools(
    "서울의 현재 날씨는 어때? 그리고 섭씨 25도는 화씨로 몇 도야?",
    tools=tools
)

# 함수 호출 처리
if response.tool_calls:
    for tool_call in response.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # 실제 함수 호출
        if function_name == "get_weather":
            result = get_weather(**arguments)
        elif function_name == "calculate":
            result = calculate(**arguments)
        
        print(f"함수 {function_name} 호출: {arguments}")
        print(f"결과: {result}")
```

### 클래스 기반 도구

```python
class ToolHandler:
    """도구 처리를 위한 기본 클래스"""
    
    def __init__(self):
        self.tools = []
        self.functions = {}
    
    def register(self, func, description: str, parameters: dict):
        """함수를 도구로 등록"""
        tool = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters
            }
        }
        self.tools.append(tool)
        self.functions[func.__name__] = func
        return func
    
    def execute(self, tool_call):
        """도구 호출 실행"""
        func_name = tool_call.function.name
        if func_name in self.functions:
            args = json.loads(tool_call.function.arguments)
            return self.functions[func_name](**args)
        else:
            raise ValueError(f"Unknown function: {func_name}")

# 사용 예
handler = ToolHandler()

@handler.register(
    description="두 숫자를 더합니다",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "첫 번째 숫자"},
            "b": {"type": "number", "description": "두 번째 숫자"}
        },
        "required": ["a", "b"]
    }
)
def add(a: float, b: float) -> float:
    return a + b

@handler.register(
    description="텍스트를 번역합니다",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "번역할 텍스트"},
            "target_lang": {"type": "string", "description": "목표 언어 코드"}
        },
        "required": ["text", "target_lang"]
    }
)
def translate(text: str, target_lang: str) -> str:
    # 실제로는 번역 API 호출
    translations = {
        "ko": {"Hello": "안녕하세요", "Thank you": "감사합니다"},
        "ja": {"Hello": "こんにちは", "Thank you": "ありがとう"},
        "es": {"Hello": "Hola", "Thank you": "Gracias"}
    }
    return translations.get(target_lang, {}).get(text, text)

# LLM과 통합
llm = LLM.create("gpt-4o-mini")
response = llm.ask_with_tools(
    "Hello를 한국어와 일본어로 번역해주고, 5 더하기 3은 얼마인지 계산해줘",
    tools=handler.tools
)

# 응답 처리
for tool_call in response.tool_calls:
    result = handler.execute(tool_call)
    print(f"{tool_call.function.name}: {result}")
```

### 비동기 도구 호출

```python
import aiohttp
import asyncio

class AsyncToolHandler:
    """비동기 도구 처리기"""
    
    def __init__(self):
        self.tools = []
        self.async_functions = {}
    
    def register_async(self, func, description: str, parameters: dict):
        """비동기 함수를 도구로 등록"""
        tool = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters
            }
        }
        self.tools.append(tool)
        self.async_functions[func.__name__] = func
        return func
    
    async def execute_async(self, tool_call):
        """비동기 도구 실행"""
        func_name = tool_call.function.name
        if func_name in self.async_functions:
            args = json.loads(tool_call.function.arguments)
            return await self.async_functions[func_name](**args)
        else:
            raise ValueError(f"Unknown function: {func_name}")

# 비동기 도구 정의
async_handler = AsyncToolHandler()

@async_handler.register_async(
    description="웹페이지의 제목을 가져옵니다",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "웹페이지 URL"}
        },
        "required": ["url"]
    }
)
async def get_page_title(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            text = await response.text()
            # 간단한 제목 추출
            import re
            match = re.search(r'<title>(.*?)</title>', text, re.IGNORECASE)
            return match.group(1) if match else "제목을 찾을 수 없습니다"

# 비동기 실행
async def main():
    llm = LLM.create("gpt-4o-mini")
    response = await llm.ask_with_tools_async(
        "Python 공식 웹사이트의 제목을 알려줘",
        tools=async_handler.tools
    )
    
    for tool_call in response.tool_calls:
        result = await async_handler.execute_async(tool_call)
        print(f"결과: {result}")

asyncio.run(main())
```

## 체이닝

### 기본 체이닝

```python
from pyhub.llm import LLM, SequentialChain

# 여러 LLM을 연결
chain = SequentialChain([
    LLM.create("gpt-4o-mini", system_prompt="한국어를 영어로 번역"),
    LLM.create("claude-3-haiku-20240307", system_prompt="영어를 일본어로 번역"),
    LLM.create("gpt-4o-mini", system_prompt="일본어를 다시 한국어로 번역")
])

# 체인 실행
result = chain.ask("안녕하세요, 오늘 날씨가 좋네요.")
print(f"최종 결과: {result.text}")

# 각 단계의 결과 확인
for i, step_result in enumerate(result.steps):
    print(f"단계 {i+1}: {step_result.text}")
```

### 파이프 연산자 사용

```python
# 파이프 연산자로 체인 구성
translator = LLM.create("gpt-4o-mini", system_prompt="한국어를 영어로 번역")
analyzer = LLM.create("gpt-4o-mini", system_prompt="텍스트의 감정을 분석")
summarizer = LLM.create("gpt-4o-mini", system_prompt="핵심 내용을 한 문장으로 요약")

# 체인 구성
chain = translator | analyzer | summarizer

# 실행
result = chain.ask("정말 기쁜 일이 생겼어요! 드디어 취업에 성공했답니다.")
print(result.text)
```

### 조건부 체이닝

```python
class ConditionalChain:
    """조건에 따라 다른 체인을 실행"""
    
    def __init__(self):
        self.classifier = LLM.create(
            "gpt-4o-mini",
            system_prompt="텍스트의 주제를 분류"
        )
        
        self.tech_chain = LLM.create(
            "gpt-4o-mini",
            system_prompt="기술 관련 질문에 전문적으로 답변"
        )
        
        self.general_chain = LLM.create(
            "gpt-4o-mini",
            system_prompt="일반적인 대화를 친근하게"
        )
    
    def process(self, text: str) -> str:
        # 먼저 분류
        classification = self.classifier.ask(
            text,
            choices=["기술", "일반"]
        )
        
        # 분류에 따라 다른 체인 사용
        if classification.choice == "기술":
            return self.tech_chain.ask(text).text
        else:
            return self.general_chain.ask(text).text

# 사용
conditional = ConditionalChain()
print(conditional.process("파이썬에서 데코레이터는 어떻게 작동하나요?"))
print(conditional.process("오늘 점심 뭐 먹을까요?"))
```

### 병렬 처리 체인

```python
async def parallel_chain(text: str):
    """여러 분석을 병렬로 수행"""
    
    sentiment_llm = LLM.create("gpt-4o-mini", system_prompt="감정 분석")
    keyword_llm = LLM.create("gpt-4o-mini", system_prompt="키워드 추출")
    summary_llm = LLM.create("gpt-4o-mini", system_prompt="요약")
    
    # 병렬 실행
    tasks = [
        sentiment_llm.ask_async(text),
        keyword_llm.ask_async(f"다음 텍스트의 키워드 3개: {text}"),
        summary_llm.ask_async(f"한 문장으로 요약: {text}")
    ]
    
    results = await asyncio.gather(*tasks)
    
    return {
        "sentiment": results[0].text,
        "keywords": results[1].text,
        "summary": results[2].text
    }

# 실행
text = "인공지능 기술의 발전으로 많은 산업이 변화하고 있습니다. 특히 자동화와 효율성 측면에서 큰 진전이 있었습니다."
result = asyncio.run(parallel_chain(text))
for key, value in result.items():
    print(f"{key}: {value}")
```

## 에러 처리

### 기본 에러 처리

```python
from pyhub.llm import LLM
from pyhub.llm.exceptions import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError
)

def safe_llm_call(prompt: str, max_retries: int = 3):
    """재시도 로직이 포함된 안전한 LLM 호출"""
    llm = LLM.create("gpt-4o-mini")
    
    for attempt in range(max_retries):
        try:
            return llm.ask(prompt)
        
        except RateLimitError as e:
            # 속도 제한 에러 - 대기 후 재시도
            wait_time = getattr(e, 'retry_after', 60)
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            
        except AuthenticationError:
            # 인증 에러 - 재시도 불가
            print("Authentication failed. Check your API key.")
            raise
            
        except InvalidRequestError as e:
            # 잘못된 요청 - 수정 필요
            print(f"Invalid request: {e}")
            raise
            
        except LLMError as e:
            # 기타 LLM 에러
            print(f"LLM error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 지수 백오프
    
    raise Exception("Max retries exceeded")
```

### 폴백 처리

```python
class LLMWithFallback:
    """폴백 LLM이 있는 래퍼"""
    
    def __init__(self, primary_model: str, fallback_model: str):
        self.primary = LLM.create(primary_model)
        self.fallback = LLM.create(fallback_model)
    
    def ask(self, prompt: str, **kwargs):
        try:
            return self.primary.ask(prompt, **kwargs)
        except Exception as e:
            print(f"Primary LLM failed: {e}. Using fallback...")
            return self.fallback.ask(prompt, **kwargs)

# 사용
llm = LLMWithFallback("gpt-4o", "gpt-4o-mini")
reply = llm.ask("복잡한 수학 문제...")
```

### 타임아웃 처리

```python
import asyncio
from asyncio import TimeoutError

async def ask_with_timeout(llm, prompt: str, timeout: float = 30.0):
    """타임아웃이 있는 LLM 호출"""
    try:
        return await asyncio.wait_for(
            llm.ask_async(prompt),
            timeout=timeout
        )
    except TimeoutError:
        print(f"Request timed out after {timeout} seconds")
        # 더 간단한 프롬프트로 재시도
        simplified = f"간단히 답해주세요: {prompt}"
        return await llm.ask_async(simplified)

# 사용
llm = LLM.create("gpt-4o-mini")
result = asyncio.run(ask_with_timeout(llm, "매우 복잡한 질문...", timeout=10))
```

## 실용적인 예제

### 챗봇 구현

```python
class AdvancedChatBot:
    """고급 기능이 포함된 챗봇"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.llm = LLM.create(model)
        self.history = []
        self.user_preferences = {}
        
    def chat(self, message: str) -> str:
        # 사용자 의도 파악
        intent = self._analyze_intent(message)
        
        # 의도에 따른 처리
        if intent == "personal_info":
            return self._handle_personal_info(message)
        elif intent == "recommendation":
            return self._handle_recommendation(message)
        else:
            return self._handle_general_chat(message)
    
    def _analyze_intent(self, message: str) -> str:
        reply = self.llm.ask(
            message,
            choices=["personal_info", "recommendation", "general_chat"]
        )
        return reply.choice
    
    def _handle_personal_info(self, message: str) -> str:
        # 개인 정보 추출 및 저장
        from pydantic import BaseModel
        
        class PersonalInfo(BaseModel):
            name: str = None
            preferences: List[str] = []
            interests: List[str] = []
        
        reply = self.llm.ask(
            f"다음 메시지에서 개인 정보 추출: {message}",
            schema=PersonalInfo
        )
        
        info = reply.structured_data
        if info.name:
            self.user_preferences['name'] = info.name
        if info.preferences:
            self.user_preferences['preferences'] = info.preferences
        
        return f"알겠습니다! 기억하고 있을게요."
    
    def _handle_recommendation(self, message: str) -> str:
        # 사용자 선호도 기반 추천
        context = f"사용자 정보: {self.user_preferences}\n질문: {message}"
        reply = self.llm.ask(context)
        return reply.text
    
    def _handle_general_chat(self, message: str) -> str:
        # 일반 대화
        self.history.append(Message(role="user", content=message))
        reply = self.llm.messages(self.history[-10:])  # 최근 10개 메시지
        self.history.append(Message(role="assistant", content=reply.text))
        return reply.text
```

### 문서 요약기

```python
class DocumentSummarizer:
    """계층적 문서 요약기"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.llm = LLM.create(model)
        self.chunk_size = 2000  # 토큰 기준
    
    def summarize(self, text: str, max_length: int = 500) -> str:
        """긴 문서를 계층적으로 요약"""
        
        # 텍스트가 짧으면 직접 요약
        if len(text) < self.chunk_size:
            return self._simple_summarize(text, max_length)
        
        # 청크로 분할
        chunks = self._split_into_chunks(text)
        
        # 각 청크 요약
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"청크 {i+1}/{len(chunks)} 요약 중...")
            summary = self._simple_summarize(chunk, max_length // len(chunks))
            chunk_summaries.append(summary)
        
        # 요약들을 다시 요약
        combined = "\n\n".join(chunk_summaries)
        final_summary = self._simple_summarize(
            f"다음 요약들을 종합하세요:\n{combined}",
            max_length
        )
        
        return final_summary
    
    def _simple_summarize(self, text: str, max_length: int) -> str:
        """단순 요약"""
        prompt = f"다음 텍스트를 {max_length}자 이내로 요약하세요:\n\n{text}"
        reply = self.llm.ask(prompt)
        return reply.text
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) > self.chunk_size:
                chunks.append(" ".join(current_chunk[:-1]))
                current_chunk = [word]
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

# 사용 예
summarizer = DocumentSummarizer()
with open("long_document.txt", "r", encoding="utf-8") as f:
    document = f.read()

summary = summarizer.summarize(document, max_length=300)
print(summary)
```

### 코드 리뷰어

```python
class CodeReviewer:
    """AI 코드 리뷰어"""
    
    def __init__(self, model="gpt-4o"):
        self.llm = LLM.create(
            model,
            system_prompt="당신은 경험 많은 소프트웨어 엔지니어입니다."
        )
    
    def review_code(self, code: str, language: str = "python") -> dict:
        """코드를 다각도로 리뷰"""
        
        from pydantic import BaseModel, Field
        
        class CodeReview(BaseModel):
            summary: str = Field(description="전체적인 평가")
            issues: List[str] = Field(description="발견된 문제점들")
            improvements: List[str] = Field(description="개선 제안사항")
            security: List[str] = Field(description="보안 관련 사항")
            performance: List[str] = Field(description="성능 관련 사항")
            best_practices: List[str] = Field(description="베스트 프랙티스")
            score: int = Field(description="전체 점수 (0-100)", ge=0, le=100)
        
        prompt = f"""
다음 {language} 코드를 리뷰해주세요:

```{language}
{code}
```

코드의 품질, 보안, 성능, 가독성 등을 종합적으로 평가해주세요.
"""
        
        reply = self.llm.ask(prompt, schema=CodeReview)
        return reply.structured_data.dict()
    
    def suggest_refactoring(self, code: str) -> str:
        """리팩토링 제안"""
        prompt = f"""
다음 코드를 더 깔끔하고 효율적으로 리팩토링해주세요:

```python
{code}
```

리팩토링된 코드와 함께 변경 사항을 설명해주세요.
"""
        reply = self.llm.ask(prompt)
        return reply.text

# 사용 예
reviewer = CodeReviewer()

code = """
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total

result = calculate_sum([1, 2, 3, 4, 5])
print("Sum is: " + str(result))
"""

review = reviewer.review_code(code)
print(f"점수: {review['score']}/100")
print(f"문제점: {review['issues']}")
print(f"개선사항: {review['improvements']}")

# 리팩토링 제안
refactored = reviewer.suggest_refactoring(code)
print(f"\n리팩토링 제안:\n{refactored}")
```

### 번역기

```python
class SmartTranslator:
    """컨텍스트를 이해하는 스마트 번역기"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.llm = LLM.create(model)
        self.context_history = []
    
    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        style: str = "formal"
    ) -> dict:
        """고급 번역 기능"""
        
        from pydantic import BaseModel
        
        class Translation(BaseModel):
            translated_text: str
            detected_language: str = None
            confidence: float = Field(ge=0, le=1)
            alternatives: List[str] = []
            notes: str = None
        
        # 컨텍스트 포함 프롬프트
        context = ""
        if self.context_history:
            context = f"이전 번역 컨텍스트:\n"
            for prev in self.context_history[-3:]:
                context += f"- {prev}\n"
        
        prompt = f"""
{context}

다음 텍스트를 {target_lang}로 번역하세요.
스타일: {style}
원문: {text}

문화적 뉘앙스와 관용표현을 고려하여 자연스럽게 번역해주세요.
"""
        
        reply = self.llm.ask(prompt, schema=Translation)
        translation = reply.structured_data
        
        # 컨텍스트 업데이트
        self.context_history.append(f"{text} → {translation.translated_text}")
        
        return translation.dict()
    
    def translate_document(self, file_path: str, target_lang: str) -> str:
        """문서 전체 번역"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 단락별로 번역
        paragraphs = content.split('\n\n')
        translated_paragraphs = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                print(f"번역 중... {i+1}/{len(paragraphs)}")
                result = self.translate(para, target_lang)
                translated_paragraphs.append(result['translated_text'])
            else:
                translated_paragraphs.append("")
        
        return '\n\n'.join(translated_paragraphs)

# 사용 예
translator = SmartTranslator()

# 단일 번역
result = translator.translate(
    "The early bird catches the worm",
    target_lang="한국어",
    style="casual"
)
print(f"번역: {result['translated_text']}")
print(f"대안: {result['alternatives']}")

# 연속 번역 (컨텍스트 유지)
texts = [
    "I love programming.",
    "It's like solving puzzles.",
    "Each bug is a new challenge."
]

for text in texts:
    result = translator.translate(text, "한국어")
    print(f"{text} → {result['translated_text']}")
```

### Q&A 시스템

```python
class QASystem:
    """문서 기반 Q&A 시스템"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.llm = LLM.create(model)
        self.documents = []
        self.embeddings = []
        self.embedding_model = LLM.create("text-embedding-3-small")
    
    def add_document(self, text: str, metadata: dict = None):
        """문서 추가"""
        # 문서를 청크로 분할
        chunks = self._split_text(text, chunk_size=500)
        
        for chunk in chunks:
            self.documents.append({
                "text": chunk,
                "metadata": metadata or {}
            })
            
            # 임베딩 생성
            embedding = self.embedding_model.embed(chunk)
            self.embeddings.append(embedding.embeddings[0])
    
    def ask(self, question: str, top_k: int = 3) -> str:
        """질문에 대한 답변"""
        # 질문 임베딩
        q_embedding = self.embedding_model.embed(question).embeddings[0]
        
        # 관련 문서 검색
        relevant_docs = self._find_relevant_docs(q_embedding, top_k)
        
        # 컨텍스트 구성
        context = "\n\n".join([doc["text"] for doc in relevant_docs])
        
        # 답변 생성
        prompt = f"""
다음 문서들을 참고하여 질문에 답변해주세요.

문서:
{context}

질문: {question}

답변:"""
        
        reply = self.llm.ask(prompt)
        
        # 출처 포함
        sources = [doc.get("metadata", {}).get("source", "Unknown") 
                  for doc in relevant_docs]
        
        return {
            "answer": reply.text,
            "sources": list(set(sources)),
            "confidence": self._calculate_confidence(relevant_docs)
        }
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """텍스트를 청크로 분할"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    def _find_relevant_docs(self, query_embedding, top_k: int):
        """관련 문서 검색"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append((sim, i))
        
        similarities.sort(reverse=True)
        
        relevant_docs = []
        for sim, idx in similarities[:top_k]:
            doc = self.documents[idx].copy()
            doc['similarity'] = sim
            relevant_docs.append(doc)
        
        return relevant_docs
    
    def _calculate_confidence(self, docs: List[dict]) -> float:
        """답변 신뢰도 계산"""
        if not docs:
            return 0.0
        
        # 평균 유사도를 신뢰도로 사용
        avg_similarity = sum(doc['similarity'] for doc in docs) / len(docs)
        return min(avg_similarity * 1.2, 1.0)  # 0-1 범위로 정규화

# 사용 예
qa = QASystem()

# 문서 추가
qa.add_document(
    "파이썬은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어입니다. "
    "파이썬은 코드의 가독성을 중시하며, 간결한 문법을 가지고 있습니다.",
    metadata={"source": "Python Wikipedia"}
)

qa.add_document(
    "파이썬은 다양한 프로그래밍 패러다임을 지원합니다. "
    "객체 지향, 함수형, 절차적 프로그래밍이 모두 가능합니다.",
    metadata={"source": "Python Documentation"}
)

# 질문
result = qa.ask("파이썬은 누가 만들었나요?")
print(f"답변: {result['answer']}")
print(f"출처: {result['sources']}")
print(f"신뢰도: {result['confidence']:.2%}")
```

이 CHEATSHEET는 pyhub-llm의 다양한 기능과 활용 방법을 보여줍니다. 각 예제는 실제로 사용 가능한 코드이며, 필요에 따라 수정하여 사용할 수 있습니다.