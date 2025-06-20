# pyhub-llm 중급 가이드

pyhub-llm의 고급 기능들을 활용하여 더 복잡하고 효율적인 LLM 애플리케이션을 구축하는 방법을 배워보세요. 이 가이드는 구조화된 출력, 비동기 처리, 캐싱, 도구 호출 등 중급 수준의 기능들을 다룹니다.

> 💡 예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.

## 목차

- [구조화된 출력](#구조화된-출력)
- [분류 및 선택](#분류-및-선택)
- [비동기 처리](#비동기-처리)
- [캐싱](#캐싱)
- [도구/함수 호출](#도구함수-호출)
- [템플릿 활용](#템플릿-활용)
- [프롬프트 허브 (Hub)](#프롬프트-허브-hub)
- [History Backup](#history-backup)
- [다음 단계](#다음-단계)
## 구조화된 출력

💻 [실행 가능한 예제](examples/intermediate/01_structured_output.py)

### Pydantic 스키마 사용

```python
from pydantic import BaseModel, Field
from typing import List
from pyhub.llm import OpenAILLM

class BookInfo(BaseModel):
    title: str = Field(description="책 제목")
    author: str = Field(description="저자")
    year: int = Field(description="출판 연도")
    genres: List[str] = Field(description="장르 목록")
    summary: str = Field(description="간단한 줄거리")

llm = OpenAILLM(model="gpt-4o-mini")
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

llm = OpenAILLM(model="gpt-4o-mini")
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

llm = OpenAILLM(model="gpt-4o-mini", system_prompt="다국어 번역 전문가")
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
from pyhub.llm import OpenAILLM

llm = OpenAILLM(model="gpt-4o-mini")

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
        self.llm = OpenAILLM(model="gpt-4o-mini")
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

llm = OpenAILLM(model="gpt-4o-mini", system_prompt="텍스트의 주제를 분류하는 전문가")

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

💻 [실행 가능한 예제](examples/intermediate/02_async_processing.py)

### 기본 비동기 사용

```python
import asyncio
from pyhub.llm import OpenAILLM

async def main():
    llm = OpenAILLM(model="gpt-4o-mini")
    
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
    llm = OpenAILLM(model="gpt-4o-mini")
    
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

💻 [실행 가능한 예제](examples/mcp_integration_example.py)

```python
from pyhub.llm import OpenAILLM

async def main():
    # 간편한 문자열 설정으로 MCP 서버와 함께 LLM 생성
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

💻 [실행 가능한 예제](examples/intermediate/03_caching.py)

### 인메모리 캐싱

```python
from pyhub.llm import OpenAILLM
from pyhub.llm.cache import InMemoryCache

# 캐시 설정
cache = InMemoryCache(ttl=3600)  # 1시간 TTL
llm = OpenAILLM(model="gpt-4o-mini", cache=cache)

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

llm = OpenAILLM(model="gpt-4o-mini", cache=cache)

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


## 도구/함수 호출

### 기본 함수 정의

```python
from pyhub.llm import OpenAILLM
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
llm = OpenAILLM(model="gpt-4o-mini")

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
llm = OpenAILLM(model="gpt-4o-mini")
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
    llm = OpenAILLM(model="gpt-4o-mini")
    response = await llm.ask_with_tools_async(
        "Python 공식 웹사이트의 제목을 알려줘",
        tools=async_handler.tools
    )
    
    for tool_call in response.tool_calls:
        result = await async_handler.execute_async(tool_call)
        print(f"결과: {result}")

asyncio.run(main())
```


## 템플릿 활용

### Jinja2 템플릿

```python
from pyhub.llm import OpenAILLM
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

llm = OpenAILLM(model="gpt-4o-mini")

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
llm = OpenAILLM(model="gpt-4o-mini")

# 같은 주제를 다른 스타일로
topic = "블록체인 기술"
for style in ["technical", "simple", "business"]:
    prompt = builder.build(style, topic)
    reply = llm.ask(prompt)
    print(f"\n[{style.upper()}]\n{reply.text[:200]}...")
```


## 프롬프트 허브 (Hub)

💻 [실행 가능한 예제](examples/advanced/15_hub_prompts.py)

pyhub-llm은 LangChain Hub와 호환되는 프롬프트 관리 시스템을 제공합니다. 인기 있는 프롬프트를 쉽게 가져와 사용하고, 자신만의 프롬프트를 저장하고 공유할 수 있습니다.

### 내장 프롬프트 사용

```python
from pyhub.llm import hub, OpenAILLM

# RAG (Retrieval Augmented Generation) 프롬프트
rag_prompt = hub.pull("rlm/rag-prompt")
print(f"입력 변수: {rag_prompt.input_variables}")  # ['context', 'question']

# 프롬프트 포맷팅
formatted = rag_prompt.format(
    context="에펠탑은 1889년 파리 만국박람회를 위해 건설된 철제 탑입니다. 높이는 330미터입니다.",
    question="에펠탑의 높이는 얼마인가요?"
)

# LLM과 함께 사용
llm = OpenAILLM(model="gpt-4o-mini")
answer = llm.ask(formatted)
print(answer.text)  # "에펠탑의 높이는 330미터입니다."
```

### ReAct 에이전트 프롬프트

```python
# ReAct (Reasoning + Acting) 프롬프트
react_prompt = hub.pull("hwchase17/react")

# 도구 정보와 함께 포맷팅
formatted = react_prompt.format(
    tools="Calculator: 수학 계산을 수행합니다.\nSearch: 인터넷에서 정보를 검색합니다.",
    tool_names="Calculator, Search",
    input="에펠탑이 완공된 연도의 제곱근은 얼마인가요?"
)

print(formatted[:200] + "...")  # 프롬프트 미리보기
```

### 커스텀 프롬프트 생성 및 저장

```python
from pyhub.llm.templates import PromptTemplate

# 커스텀 프롬프트 생성
custom_prompt = PromptTemplate(
    template="""당신은 {language} 전문가입니다.
    
다음 코드를 검토하고 개선점을 제안해주세요:

```{language}
{code}
```

검토 기준:
- 가독성
- 성능
- 보안
- 모범 사례""",
    input_variables=["language", "code"],
    metadata={
        "description": "코드 리뷰 프롬프트",
        "author": "myteam",
        "tags": ["code-review", "programming"],
        "version": "1.0.0"
    }
)

# 로컬에 저장
hub.push("myteam/code-reviewer", custom_prompt)

# 나중에 다시 사용
saved_prompt = hub.pull("myteam/code-reviewer")
```

### Partial 프롬프트

```python
# 기본 프롬프트
base_prompt = hub.pull("rlm/rag-prompt")

# 컨텍스트를 미리 채운 partial 프롬프트 생성
python_qa_prompt = base_prompt.partial(
    context="""Python은 1991년 Guido van Rossum이 개발한 프로그래밍 언어입니다.
    동적 타이핑, 자동 메모리 관리, 다양한 프로그래밍 패러다임을 지원합니다."""
)

# 이제 질문만 제공하면 됨
questions = [
    "Python은 누가 만들었나요?",
    "Python의 특징은 무엇인가요?",
    "Python은 언제 만들어졌나요?"
]

llm = OpenAILLM(model="gpt-4o-mini")
for question in questions:
    formatted = python_qa_prompt.format(question=question)
    answer = llm.ask(formatted)
    print(f"Q: {question}")
    print(f"A: {answer.text}\n")
```

### 프롬프트 버전 관리 (향후 지원 예정)

```python
# 현재는 최신 버전만 지원
prompt = hub.pull("rlm/rag-prompt")

# 향후 버전 지정 지원 예정
# prompt = hub.pull("rlm/rag-prompt", version="1.0.0")
# prompt = hub.pull("rlm/rag-prompt:v1.0.0")
```

### 사용 가능한 내장 프롬프트

| 프롬프트 이름 | 설명 | 입력 변수 |
|-------------|------|----------|
| `rlm/rag-prompt` | RAG 질문-답변 | `context`, `question` |
| `hwchase17/react` | ReAct 에이전트 | `tools`, `tool_names`, `input` |
| `hwchase17/openai-functions-agent` | OpenAI 함수 호출 | `tools`, `history`, `input` |
| `hwchase17/structured-chat-agent` | JSON 출력 에이전트 | `tools`, `tool_names`, `history`, `input`, `agent_scratchpad` |

### 프롬프트 목록 확인

```python
# 사용 가능한 모든 프롬프트 확인
all_prompts = hub.list_prompts()
for prompt_name in all_prompts:
    print(f"- {prompt_name}")
```

### Jinja2 템플릿 형식

```python
# Jinja2 형식의 프롬프트
jinja_prompt = PromptTemplate(
    template="""
{% for item in items %}
{{ loop.index }}. {{ item.name }}: {{ item.description }}
{% endfor %}

위 항목 중에서 {{ criteria }}에 가장 적합한 것을 선택하세요.
""",
    input_variables=["items", "criteria"],
    template_format="jinja2"
)

items = [
    {"name": "Python", "description": "간결하고 읽기 쉬운 언어"},
    {"name": "Java", "description": "엔터프라이즈급 애플리케이션에 적합"},
    {"name": "JavaScript", "description": "웹 개발의 필수 언어"}
]

formatted = jinja_prompt.format(
    items=items,
    criteria="초보자가 배우기 쉬운 언어"
)
```


## History Backup

대화 히스토리를 외부 저장소에 백업하고 복원하는 기능입니다. 메모리 기반 히스토리와 별도로 영구 저장소에 대화 내역을 보관할 수 있습니다.

💻 [실행 가능한 예제](examples/history_backup_example.py)

### 기본 사용법 (InMemoryHistoryBackup)

```python
from pyhub.llm import OpenAILLM
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
from pyhub.llm import OpenAILLM
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


## 다음 단계

이제 pyhub-llm의 중급 기능들을 익히셨습니다\! 더 깊이 있는 학습을 위해 다음을 추천합니다:

### 고급 가이드로 이동
- **MCP (Model Context Protocol) 통합**: 외부 도구와 서비스를 LLM에 연결
- **웹 프레임워크 통합**: FastAPI, Django와의 통합
- **체이닝**: 여러 LLM 호출을 연결하여 복잡한 워크플로우 구성
- **에러 처리**: 강력한 에러 처리 및 재시도 전략
- **실용적인 예제**: 실제 프로덕션 환경에서의 활용 사례

### 추가 학습 자료
- [전체 가이드 보기](CHEATSHEET.md)
- [API 문서](https://pyhub-llm.readthedocs.io)
- [예제 코드](examples/)
- [GitHub 저장소](https://github.com/pyhub-llm/pyhub-llm)

### 커뮤니티
- 질문이나 피드백은 GitHub Issues에 남겨주세요
- 기여를 환영합니다\! Contributing 가이드를 참고하세요

Happy coding with pyhub-llm\! 🚀
EOF < /dev/null