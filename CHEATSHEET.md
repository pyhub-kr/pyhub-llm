# pyhub-llm CHEATSHEET

## 라이브러리 설치

```
python -m pip install "pyhub-llm[all]"
```

## LLM 선택

### OpenAI 활용

```python
from pyhub.llm import OpenAILLM
from pyhub.llm.types import Reply

api_key = None  # TODO: UPSTAGE_API_KEY 환경변수 지정하기
# 디폴트 model : gpt-4o-mini
llm = OpenAILLM(api_key=api_key)

reply: Reply = llm.ask("hello")
print(reply.text)
```

### 업스테이지 활용

```python
from pyhub.llm import UpstageLLM
from pyhub.llm.types import Reply

api_key = None  # TODO: UPSTAGE_API_KEY 환경변수 지정하기
# 디폴트 model : solar-mini
llm = UpstageLLM(
    api_key=api_key,
    # model="solar-mini",  # default
    # model="solar-pro2-preview",  # 7월 15일까지 무료
)

reply: Reply = llm.ask("hello")
print(reply.text)
```

## 스트리밍 출력

```python
from pyhub.llm import UpstageLLM

api_key = None  # TODO: UPSTAGE_API_KEY 환경변수 지정하기
llm = UpstageLLM(api_key=api_key)

for chunk in llm.ask("hello", stream=True):
    print(chunk.text, end="")
print()
```

## 분류 1

`choices` 인자 지정 만으로 별도의 프롬프트없이 지정 선택지 중에 하나만 선택해주기를 LLM에게 요청.

```python
from pyhub.llm import UpstageLLM

# 시스템 프롬프트없이도, LLM이 똑똑하기에 선택을 해줍니다.
llm = UpstageLLM()  #system_prompt="유저 메시지의 감정은?")

reply = llm.ask("우울해서 빵을 샀어.", choices=["기쁨", "슬픔", "분노", "불안", "무기력함"])
print(reply.choice)        # "슬픔"
print(reply.choice_index)  # 1
```

## 다국어 응답 생성

`schema` 인자를 지정하면 `OpenAI`, `Upstage`에서는 Structured Output API를 활용하고,
다른 LLM API에서는 프롬프트를 통해 선택지를 강제합니다.

```python
from pydantic import BaseModel
from pyhub.llm import UpstageLLM

class ReplySchema(BaseModel):
    korean: str
    english: str

llm = UpstageLLM(system_prompt="감정에 맞는 철학자 명언을 추천하고, 한국어와 영어로 제공해줘")
reply = llm.ask("기쁨", schema=ReplySchema)
obj: ReplySchema = reply.structured_data
print(obj.korean)  # 행복이란 자기가 좋아하는 일을 하는 것이다. - 마하트마 간디
print(obj.english)  # Happiness is when what you think, what you say, and what you do are in harmony. - Mahatma Gandhi
```

## 분류 2

```python
from pyhub.llm import UpstageLLM

def 요리_질문_여부(s: str) -> bool:
    llm = UpstageLLM(system_prompt="요리 관련 질문인지 여부를 판정")
    reply = llm.ask(s, choices=["t", "f"])
    return reply.choice_index == 0

print(요리_질문_여부("자전거 타는 법"))  # False
```
