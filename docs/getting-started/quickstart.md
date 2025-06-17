# 빠른 시작

5분 안에 pyhub-llm으로 첫 번째 LLM 애플리케이션을 만들어봅시다!

## 1. 첫 번째 대화

가장 간단한 예제부터 시작해봅시다:

```python
from pyhub.llm import LLM

# LLM 인스턴스 생성 (모델명으로 자동 프로바이더 감지)
llm = LLM.create("gpt-4o-mini")

# 질문하기
reply = llm.ask("파이썬의 장점을 3가지만 알려주세요")
print(reply.text)
```

!!! success "출력 예시"
    ```
    파이썬의 주요 장점 3가지는 다음과 같습니다:
    
    1. **간결하고 읽기 쉬운 문법**: 자연어에 가까운 문법으로 초보자도 쉽게 배울 수 있습니다.
    2. **풍부한 라이브러리**: 데이터 분석, 웹 개발, AI/ML 등 다양한 분야의 라이브러리가 있습니다.
    3. **높은 생산성**: 빠른 개발과 프로토타이핑이 가능하여 개발 시간을 단축할 수 있습니다.
    ```

## 2. 프로바이더 전환하기

다른 LLM 프로바이더로 쉽게 전환할 수 있습니다:

```python
# OpenAI
openai_llm = LLM.create("gpt-4o-mini")

# Anthropic Claude
claude_llm = LLM.create("claude-3-5-haiku-latest")

# Google Gemini
gemini_llm = LLM.create("gemini-2.0-flash-exp")

# 모두 동일한 인터페이스 사용
question = "머신러닝이란 무엇인가요?"
for llm in [openai_llm, claude_llm, gemini_llm]:
    reply = llm.ask(question)
    print(f"\n{llm.model}의 답변:")
    print(reply.text[:100] + "...")
```

## 3. 대화 이어가기

pyhub-llm은 자동으로 대화 히스토리를 관리합니다:

```python
llm = LLM.create("gpt-4o-mini")

# 첫 번째 질문
llm.ask("제 이름은 김철수입니다.")

# 이전 대화를 기억하고 답변
reply = llm.ask("제 이름이 뭐라고 했죠?")
print(reply.text)  # "김철수님이라고 말씀하셨습니다."

# 대화 히스토리 확인
print(f"총 {len(llm.history)}개의 메시지")

# 대화 초기화
llm.clear()
```

## 4. 구조화된 출력 받기

응답을 원하는 형식으로 받을 수 있습니다:

### 선택지에서 고르기

```python
# 감정 분석
reply = llm.ask(
    "다음 문장의 감정을 분석해주세요: '오늘 정말 기분이 좋아요!'",
    choices=["긍정", "부정", "중립"]
)

print(f"감정: {reply.choice}")  # "긍정"
print(f"확신도: {reply.confidence}")  # 0.95
```

### Pydantic 스키마 사용

```python
from pydantic import BaseModel
from typing import List

class MovieReview(BaseModel):
    title: str
    rating: float  # 0-10
    pros: List[str]
    cons: List[str]
    recommend: bool

reply = llm.ask(
    "영화 '인터스텔라'에 대한 리뷰를 작성해주세요",
    schema=MovieReview
)

review = reply.structured_data
print(f"제목: {review.title}")
print(f"평점: {review.rating}/10")
print(f"장점: {', '.join(review.pros)}")
print(f"추천 여부: {'추천' if review.recommend else '비추천'}")
```

## 5. 스트리밍 응답

실시간으로 응답을 받아볼 수 있습니다:

```python
# 스트리밍으로 응답 받기
for chunk in llm.ask("파이썬으로 할 수 있는 일들을 설명해주세요", stream=True):
    print(chunk.text, end="", flush=True)
```

## 6. 비동기 처리

비동기로 여러 요청을 동시에 처리할 수 있습니다:

```python
import asyncio

async def ask_multiple():
    llm = LLM.create("gpt-4o-mini")
    
    questions = [
        "Python의 장점은?",
        "JavaScript의 장점은?",
        "Go의 장점은?"
    ]
    
    # 동시에 여러 질문 처리
    tasks = [llm.ask_async(q) for q in questions]
    replies = await asyncio.gather(*tasks)
    
    for q, r in zip(questions, replies):
        print(f"Q: {q}")
        print(f"A: {r.text[:50]}...\n")

# 실행
asyncio.run(ask_multiple())
```

## 7. 이미지와 함께 질문하기

이미지를 포함한 멀티모달 질문도 가능합니다:

```python
# 이미지 파일과 함께 질문
reply = llm.ask(
    "이 이미지에 무엇이 있나요?",
    files=["image.jpg"]
)
print(reply.text)

# 여러 이미지 비교
reply = llm.ask(
    "이 두 이미지의 차이점을 설명해주세요",
    files=["before.png", "after.png"]
)
```

## 8. 독립적인 작업 처리 (Stateless 모드)

반복적인 독립 작업에는 Stateless 모드를 사용하세요:

```python
# Stateless 모드 - 히스토리 저장 안 함
classifier = LLM.create("gpt-4o-mini", stateless=True)

# 대량의 텍스트 분류
texts = [
    "이 제품 정말 좋아요!",
    "배송이 너무 늦어요",
    "품질이 기대 이하네요",
    "다시 구매할 예정입니다"
]

for text in texts:
    reply = classifier.ask(
        f"감정 분석: {text}",
        choices=["긍정", "부정"]
    )
    print(f"{text} -> {reply.choice}")
```

## 다음 단계

축하합니다! 🎉 pyhub-llm의 기본 기능을 모두 살펴보았습니다.

더 자세한 내용을 알아보려면:

- [기본 사용법 가이드](../guides/basic-usage.md) - 더 많은 기본 기능
- [대화 관리](../guides/conversation.md) - 대화 히스토리 관리
- [구조화된 출력](../guides/structured-output.md) - 복잡한 스키마 사용
- [API 레퍼런스](../api-reference/index.md) - 전체 API 문서

## 도움말

문제가 발생했나요?

- 📧 이메일: me@pyhub.kr
- 💬 [GitHub Discussions](https://github.com/pyhub-kr/pyhub-llm/discussions)
- 🐛 [GitHub Issues](https://github.com/pyhub-kr/pyhub-llm/issues)