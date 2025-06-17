# 고급 기능

pyhub-llm의 고급 기능을 활용하여 성능을 최적화하고 복잡한 사용 사례를 구현하는 방법을 알아봅니다.

## 스트리밍 응답

### 기본 스트리밍

```python
from pyhub.llm import LLM

llm = LLM.create("gpt-4o-mini")

# 실시간 스트리밍 출력
for chunk in llm.ask("파이썬의 역사를 자세히 설명해주세요", stream=True):
    print(chunk.text, end="", flush=True)
print()  # 줄바꿈

# 스트리밍 중 통계
total_tokens = 0
chunks_received = 0

for chunk in llm.ask("머신러닝 알고리즘을 설명해주세요", stream=True):
    print(chunk.text, end="", flush=True)
    if chunk.text:
        total_tokens += len(chunk.text.split())
        chunks_received += 1

print(f"\n\n청크 수: {chunks_received}, 대략적인 토큰 수: {total_tokens}")
```

### 스트리밍 중 처리

```python
import time
from typing import List

class StreamProcessor:
    def __init__(self):
        self.buffer = ""
        self.sentences = []
        self.start_time = time.time()
    
    def process_chunk(self, chunk):
        """청크를 처리하고 완성된 문장 반환"""
        self.buffer += chunk.text
        
        # 문장 단위로 분리
        while "." in self.buffer or "!" in self.buffer or "?" in self.buffer:
            # 첫 번째 문장 종료 지점 찾기
            end_idx = min(
                self.buffer.find(".") if "." in self.buffer else float('inf'),
                self.buffer.find("!") if "!" in self.buffer else float('inf'),
                self.buffer.find("?") if "?" in self.buffer else float('inf')
            )
            
            if end_idx != float('inf'):
                sentence = self.buffer[:end_idx + 1].strip()
                self.buffer = self.buffer[end_idx + 1:].lstrip()
                
                if sentence:
                    self.sentences.append(sentence)
                    return sentence
        
        return None
    
    def get_stats(self):
        elapsed = time.time() - self.start_time
        return {
            "sentences": len(self.sentences),
            "elapsed_seconds": elapsed,
            "sentences_per_second": len(self.sentences) / elapsed if elapsed > 0 else 0
        }

# 사용
processor = StreamProcessor()

for chunk in llm.ask("인공지능의 장단점을 5가지씩 설명해주세요", stream=True):
    sentence = processor.process_chunk(chunk)
    if sentence:
        print(f"\n[문장 {len(processor.sentences)}] {sentence}")

stats = processor.get_stats()
print(f"\n통계: {stats}")
```

### 조기 종료

```python
def stream_until_keyword(prompt: str, stop_keyword: str, max_tokens: int = 500):
    """특정 키워드가 나오면 스트리밍 중단"""
    llm = LLM.create("gpt-4o-mini")
    collected_text = ""
    
    try:
        for chunk in llm.ask(prompt, stream=True, max_tokens=max_tokens):
            collected_text += chunk.text
            print(chunk.text, end="", flush=True)
            
            if stop_keyword.lower() in collected_text.lower():
                print(f"\n\n['{stop_keyword}' 키워드 감지 - 중단]")
                break
    except KeyboardInterrupt:
        print("\n\n[사용자 중단]")
    
    return collected_text

# 사용
text = stream_until_keyword(
    "파이썬의 장점을 나열해주세요",
    stop_keyword="단점"
)
```

## 비동기 처리

### 기본 비동기 호출

```python
import asyncio
from typing import List

async def async_example():
    llm = LLM.create("gpt-4o-mini")
    
    # 단일 비동기 호출
    reply = await llm.ask_async("파이썬의 비동기 프로그래밍을 설명해주세요")
    print(reply.text)
    
    # 여러 비동기 호출 동시 실행
    questions = [
        "파이썬이란?",
        "자바스크립트란?",
        "Go 언어란?"
    ]
    
    tasks = [llm.ask_async(q) for q in questions]
    replies = await asyncio.gather(*tasks)
    
    for q, r in zip(questions, replies):
        print(f"\nQ: {q}")
        print(f"A: {r.text[:100]}...")

# 실행
asyncio.run(async_example())
```

### 비동기 스트리밍

```python
async def async_streaming():
    llm = LLM.create("gpt-4o-mini")
    
    # 비동기 스트리밍
    async for chunk in llm.ask_async("AI의 미래를 예측해주세요", stream=True):
        print(chunk.text, end="", flush=True)
        # 다른 비동기 작업 수행 가능
        await asyncio.sleep(0.01)  # 시뮬레이션

# 동시 스트리밍
async def parallel_streaming():
    llm1 = LLM.create("gpt-4o-mini", stateless=True)
    llm2 = LLM.create("claude-3-5-haiku-latest", stateless=True)
    
    async def stream_with_prefix(llm, prompt, prefix):
        async for chunk in llm.ask_async(prompt, stream=True):
            print(f"[{prefix}] {chunk.text}", end="")
    
    # 두 모델에서 동시에 스트리밍
    await asyncio.gather(
        stream_with_prefix(llm1, "파이썬의 장점", "GPT"),
        stream_with_prefix(llm2, "파이썬의 장점", "Claude")
    )

asyncio.run(parallel_streaming())
```

### 비동기 파이프라인

```python
class AsyncPipeline:
    def __init__(self, models: List[str]):
        self.llms = [LLM.create(model, stateless=True) for model in models]
    
    async def process_batch(self, items: List[str], processor_prompt: str):
        """배치 아이템을 병렬 처리"""
        tasks = []
        
        for item in items:
            # 라운드 로빈으로 LLM 할당
            llm = self.llms[len(tasks) % len(self.llms)]
            prompt = processor_prompt.format(item=item)
            tasks.append(llm.ask_async(prompt))
        
        results = await asyncio.gather(*tasks)
        return [(item, result.text) for item, result in zip(items, results)]
    
    async def process_chain(self, initial_input: str, prompts: List[str]):
        """순차적 처리 체인"""
        result = initial_input
        
        for i, prompt in enumerate(prompts):
            llm = self.llms[i % len(self.llms)]
            reply = await llm.ask_async(prompt.format(input=result))
            result = reply.text
            print(f"Step {i+1}: {result[:50]}...")
        
        return result

# 사용
async def pipeline_example():
    pipeline = AsyncPipeline(["gpt-4o-mini", "claude-3-5-haiku-latest"])
    
    # 배치 처리
    items = ["사과", "바나나", "오렌지"]
    results = await pipeline.process_batch(
        items,
        "'{item}'의 영양 성분을 한 줄로 요약하세요"
    )
    
    for item, result in results:
        print(f"{item}: {result}")
    
    # 체인 처리
    final = await pipeline.process_chain(
        "인공지능",
        [
            "{input}의 정의를 한 문장으로",
            "{input}를 5살 아이에게 설명하면",
            "{input}를 이모티콘으로만 표현하면"
        ]
    )
    print(f"\n최종 결과: {final}")

asyncio.run(pipeline_example())
```

## 캐싱

### 기본 캐싱

```python
from pyhub.llm.cache import InMemoryCache, FileCache

# 인메모리 캐시
memory_cache = InMemoryCache(max_size=100, ttl=3600)  # 100개, 1시간
llm_cached = LLM.create("gpt-4o-mini", cache=memory_cache)

# 첫 호출 - API 요청
reply1 = llm_cached.ask("파이썬의 창시자는 누구인가요?")
print(f"첫 번째 호출: {reply1.text}")
print(f"캐시 히트: {reply1.cache_hit}")  # False

# 두 번째 호출 - 캐시에서
reply2 = llm_cached.ask("파이썬의 창시자는 누구인가요?")
print(f"두 번째 호출: {reply2.text}")
print(f"캐시 히트: {reply2.cache_hit}")  # True
```

### 파일 기반 캐싱

```python
# 영구 파일 캐시
file_cache = FileCache(
    cache_dir="./llm_cache",
    ttl=86400 * 7  # 7일
)

llm_persistent = LLM.create("gpt-4o-mini", cache=file_cache)

# 프로그램 재시작 후에도 캐시 유지
reply = llm_persistent.ask("복잡한 수학 문제 해결...")
```

### 커스텀 캐시 구현

```python
from pyhub.llm.cache import BaseCache
import hashlib
import redis
import json

class RedisCache(BaseCache):
    """Redis 기반 분산 캐시"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.client = redis.from_url(redis_url)
        self.ttl = ttl
    
    def _make_key(self, prompt: str, **kwargs) -> str:
        """캐시 키 생성"""
        cache_data = {
            "prompt": prompt,
            "model": kwargs.get("model", ""),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", None)
        }
        
        key_string = json.dumps(cache_data, sort_keys=True)
        return f"llm:cache:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def get(self, prompt: str, **kwargs) -> Optional[str]:
        key = self._make_key(prompt, **kwargs)
        value = self.client.get(key)
        return value.decode() if value else None
    
    def set(self, prompt: str, response: str, **kwargs):
        key = self._make_key(prompt, **kwargs)
        self.client.setex(key, self.ttl, response)
    
    def clear(self):
        pattern = "llm:cache:*"
        for key in self.client.scan_iter(match=pattern):
            self.client.delete(key)

# 사용
redis_cache = RedisCache(ttl=7200)  # 2시간
llm_distributed = LLM.create("gpt-4o-mini", cache=redis_cache)
```

### 캐시 전략

```python
class SmartCache:
    """상황별 캐싱 전략"""
    
    def __init__(self):
        self.short_cache = InMemoryCache(ttl=300)      # 5분 - 빠른 변경
        self.medium_cache = InMemoryCache(ttl=3600)    # 1시간 - 일반
        self.long_cache = FileCache(ttl=86400 * 30)    # 30일 - 정적
    
    def get_llm_for_query(self, query_type: str) -> LLM:
        if query_type == "realtime":
            # 실시간 정보 - 캐시 없음
            return LLM.create("gpt-4o-mini")
        elif query_type == "dynamic":
            # 자주 변경 - 짧은 캐시
            return LLM.create("gpt-4o-mini", cache=self.short_cache)
        elif query_type == "stable":
            # 안정적 정보 - 중간 캐시
            return LLM.create("gpt-4o-mini", cache=self.medium_cache)
        else:  # static
            # 정적 정보 - 긴 캐시
            return LLM.create("gpt-4o-mini", cache=self.long_cache)

# 사용
cache_manager = SmartCache()

# 실시간 정보
realtime_llm = cache_manager.get_llm_for_query("realtime")
weather = realtime_llm.ask("현재 서울 날씨는?")

# 정적 정보
static_llm = cache_manager.get_llm_for_query("static")
history = static_llm.ask("한국의 역사를 요약해주세요")
```

## 임베딩

### 텍스트 임베딩

```python
from typing import List
import numpy as np

# 임베딩 생성
llm = LLM.create("text-embedding-3-small")

# 단일 텍스트 임베딩
text = "파이썬은 간결하고 읽기 쉬운 프로그래밍 언어입니다."
embedding = llm.embed(text)
print(f"임베딩 차원: {len(embedding)}")
print(f"임베딩 벡터 (처음 5개): {embedding[:5]}")

# 여러 텍스트 임베딩
texts = [
    "파이썬은 데이터 과학에 널리 사용됩니다.",
    "자바스크립트는 웹 개발의 핵심 언어입니다.",
    "머신러닝은 인공지능의 한 분야입니다."
]

embeddings = llm.embed_many(texts)
print(f"임베딩 개수: {len(embeddings)}")
```

### 유사도 검색

```python
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.llm = LLM.create(model)
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[str]):
        """문서 추가 및 임베딩"""
        self.documents.extend(documents)
        new_embeddings = self.llm.embed_many(documents)
        self.embeddings.extend(new_embeddings)
    
    def search(self, query: str, top_k: int = 5) -> List[tuple[str, float]]:
        """의미 기반 검색"""
        # 쿼리 임베딩
        query_embedding = self.llm.embed(query)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]
        
        # 상위 k개 결과
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                self.documents[idx],
                similarities[idx]
            ))
        
        return results

# 사용
search_engine = SemanticSearch()

# 문서 추가
documents = [
    "파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다.",
    "머신러닝은 데이터로부터 패턴을 학습하는 인공지능 기술입니다.",
    "웹 개발에는 HTML, CSS, JavaScript가 필수적입니다.",
    "데이터베이스는 정보를 체계적으로 저장하고 관리하는 시스템입니다.",
    "API는 서로 다른 소프트웨어 간의 통신을 가능하게 합니다."
]

search_engine.add_documents(documents)

# 검색
results = search_engine.search("프로그래밍 언어 문법", top_k=3)

for doc, score in results:
    print(f"유사도: {score:.3f} - {doc}")
```

### 클러스터링

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class TextClusterer:
    def __init__(self, n_clusters: int = 3):
        self.llm = LLM.create("text-embedding-3-small")
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    def cluster_texts(self, texts: List[str]) -> Dict[int, List[str]]:
        """텍스트를 의미적으로 클러스터링"""
        # 임베딩 생성
        embeddings = self.llm.embed_many(texts)
        
        # 클러스터링
        labels = self.kmeans.fit_predict(embeddings)
        
        # 결과 정리
        clusters = {}
        for text, label in zip(texts, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text)
        
        return clusters
    
    def visualize_clusters(self, texts: List[str], embeddings: List[List[float]]):
        """클러스터 시각화 (2D)"""
        # PCA로 차원 축소
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        
        # 클러스터 레이블
        labels = self.kmeans.predict(embeddings)
        
        # 시각화
        plt.figure(figsize=(10, 8))
        for i in range(self.n_clusters):
            cluster_points = reduced[labels == i]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f'Cluster {i}',
                alpha=0.6
            )
        
        plt.title("Text Clustering Visualization")
        plt.legend()
        plt.show()

# 사용
clusterer = TextClusterer(n_clusters=3)

texts = [
    # 프로그래밍 관련
    "파이썬은 인기 있는 프로그래밍 언어입니다",
    "자바스크립트로 웹 애플리케이션을 만들 수 있습니다",
    "C++는 시스템 프로그래밍에 사용됩니다",
    
    # 음식 관련
    "김치는 한국의 전통 발효 식품입니다",
    "피자는 이탈리아에서 유래한 음식입니다",
    "초밥은 일본의 대표적인 요리입니다",
    
    # 운동 관련
    "축구는 세계에서 가장 인기 있는 스포츠입니다",
    "요가는 몸과 마음의 건강에 좋습니다",
    "수영은 전신 운동에 효과적입니다"
]

clusters = clusterer.cluster_texts(texts)

for cluster_id, cluster_texts in clusters.items():
    print(f"\n클러스터 {cluster_id}:")
    for text in cluster_texts:
        print(f"  - {text}")
```

## 메타프롬프팅

### 프롬프트 자동 생성

```python
class MetaPrompter:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = LLM.create(model)
    
    def generate_prompt(self, task: str, examples: List[Dict] = None) -> str:
        """작업에 최적화된 프롬프트 생성"""
        
        meta_prompt = f"""
        다음 작업을 수행하기 위한 최적의 프롬프트를 생성하세요:
        
        작업: {task}
        
        프롬프트는 다음을 포함해야 합니다:
        1. 명확한 지시사항
        2. 출력 형식 지정
        3. 품질 기준
        4. 예시 (제공된 경우)
        
        """
        
        if examples:
            meta_prompt += "\n제공된 예시:\n"
            for i, example in enumerate(examples):
                meta_prompt += f"{i+1}. 입력: {example['input']}\n   출력: {example['output']}\n"
        
        reply = self.llm.ask(meta_prompt)
        return reply.text
    
    def optimize_prompt(self, original_prompt: str, feedback: str) -> str:
        """피드백을 바탕으로 프롬프트 개선"""
        
        optimization_prompt = f"""
        다음 프롬프트를 피드백을 바탕으로 개선하세요:
        
        원본 프롬프트:
        {original_prompt}
        
        피드백:
        {feedback}
        
        개선된 프롬프트를 제시하세요.
        """
        
        reply = self.llm.ask(optimization_prompt)
        return reply.text

# 사용
meta_prompter = MetaPrompter()

# 프롬프트 생성
task = "고객 리뷰를 분석하여 감정과 주요 키워드를 추출"
examples = [
    {
        "input": "이 제품 정말 좋아요! 배송도 빠르고 품질도 최고입니다.",
        "output": "감정: 긍정, 키워드: 품질, 배송"
    }
]

optimized_prompt = meta_prompter.generate_prompt(task, examples)
print("생성된 프롬프트:", optimized_prompt)

# 프롬프트 개선
feedback = "감정을 더 세분화하고 신뢰도 점수도 포함해주세요"
improved_prompt = meta_prompter.optimize_prompt(optimized_prompt, feedback)
print("\n개선된 프롬프트:", improved_prompt)
```

## 체인 및 파이프라인

### 순차 체인

```python
class SequentialChain:
    def __init__(self):
        self.steps = []
    
    def add_step(self, name: str, llm: LLM, prompt_template: str):
        """체인에 단계 추가"""
        self.steps.append({
            "name": name,
            "llm": llm,
            "prompt_template": prompt_template
        })
        return self
    
    def run(self, initial_input: str) -> Dict[str, str]:
        """체인 실행"""
        results = {"initial_input": initial_input}
        current_output = initial_input
        
        for step in self.steps:
            prompt = step["prompt_template"].format(
                input=current_output,
                **results  # 이전 결과들도 사용 가능
            )
            
            reply = step["llm"].ask(prompt)
            current_output = reply.text
            results[step["name"]] = current_output
            
            print(f"\n[{step['name']}] 완료")
        
        return results

# 사용
chain = SequentialChain()

# 블로그 포스트 생성 체인
chain.add_step(
    "outline",
    LLM.create("gpt-4o-mini"),
    "다음 주제에 대한 블로그 포스트 개요를 작성하세요: {input}"
).add_step(
    "draft",
    LLM.create("gpt-4o-mini"),
    "다음 개요를 바탕으로 블로그 포스트 초안을 작성하세요:\n{outline}"
).add_step(
    "polish",
    LLM.create("gpt-4o"),
    "다음 초안을 다듬어 완성도 높은 블로그 포스트로 만들어주세요:\n{draft}"
)

results = chain.run("파이썬으로 웹 스크래핑하기")
print("\n최종 결과:", results["polish"][:200] + "...")
```

### 조건부 파이프라인

```python
class ConditionalPipeline:
    def __init__(self):
        self.analyzer = LLM.create("gpt-4o-mini")
        self.processors = {}
    
    def add_processor(self, condition: str, llm: LLM, prompt: str):
        """조건별 프로세서 추가"""
        self.processors[condition] = {
            "llm": llm,
            "prompt": prompt
        }
    
    async def process(self, text: str) -> str:
        """텍스트 분석 후 적절한 프로세서로 라우팅"""
        
        # 1. 텍스트 타입 분석
        analysis = await self.analyzer.ask_async(
            f"다음 텍스트의 타입을 분류하세요: {text}",
            choices=list(self.processors.keys())
        )
        
        text_type = analysis.choice
        
        # 2. 해당 프로세서 실행
        processor = self.processors[text_type]
        result = await processor["llm"].ask_async(
            processor["prompt"].format(text=text)
        )
        
        return result.text

# 사용
pipeline = ConditionalPipeline()

# 다양한 타입별 프로세서 설정
pipeline.add_processor(
    "기술문서",
    LLM.create("gpt-4o-mini"),
    "다음 기술 문서를 요약하고 핵심 개념을 추출하세요: {text}"
)

pipeline.add_processor(
    "고객문의",
    LLM.create("claude-3-5-haiku-latest"),
    "다음 고객 문의에 친절하게 답변하세요: {text}"
)

pipeline.add_processor(
    "코드",
    LLM.create("gpt-4-turbo"),
    "다음 코드를 분석하고 개선점을 제안하세요: {text}"
)

# 실행
text = "def calculate_sum(numbers): return sum(numbers)"
result = asyncio.run(pipeline.process(text))
print(result)
```

## 에이전트 패턴

### 도구 사용 에이전트

```python
from typing import Callable, Dict, Any

class ToolAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = LLM.create(model)
        self.tools = {}
    
    def register_tool(self, name: str, func: Callable, description: str):
        """도구 등록"""
        self.tools[name] = {
            "function": func,
            "description": description
        }
    
    def run(self, task: str) -> str:
        """작업 실행"""
        # 1. 사용할 도구 결정
        tool_prompt = f"""
        작업: {task}
        
        사용 가능한 도구:
        {self._format_tools()}
        
        어떤 도구를 사용해야 할까요? 도구 이름만 답하세요.
        """
        
        tool_choice = self.llm.ask(
            tool_prompt,
            choices=list(self.tools.keys()) + ["없음"]
        ).choice
        
        if tool_choice == "없음":
            # 도구 없이 직접 답변
            return self.llm.ask(task).text
        
        # 2. 도구 파라미터 추출
        param_prompt = f"""
        작업: {task}
        선택된 도구: {tool_choice}
        
        이 도구를 실행하기 위한 파라미터를 JSON 형식으로 제공하세요.
        """
        
        params_reply = self.llm.ask(param_prompt)
        # 실제로는 JSON 파싱 필요
        
        # 3. 도구 실행
        tool = self.tools[tool_choice]
        result = tool["function"](**params)  # 간단화된 예시
        
        # 4. 결과 정리
        final_prompt = f"""
        작업: {task}
        도구 실행 결과: {result}
        
        이 결과를 바탕으로 사용자에게 답변하세요.
        """
        
        return self.llm.ask(final_prompt).text
    
    def _format_tools(self) -> str:
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool['description']}")
        return "\n".join(lines)

# 사용
agent = ToolAgent()

# 도구 등록
def calculate(expression: str) -> float:
    """수식 계산"""
    return eval(expression)  # 실제로는 안전한 파서 사용

def web_search(query: str) -> str:
    """웹 검색 시뮬레이션"""
    return f"검색 결과: {query}에 대한 최신 정보..."

agent.register_tool("계산기", calculate, "수학 계산을 수행합니다")
agent.register_tool("웹검색", web_search, "인터넷에서 정보를 검색합니다")

# 작업 실행
result = agent.run("2024년 현재 파이썬 최신 버전은 무엇인가요?")
print(result)
```

## 성능 모니터링

### 메트릭 수집

```python
import time
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Metrics:
    prompt: str
    model: str
    response_time: float
    tokens_used: int
    cache_hit: bool
    error: Optional[str] = None

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
        self.model_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0,
            "total_tokens": 0,
            "errors": 0,
            "cache_hits": 0
        })
    
    def track(self, llm: LLM):
        """LLM 인스턴스에 모니터링 추가"""
        original_ask = llm.ask
        
        def wrapped_ask(prompt: str, **kwargs):
            start_time = time.time()
            error = None
            
            try:
                reply = original_ask(prompt, **kwargs)
                response_time = time.time() - start_time
                
                metric = Metrics(
                    prompt=prompt[:100],  # 처음 100자만
                    model=llm.model,
                    response_time=response_time,
                    tokens_used=reply.usage.total_tokens if reply.usage else 0,
                    cache_hit=getattr(reply, 'cache_hit', False)
                )
                
                self._update_stats(metric)
                return reply
                
            except Exception as e:
                error = str(e)
                metric = Metrics(
                    prompt=prompt[:100],
                    model=llm.model,
                    response_time=time.time() - start_time,
                    tokens_used=0,
                    cache_hit=False,
                    error=error
                )
                self._update_stats(metric)
                raise
        
        llm.ask = wrapped_ask
        return llm
    
    def _update_stats(self, metric: Metrics):
        self.metrics.append(metric)
        stats = self.model_stats[metric.model]
        
        stats["count"] += 1
        stats["total_time"] += metric.response_time
        stats["total_tokens"] += metric.tokens_used
        
        if metric.error:
            stats["errors"] += 1
        if metric.cache_hit:
            stats["cache_hits"] += 1
    
    def get_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        report = {}
        
        for model, stats in self.model_stats.items():
            count = stats["count"]
            if count > 0:
                report[model] = {
                    "requests": count,
                    "avg_response_time": stats["total_time"] / count,
                    "total_tokens": stats["total_tokens"],
                    "avg_tokens_per_request": stats["total_tokens"] / count,
                    "error_rate": stats["errors"] / count,
                    "cache_hit_rate": stats["cache_hits"] / count
                }
        
        return report

# 사용
monitor = PerformanceMonitor()

# 모니터링할 LLM 생성
llm1 = monitor.track(LLM.create("gpt-4o-mini"))
llm2 = monitor.track(LLM.create("claude-3-5-haiku-latest"))

# 작업 수행
for i in range(10):
    llm1.ask(f"질문 {i}")
    llm2.ask(f"질문 {i}")

# 리포트 확인
report = monitor.get_report()
for model, stats in report.items():
    print(f"\n{model}:")
    print(f"  평균 응답 시간: {stats['avg_response_time']:.2f}초")
    print(f"  평균 토큰 사용: {stats['avg_tokens_per_request']:.0f}")
    print(f"  캐시 히트율: {stats['cache_hit_rate']:.1%}")
```

## 다음 단계

- [API 레퍼런스](../api-reference/index.md) - 전체 API 문서
- [예제](../examples/index.md) - 실제 사용 사례
- [문제 해결](../troubleshooting.md) - 일반적인 문제와 해결법