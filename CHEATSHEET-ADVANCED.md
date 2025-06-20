# pyhub-llm 고급 가이드

이 문서는 pyhub-llm의 고급 기능들을 다룹니다. 임베딩, MCP 통합, 웹 프레임워크 통합, 체이닝, 에러 처리 등 복잡한 사용 사례와 패턴들을 포함합니다.

> 💡 예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.

## 목차

- [임베딩](#임베딩)
- [MCP 통합](#mcp-통합)
- [웹 프레임워크 통합](#웹-프레임워크-통합)
- [체이닝](#체이닝)
- [에러 처리](#에러-처리)
- [성능 최적화](#성능-최적화)
- [아키텍처 패턴](#아키텍처-패턴)
- [실용적인 예제](#실용적인-예제)
- [추가 자료](#추가-자료)

## 임베딩

💻 [실행 가능한 예제](examples/advanced/01_embeddings.py)

### 텍스트 임베딩 생성

```python
from pyhub.llm import OpenAILLM
import numpy as np

# 임베딩 모델 사용
llm = OpenAILLM(embedding_model="text-embedding-3-small")

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
from pyhub.llm import OpenAILLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(text1: str, text2: str) -> float:
    """두 텍스트의 유사도 계산"""
    llm = OpenAILLM(embedding_model="text-embedding-3-small")
    
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

## MCP 통합

💻 [실행 가능한 예제](examples/mcp_integration_example.py)

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
    url="wss://localhost:8080/ws"  # websocket transport로 자동 감지
)

sse_config = McpConfig(
    url="http://localhost:8080/sse"  # sse transport로 자동 감지
)

# 2. 문자열로 간편 설정 - 팩토리 함수 사용
config1 = create_mcp_config("python server.py")  # stdio
config2 = create_mcp_config("http://localhost:8080")  # http
config3 = create_mcp_config("wss://localhost:8080")  # websocket

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

💻 [실행 가능한 예제](examples/advanced/03_web_frameworks/)

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

# 이미지 생성 뷰 (views.py)
from django.http import HttpResponse
from pyhub.llm import OpenAILLM
from io import BytesIO
import uuid

class ImageGenerationView(View):
    """AI 이미지 생성 뷰"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request):
        """텍스트로부터 이미지 생성"""
        data = json.loads(request.body)
        prompt = data.get('prompt', '')
        size = data.get('size', '1024x1024')
        quality = data.get('quality', 'standard')
        
        if not prompt:
            return JsonResponse({'error': '프롬프트가 필요합니다.'}, status=400)
        
        try:
            # DALL-E 3 모델로 이미지 생성
            dalle = OpenAILLM(model="dall-e-3")
            reply = dalle.generate_image(
                prompt,
                size=size,
                quality=quality,
                style="natural"
            )
            
            # 방법 1: URL 반환
            return JsonResponse({
                'url': reply.url,
                'revised_prompt': reply.revised_prompt,
                'size': reply.size
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

class ImageDownloadView(View):
    """생성된 이미지 다운로드"""
    
    def post(self, request):
        """이미지를 생성하고 직접 반환"""
        data = json.loads(request.body)
        prompt = data.get('prompt', '')
        
        try:
            # 이미지 생성
            dalle = OpenAILLM(model="dall-e-3")
            reply = dalle.generate_image(prompt)
            
            # HttpResponse에 직접 저장
            response = HttpResponse(content_type='image/png')
            response['Content-Disposition'] = f'attachment; filename="generated_{uuid.uuid4().hex[:8]}.png"'
            
            # ImageReply.save()가 HttpResponse를 직접 지원
            reply.save(response)
            
            return response
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

# 이미지 생성 후 DB 저장 (models.py 추가)
class GeneratedImage(models.Model):
    """생성된 이미지 모델"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    prompt = models.TextField()
    revised_prompt = models.TextField(blank=True)
    image = models.ImageField(upload_to='generated/%Y/%m/%d/')
    size = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class ImageGenerationWithSaveView(View):
    """이미지 생성 후 DB 저장"""
    
    def post(self, request):
        data = json.loads(request.body)
        prompt = data.get('prompt', '')
        
        try:
            # 이미지 생성
            dalle = OpenAILLM(model="dall-e-3")
            reply = dalle.generate_image(prompt, quality="hd")
            
            # 방법 1: to_django_file() 사용 (v0.9.0+) - 권장
            generated = GeneratedImage.objects.create(
                user=request.user if request.user.is_authenticated else None,
                prompt=prompt,
                revised_prompt=reply.revised_prompt or prompt,
                size=reply.size,
                image=reply.to_django_file(f'dalle_{uuid.uuid4().hex[:8]}.png')
            )
            
            # 방법 2: BytesIO 사용 (이전 버전 호환)
            # buffer = BytesIO()
            # reply.save(buffer)
            # buffer.seek(0)
            # from django.core.files.base import ContentFile
            # generated.image.save(
            #     f'dalle_{uuid.uuid4().hex[:8]}.png',
            #     ContentFile(buffer.getvalue()),
            #     save=True
            # )
            
            return JsonResponse({
                'id': generated.id,
                'url': generated.image.url,
                'prompt': generated.prompt,
                'revised_prompt': generated.revised_prompt,
                'created_at': generated.created_at.isoformat()
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

# 비동기 이미지 생성 (Django 4.1+)
from django.views.generic import View
import asyncio

class AsyncImageGenerationView(View):
    """비동기 이미지 생성"""
    
    async def post(self, request):
        data = json.loads(request.body)
        prompts = data.get('prompts', [])
        
        if not prompts:
            return JsonResponse({'error': '프롬프트 목록이 필요합니다.'}, status=400)
        
        try:
            dalle = OpenAILLM(model="dall-e-3")
            
            # 여러 이미지 동시 생성
            tasks = [dalle.generate_image_async(prompt) for prompt in prompts]
            images = await asyncio.gather(*tasks)
            
            # 결과 수집
            results = []
            for i, reply in enumerate(images):
                # BytesIO에 저장
                buffer = BytesIO()
                await reply.save_async(buffer)
                buffer.seek(0)
                
                # Base64 인코딩
                import base64
                image_data = base64.b64encode(buffer.getvalue()).decode()
                
                results.append({
                    'prompt': prompts[i],
                    'data': f'data:image/png;base64,{image_data}',
                    'size': reply.size
                })
            
            return JsonResponse({'images': results})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

# 이미지 변형 및 분석 뷰
class ImageAnalysisAndGenerationView(View):
    """업로드된 이미지를 분석하고 유사한 이미지 생성"""
    
    def post(self, request):
        uploaded_file = request.FILES.get('image')
        
        if not uploaded_file:
            return JsonResponse({'error': '이미지가 필요합니다.'}, status=400)
        
        try:
            # 1. 업로드된 이미지 분석
            analyzer = LLM.create("gpt-4o-mini")
            
            # 임시 파일로 저장
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            # 이미지 분석하여 생성 프롬프트 만들기
            from pydantic import BaseModel, Field
            
            class ImagePrompt(BaseModel):
                detailed_prompt: str = Field(description="DALL-E 3를 위한 상세한 프롬프트")
                style: str = Field(description="이미지 스타일")
                main_elements: list[str] = Field(description="주요 요소들")
            
            analysis = analyzer.ask(
                "이 이미지를 분석하고 유사한 이미지를 생성하기 위한 DALL-E 프롬프트를 만들어주세요",
                files=[tmp_path],
                schema=ImagePrompt
            )
            
            prompt_data = analysis.structured_data
            
            # 2. 분석 결과로 새 이미지 생성
            dalle = OpenAILLM(model="dall-e-3")
            new_image = dalle.generate_image(
                prompt_data.detailed_prompt,
                quality="hd"
            )
            
            # 3. 결과 반환
            return JsonResponse({
                'analysis': {
                    'prompt': prompt_data.detailed_prompt,
                    'style': prompt_data.style,
                    'elements': prompt_data.main_elements
                },
                'generated_image': {
                    'url': new_image.url,
                    'size': new_image.size
                }
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        finally:
            # 임시 파일 정리
            import os
            if 'tmp_path' in locals():
                os.unlink(tmp_path)

# urls.py 설정 예시
from django.urls import path
from .views import (
    ImageGenerationView, 
    ImageDownloadView, 
    ImageGenerationWithSaveView,
    AsyncImageGenerationView,
    ImageAnalysisAndGenerationView
)

urlpatterns = [
    path('api/generate-image/', ImageGenerationView.as_view(), name='generate-image'),
    path('api/download-image/', ImageDownloadView.as_view(), name='download-image'),
    path('api/save-image/', ImageGenerationWithSaveView.as_view(), name='save-image'),
    path('api/batch-generate/', AsyncImageGenerationView.as_view(), name='batch-generate'),
    path('api/analyze-and-generate/', ImageAnalysisAndGenerationView.as_view(), name='analyze-generate'),
]

# Django 템플릿에서 사용 예시
"""
<!-- image_generator.html -->
<form id="imageForm">
    <textarea name="prompt" placeholder="이미지 설명을 입력하세요"></textarea>
    <button type="submit">이미지 생성</button>
</form>

<div id="result"></div>

<script>
document.getElementById('imageForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = e.target.prompt.value;
    
    const response = await fetch('/api/generate-image/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt})
    });
    
    const data = await response.json();
    if (data.url) {
        document.getElementById('result').innerHTML = 
            `<img src="${data.url}" alt="Generated image">`;
    }
});
</script>
"""
```

### Django ImageField 활용 가이드 (v0.9.1+)

v0.9.1부터 `ImageReply.to_django_file()` 메서드가 추가되어 Django ImageField와의 통합이 더욱 간편해졌습니다.

#### 기본 사용법

```python
from django.db import models
from django.views import View
from django.http import JsonResponse
from pyhub.llm import OpenAILLM
import json

# 모델 정의
class AIGeneratedImage(models.Model):
    """AI로 생성된 이미지 관리"""
    prompt = models.TextField()
    image = models.ImageField(upload_to='ai_generated/%Y/%m/%d/')
    created_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)

# 뷰 정의
class GenerateImageView(View):
    def post(self, request):
        prompt = json.loads(request.body).get('prompt')
        
        # 이미지 생성
        dalle = OpenAILLM(model="dall-e-3")
        reply = dalle.generate_image(prompt, quality="hd", size="1024x1792")
        
        # to_django_file()로 간단하게 저장
        image_instance = AIGeneratedImage.objects.create(
            prompt=prompt,
            image=reply.to_django_file(),  # 자동으로 고유 파일명 생성
            metadata={
                'revised_prompt': reply.revised_prompt,
                'size': reply.size,
                'model': 'dall-e-3'
            }
        )
        
        return JsonResponse({
            'id': image_instance.id,
            'url': image_instance.image.url
        })
```

#### 고급 활용 예제

```python
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
from PIL import Image as PILImage
import hashlib

class AdvancedImageModel(models.Model):
    """고급 이미지 관리 모델"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prompt = models.TextField()
    original = models.ImageField(
        upload_to='originals/%Y/%m/',
        validators=[FileExtensionValidator(['png', 'jpg', 'jpeg'])]
    )
    thumbnail = models.ImageField(upload_to='thumbnails/%Y/%m/', blank=True)
    hash = models.CharField(max_length=64, unique=True, editable=False)
    
    def save(self, *args, **kwargs):
        # 이미지 해시 생성 (중복 방지)
        if self.original and not self.hash:
            self.original.seek(0)
            self.hash = hashlib.sha256(self.original.read()).hexdigest()
            self.original.seek(0)
        super().save(*args, **kwargs)

class SmartImageGenerationView(View):
    """스마트 이미지 생성 및 처리"""
    
    def post(self, request):
        data = json.loads(request.body)
        prompt = data.get('prompt')
        generate_thumbnail = data.get('thumbnail', True)
        
        # 1. 이미지 생성
        dalle = OpenAILLM(model="dall-e-3")
        reply = dalle.generate_image(prompt, quality="hd")
        
        # 2. 모델 인스턴스 생성
        instance = AdvancedImageModel(
            user=request.user,
            prompt=prompt,
            original=reply.to_django_file(f'original_{uuid.uuid4().hex[:8]}.png')
        )
        
        # 3. 썸네일 생성 (선택사항)
        if generate_thumbnail:
            # to_pil()로 PIL 이미지로 변환
            pil_image = reply.to_pil()
            pil_image.thumbnail((256, 256), PILImage.Resampling.LANCZOS)
            
            # 썸네일을 BytesIO로 저장
            from io import BytesIO
            thumb_io = BytesIO()
            pil_image.save(thumb_io, format='PNG', optimize=True)
            thumb_io.seek(0)
            
            # Django 파일로 변환
            from django.core.files.base import ContentFile
            instance.thumbnail.save(
                f'thumb_{uuid.uuid4().hex[:8]}.png',
                ContentFile(thumb_io.getvalue()),
                save=False
            )
        
        instance.save()
        
        return JsonResponse({
            'id': instance.id,
            'original_url': instance.original.url,
            'thumbnail_url': instance.thumbnail.url if instance.thumbnail else None,
            'hash': instance.hash
        })
```

#### 배치 처리 및 비동기 예제

```python
from django.db import transaction
from asgiref.sync import sync_to_async
import asyncio

class BatchImageGeneration(models.Model):
    """배치 이미지 생성 작업"""
    name = models.CharField(max_length=200)
    prompts = models.JSONField()  # 프롬프트 리스트
    status = models.CharField(max_length=20, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

class BatchImage(models.Model):
    """배치로 생성된 개별 이미지"""
    batch = models.ForeignKey(BatchImageGeneration, on_delete=models.CASCADE, related_name='images')
    prompt = models.TextField()
    image = models.ImageField(upload_to='batch/%Y/%m/%d/')
    order = models.IntegerField()

class BatchGenerationView(View):
    """비동기 배치 이미지 생성"""
    
    async def post(self, request):
        data = json.loads(request.body)
        prompts = data.get('prompts', [])
        batch_name = data.get('name', 'Untitled Batch')
        
        # 배치 작업 생성
        batch = await sync_to_async(BatchImageGeneration.objects.create)(
            name=batch_name,
            prompts=prompts
        )
        
        # 비동기로 이미지 생성
        dalle = OpenAILLM(model="dall-e-3")
        
        async def generate_and_save(prompt, order):
            try:
                # 이미지 생성
                reply = await dalle.generate_image_async(prompt)
                
                # Django ORM은 동기식이므로 sync_to_async 사용
                await sync_to_async(BatchImage.objects.create)(
                    batch=batch,
                    prompt=prompt,
                    image=reply.to_django_file(f'batch_{batch.id}_{order}.png'),
                    order=order
                )
                return True
            except Exception as e:
                print(f"Error generating image {order}: {e}")
                return False
        
        # 모든 이미지 동시 생성 (최대 5개씩)
        tasks = []
        for i, prompt in enumerate(prompts):
            if len(tasks) >= 5:
                await asyncio.gather(*tasks)
                tasks = []
            tasks.append(generate_and_save(prompt, i))
        
        if tasks:
            await asyncio.gather(*tasks)
        
        # 배치 상태 업데이트
        batch.status = 'completed'
        batch.completed_at = timezone.now()
        await sync_to_async(batch.save)()
        
        # 결과 반환
        images = await sync_to_async(list)(
            batch.images.values('id', 'prompt', 'image', 'order')
        )
        
        return JsonResponse({
            'batch_id': batch.id,
            'total': len(prompts),
            'completed': len(images),
            'images': images
        })
```

#### 이미지 변형 파이프라인

```python
class ImageVariation(models.Model):
    """원본과 변형 이미지 관리"""
    original_prompt = models.TextField()
    variation_prompt = models.TextField()
    original = models.ImageField(upload_to='variations/original/')
    variation = models.ImageField(upload_to='variations/generated/')
    style = models.CharField(max_length=50)
    
class VariationPipelineView(View):
    """이미지 분석 후 변형 생성"""
    
    def post(self, request):
        uploaded_file = request.FILES.get('image')
        style = request.POST.get('style', 'artistic')
        
        # 1. 업로드된 이미지 분석
        analyzer = LLM.create("gpt-4o-mini")
        
        # 임시 파일로 저장
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        # 이미지 분석하여 프롬프트 생성
        analysis_reply = analyzer.ask(
            f"이 이미지를 {style} 스타일로 재해석하기 위한 DALL-E 프롬프트를 만들어주세요",
            files=[tmp_path]
        )
        
        variation_prompt = analysis_reply.text
        
        # 2. 변형 이미지 생성
        dalle = OpenAILLM(model="dall-e-3")
        image_reply = dalle.generate_image(variation_prompt, quality="hd")
        
        # 3. 모델에 저장
        with transaction.atomic():
            variation = ImageVariation.objects.create(
                original_prompt="Uploaded image",
                variation_prompt=variation_prompt,
                style=style
            )
            
            # 원본 이미지 저장
            variation.original.save(
                f'original_{variation.id}.png',
                uploaded_file,
                save=False
            )
            
            # 생성된 이미지 저장
            variation.variation = image_reply.to_django_file(
                f'variation_{variation.id}.png'
            )
            
            variation.save()
        
        # 임시 파일 삭제
        import os
        os.unlink(tmp_path)
        
        return JsonResponse({
            'id': variation.id,
            'original_url': variation.original.url,
            'variation_url': variation.variation.url,
            'prompt': variation.variation_prompt
        })
```

#### 모델 시그널과 후처리

```python
from django.db.models.signals import post_save
from django.dispatch import receiver
import requests

@receiver(post_save, sender=AIGeneratedImage)
def process_generated_image(sender, instance, created, **kwargs):
    """생성된 이미지 후처리"""
    if created and instance.image:
        # 예: 이미지 메타데이터 추가
        try:
            pil_img = PILImage.open(instance.image.path)
            instance.metadata.update({
                'width': pil_img.width,
                'height': pil_img.height,
                'format': pil_img.format,
                'mode': pil_img.mode
            })
            instance.save(update_fields=['metadata'])
        except Exception as e:
            print(f"Error processing image metadata: {e}")

# 커스텀 스토리지 백엔드
from django.core.files.storage import Storage
from storages.backends.s3boto3 import S3Boto3Storage

class OptimizedS3Storage(S3Boto3Storage):
    """최적화된 S3 스토리지"""
    def __init__(self, *args, **kwargs):
        kwargs['object_parameters'] = {
            'CacheControl': 'max-age=86400',
            'ContentDisposition': 'inline'
        }
        super().__init__(*args, **kwargs)

# settings.py에서 사용
# DEFAULT_FILE_STORAGE = 'myapp.storage.OptimizedS3Storage'
```

#### 관리자 인터페이스 커스터마이징

```python
from django.contrib import admin
from django.utils.html import format_html

@admin.register(AIGeneratedImage)
class AIGeneratedImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'prompt_preview', 'image_preview', 'created_at']
    list_filter = ['created_at']
    search_fields = ['prompt']
    readonly_fields = ['image_preview_large', 'metadata_display']
    
    def prompt_preview(self, obj):
        return obj.prompt[:50] + '...' if len(obj.prompt) > 50 else obj.prompt
    prompt_preview.short_description = 'Prompt'
    
    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" width="100" height="100" style="object-fit: cover;"/>',
                obj.image.url
            )
        return '-'
    image_preview.short_description = 'Preview'
    
    def image_preview_large(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" width="400"/>',
                obj.image.url
            )
        return '-'
    image_preview_large.short_description = 'Image'
    
    def metadata_display(self, obj):
        import json
        return format_html(
            '<pre>{}</pre>',
            json.dumps(obj.metadata, indent=2)
        )
    metadata_display.short_description = 'Metadata'
```

> **팁**: 
> - `to_django_file()`은 파일명을 지정하지 않으면 자동으로 타임스탬프 기반 고유 이름을 생성합니다
> - ImageField의 `upload_to` 옵션과 함께 사용하면 체계적인 파일 관리가 가능합니다
> - 대용량 이미지 처리 시 비동기 뷰를 활용하면 성능을 크게 개선할 수 있습니다
> - S3 같은 외부 스토리지 사용 시 `django-storages` 패키지와 완벽하게 호환됩니다

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

## 체이닝

💻 [실행 가능한 예제](examples/advanced/02_chaining.py)

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

💻 [실행 가능한 예제](examples/advanced/04_advanced_error_handling.py)

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

## 성능 최적화

### Stateless 모드를 통한 메모리/API 비용 절감

반복적인 독립 작업에서 Stateless 모드를 사용하면 메모리 사용량과 API 비용을 크게 절감할 수 있습니다.

```python
from pyhub.llm import LLM
import time
import psutil
import os

def compare_memory_usage():
    """일반 모드 vs Stateless 모드 메모리 사용량 비교"""
    process = psutil.Process(os.getpid())
    
    # 일반 모드 (히스토리 누적)
    normal_llm = LLM.create("gpt-4o-mini")
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    for i in range(100):
        normal_llm.ask(f"텍스트 {i} 분류", choices=["A", "B", "C"])
    
    normal_memory = process.memory_info().rss / 1024 / 1024
    print(f"일반 모드 메모리 증가: {normal_memory - start_memory:.2f} MB")
    print(f"히스토리 크기: {len(normal_llm.history)} 메시지")
    
    # Stateless 모드 (히스토리 없음)
    stateless_llm = LLM.create("gpt-4o-mini", stateless=True)
    start_memory = process.memory_info().rss / 1024 / 1024
    
    for i in range(100):
        stateless_llm.ask(f"텍스트 {i} 분류", choices=["A", "B", "C"])
    
    stateless_memory = process.memory_info().rss / 1024 / 1024
    print(f"Stateless 모드 메모리 증가: {stateless_memory - start_memory:.2f} MB")
    print(f"히스토리 크기: {len(stateless_llm.history)} 메시지")
```

### API 토큰 사용량 비교

```python
def estimate_token_usage(num_requests: int):
    """토큰 사용량 추정"""
    # 각 요청이 약 50 토큰이라고 가정
    tokens_per_request = 50
    
    # 일반 모드: 히스토리가 누적되어 토큰 사용량 증가
    normal_tokens = 0
    for i in range(num_requests):
        # 이전 대화 내역 + 새 요청
        history_tokens = i * tokens_per_request * 2  # 질문 + 답변
        normal_tokens += history_tokens + tokens_per_request
    
    # Stateless 모드: 항상 일정한 토큰 사용
    stateless_tokens = num_requests * tokens_per_request
    
    print(f"요청 {num_requests}개 처리 시:")
    print(f"일반 모드: {normal_tokens:,} 토큰")
    print(f"Stateless 모드: {stateless_tokens:,} 토큰")
    print(f"절감률: {(1 - stateless_tokens/normal_tokens) * 100:.1f}%")
    
    # 비용 계산 (GPT-4o-mini 기준)
    cost_per_1k_tokens = 0.15 / 1000  # $0.15 per 1M tokens
    normal_cost = normal_tokens * cost_per_1k_tokens
    stateless_cost = stateless_tokens * cost_per_1k_tokens
    
    print(f"\n예상 비용:")
    print(f"일반 모드: ${normal_cost:.4f}")
    print(f"Stateless 모드: ${stateless_cost:.4f}")
    print(f"절감액: ${normal_cost - stateless_cost:.4f}")

# 사용 예
estimate_token_usage(100)
```

### 병렬 처리 최적화

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import asyncio

class ParallelProcessor:
    """Stateless LLM을 활용한 병렬 처리기"""
    
    def __init__(self, model: str = "gpt-4o-mini", max_workers: int = 5):
        self.model = model
        self.max_workers = max_workers
        # 각 워커용 Stateless LLM 인스턴스
        self.llms = [
            LLM.create(model, stateless=True) 
            for _ in range(max_workers)
        ]
    
    def process_batch(self, 
                     items: List[str], 
                     task_template: str,
                     **kwargs) -> List[Tuple[str, Reply]]:
        """배치 처리"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 아이템을 워커에 할당
            futures = {}
            for i, item in enumerate(items):
                llm = self.llms[i % self.max_workers]
                future = executor.submit(
                    llm.ask, 
                    task_template.format(item=item),
                    **kwargs
                )
                futures[future] = item
            
            # 결과 수집
            for future in as_completed(futures):
                item = futures[future]
                try:
                    reply = future.result()
                    results.append((item, reply))
                except Exception as e:
                    print(f"Error processing {item}: {e}")
        
        return results

# 사용 예
processor = ParallelProcessor(max_workers=10)

# 대량 분류 작업
items = ["텍스트1", "텍스트2", "텍스트3"] * 100
results = processor.process_batch(
    items,
    "다음 텍스트의 감정을 분석하세요: {item}",
    choices=["긍정", "부정", "중립"]
)

print(f"처리 완료: {len(results)}건")
```

## 아키텍처 패턴

### Stateless LLM을 활용한 마이크로서비스

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI()

# 각 서비스용 Stateless LLM 인스턴스
classifiers = {
    "sentiment": LLM.create("gpt-4o-mini", stateless=True),
    "intent": LLM.create("gpt-4o-mini", stateless=True),
    "category": LLM.create("gpt-4o-mini", stateless=True),
}

class AnalysisRequest(BaseModel):
    text: str
    services: List[str] = ["sentiment", "intent", "category"]

class AnalysisResponse(BaseModel):
    text: str
    sentiment: Optional[str] = None
    intent: Optional[str] = None
    category: Optional[str] = None

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """텍스트 멀티 분석 API"""
    results = {"text": request.text}
    
    # 각 서비스를 병렬로 실행
    tasks = []
    
    if "sentiment" in request.services:
        tasks.append(("sentiment", classifiers["sentiment"].ask_async(
            f"감정 분석: {request.text}",
            choices=["긍정", "부정", "중립"]
        )))
    
    if "intent" in request.services:
        tasks.append(("intent", classifiers["intent"].ask_async(
            f"의도 분류: {request.text}",
            choices=["질문", "요청", "불만", "정보"]
        )))
    
    if "category" in request.services:
        tasks.append(("category", classifiers["category"].ask_async(
            f"카테고리 분류: {request.text}",
            choices=["기술", "일반", "긴급", "기타"]
        )))
    
    # 모든 분석 완료 대기
    for service_name, task in tasks:
        try:
            reply = await task
            results[service_name] = reply.choice
        except Exception as e:
            print(f"Error in {service_name}: {e}")
    
    return AnalysisResponse(**results)

# 서버 실행: uvicorn main:app --reload
```

### 이벤트 기반 처리 시스템

```python
from typing import Dict, Any, Callable
import asyncio
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    id: str
    type: str
    data: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class EventProcessor:
    """Stateless LLM 기반 이벤트 처리기"""
    
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.llms: Dict[str, LLM] = {}
    
    def register_handler(self, event_type: str, model: str = "gpt-4o-mini"):
        """이벤트 타입별 핸들러 등록"""
        def decorator(func):
            self.handlers[event_type] = func
            # 각 핸들러용 Stateless LLM 생성
            self.llms[event_type] = LLM.create(model, stateless=True)
            return func
        return decorator
    
    async def process_event(self, event: Event):
        """이벤트 처리"""
        handler = self.handlers.get(event.type)
        if not handler:
            print(f"No handler for event type: {event.type}")
            return
        
        llm = self.llms[event.type]
        return await handler(event, llm)

# 사용 예
processor = EventProcessor()

@processor.register_handler("customer_feedback")
async def handle_feedback(event: Event, llm: LLM):
    """고객 피드백 처리"""
    feedback = event.data.get("message", "")
    
    # 감정 분석
    sentiment = await llm.ask_async(
        f"고객 피드백 감정 분석: {feedback}",
        choices=["매우 긍정", "긍정", "중립", "부정", "매우 부정"]
    )
    
    # 우선순위 결정
    priority = await llm.ask_async(
        f"피드백 우선순위: {feedback}",
        choices=["긴급", "높음", "보통", "낮음"]
    )
    
    return {
        "event_id": event.id,
        "sentiment": sentiment.choice,
        "priority": priority.choice,
        "processed_at": datetime.now()
    }

@processor.register_handler("support_ticket")
async def handle_ticket(event: Event, llm: LLM):
    """지원 티켓 처리"""
    ticket = event.data.get("content", "")
    
    # 카테고리 분류
    category = await llm.ask_async(
        f"지원 요청 분류: {ticket}",
        choices=["기술지원", "결제문의", "계정문제", "기타"]
    )
    
    return {
        "event_id": event.id,
        "category": category.choice,
        "auto_response": f"{category.choice} 팀으로 전달되었습니다."
    }

# 이벤트 처리 실행
async def main():
    events = [
        Event("1", "customer_feedback", {"message": "정말 훌륭한 서비스입니다!"}),
        Event("2", "support_ticket", {"content": "비밀번호를 잊어버렸어요"}),
        Event("3", "customer_feedback", {"message": "배송이 너무 늦어요"}),
    ]
    
    # 모든 이벤트 병렬 처리
    tasks = [processor.process_event(event) for event in events]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(result)

# asyncio.run(main())
```

## 실용적인 예제

💻 [실행 가능한 예제](examples/advanced/05_practical_examples/)

### 챗봇 구현

💻 [실행 가능한 예제](examples/advanced/05_practical_examples/chatbot.py)

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

💻 [실행 가능한 예제](examples/advanced/05_practical_examples/document_summarizer.py)

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

💻 [실행 가능한 예제](examples/advanced/05_practical_examples/code_reviewer.py)

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
        
        # 백틱을 변수로 분리하여 마크다운 파싱 문제 방지
        code_fence = "```"
        prompt = f"""
다음 {language} 코드를 리뷰해주세요:

{code_fence}{language}
{code}
{code_fence}

코드의 품질, 보안, 성능, 가독성 등을 종합적으로 평가해주세요.
"""
        
        reply = self.llm.ask(prompt, schema=CodeReview)
        return reply.structured_data.dict()
    
    def suggest_refactoring(self, code: str) -> str:
        """리팩토링 제안"""
        # 백틱을 변수로 분리하여 마크다운 파싱 문제 방지
        code_fence = "```"
        prompt = f"""
다음 코드를 더 깔끔하고 효율적으로 리팩토링해주세요:

{code_fence}python
{code}
{code_fence}

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

💻 [실행 가능한 예제](examples/advanced/05_practical_examples/translator.py)

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

💻 [실행 가능한 예제](examples/advanced/05_practical_examples/qa_system.py)

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

## 추가 자료

### 공식 문서
- [pyhub-llm GitHub 저장소](https://github.com/pyhub-kr/pyhub-llm)
- [API 레퍼런스](https://pyhub-llm.readthedocs.io/)
- [기본 사용법 CHEATSHEET](./CHEATSHEET-BASIC.md)

### 예제 코드
- [GitHub Examples 폴더](https://github.com/pyhub-kr/pyhub-llm/tree/main/examples)
- [Jupyter Notebooks](https://github.com/pyhub-kr/pyhub-llm/tree/main/notebooks)

### 커뮤니티
- [GitHub Issues](https://github.com/pyhub-kr/pyhub-llm/issues)
- [GitHub Discussions](https://github.com/pyhub-kr/pyhub-llm/discussions)

### 추가 리소스
- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Django 공식 문서](https://docs.djangoproject.com/)
- [Streamlit 공식 문서](https://docs.streamlit.io/)

### 기여하기
버그 리포트, 기능 요청, 코드 기여를 환영합니다!

---

이 고급 가이드는 pyhub-llm의 전문적인 기능들을 다룹니다. 각 예제는 실제 프로덕션 환경에서 사용할 수 있도록 설계되었으며, 필요에 따라 수정하여 사용하시기 바랍니다.
