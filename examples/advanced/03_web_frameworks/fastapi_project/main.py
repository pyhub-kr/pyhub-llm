#!/usr/bin/env python3
"""
예제: FastAPI와 pyhub-llm 통합
난이도: 고급
설명: FastAPI를 사용한 AI 챗봇 API 서버
요구사항: 
  - pyhub-llm (pip install pyhub-llm)
  - fastapi (pip install fastapi)
  - uvicorn (pip install uvicorn)
  - python-multipart (pip install python-multipart)
  - OPENAI_API_KEY 환경 변수

실행 방법:
  uvicorn main:app --reload
"""

import os
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyhub.llm import LLM
from pyhub.llm.types import Message


# Pydantic 모델들
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    stream: bool = False
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0, le=2)


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tokens_used: int
    timestamp: datetime


class AnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "sentiment"  # sentiment, summary, keywords, all


class SentimentAnalysis(BaseModel):
    sentiment: str
    confidence: float
    emotions: List[str]


class TextAnalysis(BaseModel):
    sentiment: Optional[SentimentAnalysis]
    summary: Optional[str]
    keywords: Optional[List[str]]
    entities: Optional[List[str]]


class ImageAnalysisRequest(BaseModel):
    question: str = "이 이미지를 설명해주세요."


# 전역 변수
app = FastAPI(title="pyhub-llm API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 대화 내역 저장소 (실제로는 DB 사용)
conversations: Dict[str, List[Message]] = {}

# LLM 인스턴스 캐시
llm_cache: Dict[str, LLM] = {}


def get_llm(model: str = "gpt-4o-mini", **kwargs) -> LLM:
    """LLM 인스턴스 가져오기 (캐싱)"""
    cache_key = f"{model}_{hash(frozenset(kwargs.items()))}"
    if cache_key not in llm_cache:
        llm_cache[cache_key] = LLM.create(model, **kwargs)
    return llm_cache[cache_key]


@app.get("/")
async def root():
    """API 정보"""
    return {
        "message": "pyhub-llm FastAPI Server",
        "endpoints": {
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "analyze": "/analyze",
            "analyze_image": "/analyze/image",
            "embeddings": "/embeddings",
            "conversations": "/conversations/{conversation_id}"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """일반 채팅 엔드포인트"""
    try:
        # 대화 ID 생성 또는 가져오기
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 대화 내역 가져오기
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        messages = conversations[conversation_id]
        messages.append(Message(role="user", content=request.message))
        
        # LLM 호출
        llm = get_llm(request.model, temperature=request.temperature)
        
        # 대화 컨텍스트 구성
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in messages[-10:]])
        reply = await llm.ask_async(context)
        
        # 응답 저장
        messages.append(Message(role="assistant", content=reply.text))
        
        return ChatResponse(
            response=reply.text,
            conversation_id=conversation_id,
            tokens_used=reply.usage.total if reply.usage else 0,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """스트리밍 채팅 엔드포인트"""
    async def generate():
        try:
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            # 대화 내역
            if conversation_id not in conversations:
                conversations[conversation_id] = []
            
            messages = conversations[conversation_id]
            messages.append(Message(role="user", content=request.message))
            
            # LLM 스트리밍
            llm = get_llm(request.model, temperature=request.temperature)
            context = "\n".join([f"{msg.role}: {msg.content}" for msg in messages[-10:]])
            
            full_response = ""
            async for chunk in llm.ask_async(context, stream=True):
                full_response += chunk.text
                yield f"data: {chunk.text}\n\n"
            
            # 전체 응답 저장
            messages.append(Message(role="assistant", content=full_response))
            
            # 완료 신호
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/analyze", response_model=TextAnalysis)
async def analyze_text(request: AnalysisRequest):
    """텍스트 분석 엔드포인트"""
    try:
        llm = get_llm("gpt-4o-mini")
        result = TextAnalysis()
        
        if request.analysis_type in ["sentiment", "all"]:
            sentiment_prompt = f"""
다음 텍스트의 감정을 분석하여 JSON으로 출력하세요:
{{
    "sentiment": "긍정/부정/중립",
    "confidence": 0.0-1.0,
    "emotions": ["감정1", "감정2"]
}}

텍스트: {request.text}
"""
            sentiment_reply = await llm.ask_async(sentiment_prompt)
            try:
                import json
                sentiment_data = json.loads(sentiment_reply.text)
                result.sentiment = SentimentAnalysis(**sentiment_data)
            except:
                pass
        
        if request.analysis_type in ["summary", "all"]:
            summary_reply = await llm.ask_async(
                f"다음 텍스트를 한 문장으로 요약하세요: {request.text}"
            )
            result.summary = summary_reply.text
        
        if request.analysis_type in ["keywords", "all"]:
            keywords_reply = await llm.ask_async(
                f"다음 텍스트의 핵심 키워드 5개를 추출하세요: {request.text}"
            )
            # 간단한 파싱
            result.keywords = [k.strip() for k in keywords_reply.text.split(",")][:5]
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    request: ImageAnalysisRequest = ImageAnalysisRequest()
):
    """이미지 분석 엔드포인트"""
    try:
        # 파일 확인
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 허용됩니다.")
        
        # 임시 파일로 저장
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            # 이미지 분석
            llm = get_llm("gpt-4o-mini")  # vision 지원 모델
            reply = await llm.ask_async(request.question, files=[tmp_path])
            
            return {
                "filename": file.filename,
                "question": request.question,
                "analysis": reply.text,
                "timestamp": datetime.now()
            }
            
        finally:
            # 임시 파일 삭제
            import os
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings")
async def create_embeddings(texts: List[str]):
    """텍스트 임베딩 생성"""
    try:
        if not texts:
            raise HTTPException(status_code=400, detail="텍스트가 필요합니다.")
        
        # 임베딩 모델 사용
        embed_llm = get_llm("text-embedding-3-small")
        embeddings = embed_llm.embed(texts)
        
        return {
            "model": "text-embedding-3-small",
            "embeddings": embeddings.embeddings,
            "dimensions": len(embeddings.embeddings[0]) if embeddings.embeddings else 0,
            "count": len(embeddings.embeddings)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """대화 내역 조회"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    
    messages = conversations[conversation_id]
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ],
        "message_count": len(messages)
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """대화 삭제"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    
    del conversations[conversation_id]
    return {"message": "대화가 삭제되었습니다.", "conversation_id": conversation_id}


# 백그라운드 작업 예제
async def process_long_task(task_id: str, prompt: str):
    """긴 작업을 백그라운드에서 처리"""
    llm = get_llm("gpt-4o-mini")
    reply = await llm.ask_async(prompt)
    # 실제로는 결과를 DB나 캐시에 저장
    print(f"Task {task_id} completed: {reply.text[:100]}...")


@app.post("/tasks/create")
async def create_task(
    prompt: str,
    background_tasks: BackgroundTasks
):
    """백그라운드 작업 생성"""
    task_id = str(uuid.uuid4())
    background_tasks.add_task(process_long_task, task_id, prompt)
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "작업이 백그라운드에서 처리되고 있습니다."
    }


# 헬스체크
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    try:
        # API 키 확인
        api_key_set = bool(os.getenv("OPENAI_API_KEY"))
        
        # 간단한 LLM 테스트
        llm_working = False
        try:
            test_llm = get_llm("gpt-4o-mini")
            # 실제 호출은 하지 않고 인스턴스 생성만 확인
            llm_working = True
        except:
            pass
        
        return {
            "status": "healthy" if api_key_set and llm_working else "unhealthy",
            "api_key_configured": api_key_set,
            "llm_available": llm_working,
            "active_conversations": len(conversations),
            "cached_llms": len(llm_cache),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        exit(1)
    
    print("🚀 FastAPI 서버 시작...")
    print("📍 http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)