"""
FastAPI + pyhub-llm 기본 통합 예제

이 파일은 pyhub-llm을 FastAPI와 연동하는 기본적인 예제를 제공합니다.
단일 질문 처리, 배치 처리, 스트리밍 응답 등의 기능을 포함합니다.
"""

import asyncio
import os
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from pyhub.llm import LLM


# =============================================================================
# Pydantic 모델 정의
# =============================================================================

class ChatRequest(BaseModel):
    """단일 채팅 요청"""
    message: str = Field(..., description="사용자 메시지", min_length=1)
    model: str = Field(default="gpt-4o-mini", description="사용할 LLM 모델")
    system_prompt: Optional[str] = Field(None, description="시스템 프롬프트")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="생성 온도")
    max_tokens: Optional[int] = Field(None, gt=0, description="최대 토큰 수")


class BatchRequest(BaseModel):
    """배치 처리 요청"""
    messages: List[str] = Field(..., description="처리할 메시지 목록", min_items=1, max_items=10)
    model: str = Field(default="gpt-4o-mini", description="사용할 LLM 모델")
    system_prompt: Optional[str] = Field(None, description="시스템 프롬프트")
    max_parallel: int = Field(default=3, ge=1, le=5, description="최대 병렬 처리 수")
    history_mode: str = Field(default="independent", description="히스토리 모드")


class SessionChatRequest(BaseModel):
    """세션 기반 채팅 요청"""
    message: str = Field(..., description="사용자 메시지", min_length=1)
    session_id: str = Field(..., description="세션 ID", min_length=1)
    model: str = Field(default="gpt-4o-mini", description="사용할 LLM 모델")


class ChatResponse(BaseModel):
    """채팅 응답"""
    response: str = Field(..., description="LLM 응답")
    model: str = Field(..., description="사용된 모델")
    usage: Optional[Dict[str, Any]] = Field(None, description="토큰 사용량")


class BatchResponse(BaseModel):
    """배치 처리 응답"""
    responses: List[ChatResponse] = Field(..., description="각 메시지에 대한 응답")
    total_count: int = Field(..., description="처리된 메시지 수")
    success_count: int = Field(..., description="성공한 메시지 수")
    execution_time: float = Field(..., description="총 실행 시간(초)")


# =============================================================================
# 전역 변수 및 설정
# =============================================================================

# LLM 인스턴스를 저장할 전역 딕셔너리
llm_instances: Dict[str, Any] = {}

# 간단한 인메모리 세션 저장소 (실제 환경에서는 Redis 등 사용 권장)
sessions: Dict[str, Any] = {}


# =============================================================================
# FastAPI 애플리케이션 설정
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행될 로직"""
    # 시작 시: 자주 사용되는 LLM 모델 미리 로드
    print("🚀 FastAPI + pyhub-llm 서비스 시작")
    print("📡 LLM 모델 초기화 중...")
    
    try:
        # 기본 모델 미리 로드
        llm_instances["gpt-4o-mini"] = LLM.create("gpt-4o-mini")
        print("✅ gpt-4o-mini 모델 로드 완료")
    except Exception as e:
        print(f"⚠️ 모델 로드 중 오류: {e}")
    
    yield
    
    # 종료 시: 정리 작업
    print("🛑 FastAPI + pyhub-llm 서비스 종료")
    llm_instances.clear()
    sessions.clear()


app = FastAPI(
    title="pyhub-llm FastAPI Integration",
    description="pyhub-llm을 FastAPI와 연동한 예제 서비스",
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_llm_instance(model: str) -> Any:
    """LLM 인스턴스를 가져오거나 생성"""
    if model not in llm_instances:
        try:
            llm_instances[model] = LLM.create(model)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 모델입니다: {model}. 오류: {str(e)}"
            )
    return llm_instances[model]


def validate_api_key():
    """API 키 검증 (간단한 예제)"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="API 키가 설정되지 않았습니다. OPENAI_API_KEY 환경변수를 설정해주세요."
        )


# =============================================================================
# API 엔드포인트
# =============================================================================

@app.get("/")
async def root():
    """루트 엔드포인트 - 서비스 상태 확인"""
    return {
        "service": "pyhub-llm FastAPI Integration",
        "status": "running",
        "endpoints": {
            "chat": "/chat - 단일 질문 처리",
            "batch": "/batch - 배치 처리", 
            "stream": "/stream - 스트리밍 응답",
            "session": "/chat/session - 세션 기반 채팅",
            "docs": "/docs - API 문서"
        }
    }


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "models_loaded": list(llm_instances.keys()),
        "active_sessions": len(sessions)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    단일 질문 처리 엔드포인트
    
    사용자의 질문에 대해 LLM이 응답을 생성합니다.
    """
    validate_api_key()
    
    try:
        # LLM 인스턴스 가져오기
        llm = get_llm_instance(request.model)
        
        # LLM 호출
        response = await llm.ask_async(
            input=request.message,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return ChatResponse(
            response=response.text,
            model=request.model,
            usage=response.usage.__dict__ if response.usage else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 처리 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/batch", response_model=BatchResponse)
async def batch_process(request: BatchRequest):
    """
    배치 처리 엔드포인트
    
    여러 질문을 동시에 처리하여 효율성을 높입니다.
    """
    validate_api_key()
    
    import time
    start_time = time.time()
    
    try:
        # LLM 인스턴스 가져오기
        llm = get_llm_instance(request.model)
        
        # 배치 처리 실행
        responses = await llm.batch(
            prompts=request.messages,
            system_prompt=request.system_prompt,
            max_parallel=request.max_parallel,
            history_mode=request.history_mode
        )
        
        # 응답 변환
        chat_responses = []
        success_count = 0
        
        for i, response in enumerate(responses):
            if "Error processing prompt" not in response.text:
                success_count += 1
            
            chat_responses.append(ChatResponse(
                response=response.text,
                model=request.model,
                usage=response.usage.__dict__ if response.usage else None
            ))
        
        execution_time = time.time() - start_time
        
        return BatchResponse(
            responses=chat_responses,
            total_count=len(request.messages),
            success_count=success_count,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"배치 처리 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    스트리밍 응답 엔드포인트
    
    LLM 응답을 실시간으로 스트리밍합니다.
    """
    validate_api_key()
    
    try:
        # LLM 인스턴스 가져오기
        llm = get_llm_instance(request.model)
        
        async def generate_stream():
            """스트리밍 응답 생성기"""
            try:
                async for chunk in llm.ask_stream_async(
                    input=request.message,
                    system_prompt=request.system_prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    if chunk.text:
                        yield f"data: {chunk.text}\n\n"
                
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: ERROR: {str(e)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"스트리밍 처리 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/chat/session", response_model=ChatResponse)
async def session_chat(request: SessionChatRequest):
    """
    세션 기반 채팅 엔드포인트
    
    세션을 유지하며 대화 히스토리를 관리합니다.
    """
    validate_api_key()
    
    try:
        # LLM 인스턴스 가져오기  
        llm = get_llm_instance(request.model)
        
        # 세션별로 별도의 LLM 인스턴스 관리
        session_key = f"{request.session_id}_{request.model}"
        
        if session_key not in sessions:
            # 새 세션 생성
            sessions[session_key] = LLM.create(request.model)
        
        session_llm = sessions[session_key]
        
        # 히스토리를 유지하며 응답 생성
        response = await session_llm.ask_async(
            input=request.message,
            use_history=True
        )
        
        return ChatResponse(
            response=response.text,
            model=request.model,
            usage=response.usage.__dict__ if response.usage else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"세션 채팅 처리 중 오류가 발생했습니다: {str(e)}"
        )


@app.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """
    세션 삭제 엔드포인트
    
    특정 세션의 대화 히스토리를 삭제합니다.
    """
    deleted_sessions = []
    
    # 해당 세션 ID로 시작하는 모든 세션 삭제
    for key in list(sessions.keys()):
        if key.startswith(f"{session_id}_"):
            del sessions[key]
            deleted_sessions.append(key)
    
    if not deleted_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"세션을 찾을 수 없습니다: {session_id}"
        )
    
    return {
        "message": f"세션이 삭제되었습니다: {session_id}",
        "deleted_sessions": deleted_sessions
    }


@app.get("/sessions")
async def list_sessions():
    """활성 세션 목록 조회"""
    session_info = {}
    
    for session_key in sessions.keys():
        session_id, model = session_key.rsplit("_", 1)
        if session_id not in session_info:
            session_info[session_id] = []
        session_info[session_id].append(model)
    
    return {
        "total_sessions": len(session_info),
        "sessions": session_info
    }


# =============================================================================
# 메인 실행부
# =============================================================================

if __name__ == "__main__":
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("💡 사용법: OPENAI_API_KEY=your_key python main.py")
    
    print("🌟 FastAPI + pyhub-llm 서비스를 시작합니다...")
    print("📖 API 문서: http://localhost:8000/docs")
    print("🔍 대화형 API: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )