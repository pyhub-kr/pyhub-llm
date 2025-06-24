"""
FastAPI + pyhub-llm 고급 통합 예제

이 파일은 인증, Rate Limiting, 고급 서비스 등을 포함한
완전한 프로덕션 레벨의 FastAPI 애플리케이션 예제입니다.
"""

import os
import asyncio
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 로컬 모듈 임포트
from services.translation import (
    TranslationService, 
    TranslationRequest, 
    TranslationResponse,
    SummarizeRequest,
    SummarizeResponse,
    translation_service
)
from services.chat import ChatService, chat_service
from middleware.auth import api_key_auth, optional_api_key_auth
from middleware.rate_limit import RateLimitMiddleware
from main import ChatRequest, ChatResponse, BatchRequest, BatchResponse


# =============================================================================
# 애플리케이션 설정
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    print("🚀 고급 FastAPI + pyhub-llm 서비스 시작")
    print("🔐 인증 및 보안 기능 활성화")
    print("⚡ Rate Limiting 활성화")
    
    # 서비스 초기화
    try:
        # 기본 모델들 미리 로드
        await asyncio.create_task(preload_models())
        print("✅ 모델 사전 로딩 완료")
    except Exception as e:
        print(f"⚠️ 모델 로딩 중 오류: {e}")
    
    yield
    
    print("🛑 고급 FastAPI + pyhub-llm 서비스 종료")
    chat_service.clear_cache()


async def preload_models():
    """자주 사용되는 모델들을 미리 로드"""
    models = ["gpt-4o-mini", "gpt-4o"]
    for model in models:
        try:
            chat_service.get_llm(model)
            print(f"✅ {model} 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ {model} 모델 로드 실패: {e}")


# FastAPI 앱 생성
app = FastAPI(
    title="pyhub-llm Advanced FastAPI Integration",
    description="인증, Rate Limiting, 고급 서비스를 포함한 완전한 FastAPI 연동 예제",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting 미들웨어 추가
rate_limit_per_minute = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=rate_limit_per_minute
)


# =============================================================================
# 보안이 적용된 API 엔드포인트
# =============================================================================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "pyhub-llm Advanced FastAPI Integration",
        "version": "2.0.0",
        "features": [
            "API Key Authentication",
            "Rate Limiting", 
            "Translation Service",
            "Summarization Service",
            "Batch Processing",
            "Session Management"
        ],
        "endpoints": {
            "protected": {
                "/api/chat": "인증이 필요한 채팅",
                "/api/batch": "인증이 필요한 배치 처리",
                "/api/translate": "번역 서비스",
                "/api/summarize": "요약 서비스"
            },
            "public": {
                "/health": "헬스체크",
                "/docs": "API 문서",
                "/redoc": "대화형 API 문서"
            }
        }
    }


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트 (인증 불필요)"""
    return {
        "status": "healthy",
        "service": "pyhub-llm Advanced FastAPI",
        "rate_limit": f"{rate_limit_per_minute} requests/minute"
    }


# =============================================================================
# 인증이 필요한 API 엔드포인트
# =============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def protected_chat(
    request: ChatRequest,
    api_key: str = Depends(api_key_auth)
):
    """
    보호된 채팅 엔드포인트
    
    API 키 인증이 필요합니다.
    """
    try:
        response = await chat_service.process_message(
            message=request.message,
            model=request.model,
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
            detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/api/batch", response_model=BatchResponse)
async def protected_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(api_key_auth)
):
    """
    보호된 배치 처리 엔드포인트
    
    API 키 인증이 필요하며, 백그라운드에서 로깅을 수행합니다.
    """
    import time
    start_time = time.time()
    
    try:
        llm = chat_service.get_llm(request.model)
        
        responses = await llm.batch(
            prompts=request.messages,
            system_prompt=request.system_prompt,
            max_parallel=request.max_parallel,
            history_mode=request.history_mode
        )
        
        # 응답 처리
        chat_responses = []
        success_count = 0
        
        for response in responses:
            if "Error processing prompt" not in response.text:
                success_count += 1
            
            chat_responses.append(ChatResponse(
                response=response.text,
                model=request.model,
                usage=response.usage.__dict__ if response.usage else None
            ))
        
        execution_time = time.time() - start_time
        
        # 백그라운드 로깅 추가
        background_tasks.add_task(
            log_batch_request,
            api_key=api_key,
            message_count=len(request.messages),
            success_count=success_count,
            execution_time=execution_time
        )
        
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


@app.post("/api/stream")
async def protected_stream(
    request: ChatRequest,
    api_key: str = Depends(api_key_auth)
):
    """
    보호된 스트리밍 엔드포인트
    """
    try:
        async def generate_stream():
            try:
                async for chunk in chat_service.process_stream(
                    message=request.message,
                    model=request.model,
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
            detail=f"Error during streaming: {str(e)}"
        )


# =============================================================================
# 고급 서비스 엔드포인트
# =============================================================================

@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    api_key: str = Depends(api_key_auth)
):
    """
    텍스트 번역 서비스
    
    다양한 언어 간 번역을 지원합니다.
    """
    try:
        return await translation_service.translate(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during translation: {str(e)}"
        )


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize_text(
    request: SummarizeRequest,
    api_key: str = Depends(api_key_auth)
):
    """
    텍스트 요약 서비스
    
    긴 텍스트를 지정된 길이로 요약합니다.
    """
    try:
        return await translation_service.summarize(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during summarization: {str(e)}"
        )


@app.get("/api/supported-languages")
async def get_supported_languages(
    api_key: str = Depends(optional_api_key_auth)
):
    """
    지원하는 언어 목록 조회
    
    선택적 인증 (API 키가 있으면 더 자세한 정보 제공)
    """
    basic_languages = {
        "ko": "한국어",
        "en": "English", 
        "ja": "日本語",
        "zh": "中文"
    }
    
    if api_key:
        # 인증된 사용자에게는 더 많은 언어 제공
        return {
            "languages": translation_service.LANGUAGE_CODES,
            "total_count": len(translation_service.LANGUAGE_CODES),
            "premium_user": True
        }
    else:
        # 비인증 사용자에게는 기본 언어만 제공
        return {
            "languages": basic_languages,
            "total_count": len(basic_languages),
            "premium_user": False,
            "note": "API 키를 제공하면 더 많은 언어를 이용할 수 있습니다."
        }


# =============================================================================
# 관리자 엔드포인트
# =============================================================================

@app.get("/admin/stats")
async def get_service_stats(
    api_key: str = Depends(api_key_auth)
):
    """
    서비스 통계 조회
    
    관리자용 엔드포인트입니다.
    """
    # 실제로는 더 정교한 권한 체크가 필요
    admin_keys = os.getenv("ADMIN_API_KEYS", "").split(",")
    if api_key not in admin_keys:
        raise HTTPException(
            status_code=403,
            detail="관리자 권한이 필요합니다."
        )
    
    return {
        "service_status": "running",
        "loaded_models": list(chat_service._llm_cache.keys()),
        "supported_languages": len(translation_service.LANGUAGE_CODES),
        "rate_limit": f"{rate_limit_per_minute} requests/minute"
    }


# =============================================================================
# 백그라운드 작업
# =============================================================================

async def log_batch_request(
    api_key: str,
    message_count: int,
    success_count: int,
    execution_time: float
):
    """배치 요청 로깅 (백그라운드 작업)"""
    # 실제로는 데이터베이스나 로깅 시스템에 저장
    print(f"📊 배치 요청 로그:")
    print(f"   - API Key: {api_key[:8]}...")
    print(f"   - 메시지 수: {message_count}")
    print(f"   - 성공 수: {success_count}")
    print(f"   - 실행 시간: {execution_time:.2f}초")


# =============================================================================
# 메인 실행부
# =============================================================================

if __name__ == "__main__":
    # 환경 변수 확인
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️ 필수 환경변수가 설정되지 않았습니다: {missing_vars}")
        print("💡 .env 파일을 생성하거나 환경변수를 설정해주세요.")
    
    # API 키 정보 출력
    api_keys_env = os.getenv("ALLOWED_API_KEYS", "")
    if not api_keys_env:
        default_key = os.getenv("API_SECRET_KEY", "demo-key-12345")
        print(f"🔑 개발용 API 키: {default_key}")
        print(f"💡 사용법: curl -H 'Authorization: Bearer {default_key}' ...")
    
    print("🌟 고급 FastAPI + pyhub-llm 서비스를 시작합니다...")
    print("📖 API 문서: http://localhost:8000/docs")
    print("🔍 대화형 API: http://localhost:8000/redoc")
    print(f"🚦 Rate Limit: {rate_limit_per_minute} requests/minute")
    
    uvicorn.run(
        "advanced:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", "8000")),
        reload=os.getenv("FASTAPI_RELOAD", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info")
    )