"""Rate Limiting 미들웨어"""

import time
from typing import Dict, Tuple
from collections import defaultdict, deque
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class InMemoryRateLimiter:
    """인메모리 Rate Limiter (개발/테스트용)"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """
        요청이 허용되는지 확인
        
        Returns:
            (허용 여부, 남은 요청 수)
        """
        now = time.time()
        minute_ago = now - 60
        
        # 1분 이전 요청들 제거
        client_requests = self.requests[client_id]
        while client_requests and client_requests[0] < minute_ago:
            client_requests.popleft()
        
        # 요청 수 확인
        if len(client_requests) >= self.requests_per_minute:
            return False, 0
        
        # 새 요청 추가
        client_requests.append(now)
        remaining = self.requests_per_minute - len(client_requests)
        
        return True, remaining
    
    def get_reset_time(self, client_id: str) -> int:
        """다음 리셋 시간 (Unix timestamp)"""
        client_requests = self.requests[client_id]
        if not client_requests:
            return int(time.time() + 60)
        
        return int(client_requests[0] + 60)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate Limiting 미들웨어"""
    
    def __init__(
        self, 
        app,
        requests_per_minute: int = 60,
        exempt_paths: list = None
    ):
        super().__init__(app)
        self.limiter = InMemoryRateLimiter(requests_per_minute)
        self.exempt_paths = exempt_paths or ["/docs", "/redoc", "/openapi.json", "/health"]
    
    async def dispatch(self, request: Request, call_next):
        """요청 처리"""
        # 예외 경로 확인
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # 클라이언트 식별 (IP 주소 사용)
        client_ip = request.client.host
        
        # Rate limit 확인
        allowed, remaining = self.limiter.is_allowed(client_ip)
        
        if not allowed:
            reset_time = self.limiter.get_reset_time(client_ip)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "X-RateLimit-Limit": str(self.limiter.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": "60"
                }
            )
        
        # 요청 처리
        response = await call_next(request)
        
        # Rate limit 헤더 추가
        response.headers["X-RateLimit-Limit"] = str(self.limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(self.limiter.get_reset_time(client_ip))
        
        return response


def create_rate_limit_middleware(requests_per_minute: int = 60) -> RateLimitMiddleware:
    """Rate limit 미들웨어 팩토리 함수"""
    def middleware_factory(app):
        return RateLimitMiddleware(app, requests_per_minute)
    return middleware_factory