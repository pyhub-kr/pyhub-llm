"""인증 미들웨어"""

import os
import secrets
from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class APIKeyAuth:
    """API 키 기반 인증"""
    
    def __init__(self):
        self.security = HTTPBearer(auto_error=False)
        # 환경변수에서 허용된 API 키들을 가져옴 (콤마로 구분)
        api_keys_env = os.getenv("ALLOWED_API_KEYS", "")
        self.allowed_keys = set(key.strip() for key in api_keys_env.split(",") if key.strip())
        
        # 개발 환경용 기본 키 생성
        if not self.allowed_keys:
            default_key = os.getenv("API_SECRET_KEY", "dev-key-" + secrets.token_hex(16))
            self.allowed_keys.add(default_key)
            print(f"🔑 개발용 API 키: {default_key}")
    
    async def __call__(
        self, 
        credentials: Optional[HTTPAuthorizationCredentials] = Security(HTTPBearer(auto_error=False))
    ) -> str:
        """API 키 검증"""
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="API 키가 필요합니다. Authorization: Bearer <api_key> 헤더를 포함해주세요."
            )
        
        if credentials.credentials not in self.allowed_keys:
            raise HTTPException(
                status_code=401,
                detail="유효하지 않은 API 키입니다."
            )
        
        return credentials.credentials


class OptionalAPIKeyAuth:
    """선택적 API 키 인증 (키가 있으면 검증, 없으면 통과)"""
    
    def __init__(self):
        self.security = HTTPBearer(auto_error=False)
        api_keys_env = os.getenv("ALLOWED_API_KEYS", "")
        self.allowed_keys = set(key.strip() for key in api_keys_env.split(",") if key.strip())
    
    async def __call__(
        self, 
        credentials: Optional[HTTPAuthorizationCredentials] = Security(HTTPBearer(auto_error=False))
    ) -> Optional[str]:
        """선택적 API 키 검증"""
        if not credentials:
            return None
        
        if self.allowed_keys and credentials.credentials not in self.allowed_keys:
            raise HTTPException(
                status_code=401,
                detail="유효하지 않은 API 키입니다."
            )
        
        return credentials.credentials


# 전역 인스턴스
api_key_auth = APIKeyAuth()
optional_api_key_auth = OptionalAPIKeyAuth()


def get_api_key_auth() -> APIKeyAuth:
    """의존성 주입용 함수"""
    return api_key_auth


def get_optional_api_key_auth() -> OptionalAPIKeyAuth:
    """선택적 인증용 의존성 주입 함수"""
    return optional_api_key_auth