"""채팅 서비스 로직"""

from typing import Optional, Dict, Any, AsyncGenerator
from pyhub.llm import LLM
from pyhub.llm.types import Reply


class ChatService:
    """채팅 관련 비즈니스 로직을 담당하는 서비스 클래스"""
    
    def __init__(self):
        self._llm_cache: Dict[str, Any] = {}
    
    def get_llm(self, model: str) -> Any:
        """LLM 인스턴스를 캐시에서 가져오거나 생성"""
        if model not in self._llm_cache:
            self._llm_cache[model] = LLM.create(model)
        return self._llm_cache[model]
    
    async def process_message(
        self,
        message: str,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Reply:
        """단일 메시지 처리"""
        llm = self.get_llm(model)
        
        return await llm.ask_async(
            input=message,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def process_stream(
        self,
        message: str,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[Reply, None]:
        """스트리밍 메시지 처리"""
        llm = self.get_llm(model)
        
        async for chunk in llm.ask_stream_async(
            input=message,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ):
            yield chunk
    
    def clear_cache(self):
        """LLM 캐시 정리"""
        self._llm_cache.clear()


# 전역 인스턴스
chat_service = ChatService()