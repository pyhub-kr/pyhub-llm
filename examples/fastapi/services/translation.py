"""번역 서비스 예제"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pyhub.llm import LLM


class TranslationRequest(BaseModel):
    """번역 요청"""
    text: str = Field(..., description="번역할 텍스트", min_length=1)
    target_language: str = Field(..., description="목표 언어 (예: ko, en, ja, zh)")
    source_language: Optional[str] = Field(None, description="원본 언어 (자동 감지)")
    model: str = Field(default="gpt-4o-mini", description="사용할 LLM 모델")


class TranslationResponse(BaseModel):
    """번역 응답"""
    original_text: str = Field(..., description="원본 텍스트")
    translated_text: str = Field(..., description="번역된 텍스트")
    source_language: str = Field(..., description="감지된 원본 언어")
    target_language: str = Field(..., description="목표 언어")
    confidence: Optional[float] = Field(None, description="번역 신뢰도")


class SummarizeRequest(BaseModel):
    """요약 요청"""
    text: str = Field(..., description="요약할 텍스트", min_length=10)
    max_length: int = Field(default=200, ge=50, le=1000, description="최대 요약 길이")
    language: str = Field(default="ko", description="요약 언어")
    model: str = Field(default="gpt-4o-mini", description="사용할 LLM 모델")


class SummarizeResponse(BaseModel):
    """요약 응답"""
    original_text: str = Field(..., description="원본 텍스트")
    summary: str = Field(..., description="요약된 텍스트")
    original_length: int = Field(..., description="원본 텍스트 길이")
    summary_length: int = Field(..., description="요약 텍스트 길이")
    compression_ratio: float = Field(..., description="압축 비율")


class TranslationService:
    """번역 서비스"""
    
    LANGUAGE_CODES = {
        "ko": "한국어",
        "en": "English", 
        "ja": "日本語",
        "zh": "中文",
        "es": "Español",
        "fr": "Français",
        "de": "Deutsch",
        "it": "Italiano",
        "pt": "Português",
        "ru": "Русский"
    }
    
    def __init__(self):
        self._llm_cache: Dict[str, Any] = {}
    
    def get_llm(self, model: str) -> Any:
        """LLM 인스턴스 가져오기"""
        if model not in self._llm_cache:
            self._llm_cache[model] = LLM.create(model)
        return self._llm_cache[model]
    
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """텍스트 번역"""
        llm = self.get_llm(request.model)
        
        # 목표 언어명 가져오기
        target_lang_name = self.LANGUAGE_CODES.get(
            request.target_language, 
            request.target_language
        )
        
        # 번역 프롬프트 생성
        if request.source_language:
            source_lang_name = self.LANGUAGE_CODES.get(
                request.source_language,
                request.source_language
            )
            prompt = f"""다음 텍스트를 {source_lang_name}에서 {target_lang_name}으로 번역해주세요.

원본 텍스트:
{request.text}

번역 시 주의사항:
- 원문의 의미와 뉘앙스를 정확히 전달해주세요
- 자연스러운 {target_lang_name} 표현을 사용해주세요
- 번역된 텍스트만 응답해주세요"""
        else:
            prompt = f"""다음 텍스트를 {target_lang_name}으로 번역해주세요.

원본 텍스트:
{request.text}

번역 시 주의사항:
- 원문의 언어를 자동으로 감지해주세요
- 원문의 의미와 뉘앙스를 정확히 전달해주세요  
- 자연스러운 {target_lang_name} 표현을 사용해주세요
- 번역된 텍스트만 응답해주세요"""
        
        # 번역 실행
        response = await llm.ask_async(
            input=prompt,
            temperature=0.3,  # 일관성을 위해 낮은 온도 사용
            max_tokens=len(request.text) * 2  # 충분한 토큰 할당
        )
        
        # 원본 언어 감지 (간단한 휴리스틱)
        detected_language = self._detect_language(request.text)
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=response.text.strip(),
            source_language=request.source_language or detected_language,
            target_language=request.target_language,
            confidence=0.95  # 실제로는 더 정교한 신뢰도 계산 필요
        )
    
    async def summarize(self, request: SummarizeRequest) -> SummarizeResponse:
        """텍스트 요약"""
        llm = self.get_llm(request.model)
        
        # 요약 프롬프트 생성
        prompt = f"""다음 텍스트를 {request.max_length}자 이내로 요약해주세요.

원본 텍스트:
{request.text}

요약 시 주의사항:
- 핵심 내용과 주요 포인트를 유지해주세요
- {request.language}로 작성해주세요
- 간결하고 명확하게 작성해주세요
- {request.max_length}자를 초과하지 마세요"""
        
        # 요약 실행
        response = await llm.ask_async(
            input=prompt,
            temperature=0.5,
            max_tokens=request.max_length + 100  # 여유분 추가
        )
        
        summary = response.text.strip()
        original_length = len(request.text)
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        
        return SummarizeResponse(
            original_text=request.text,
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio
        )
    
    def _detect_language(self, text: str) -> str:
        """간단한 언어 감지 (실제로는 더 정교한 라이브러리 사용 권장)"""
        # 한글 문자 체크
        if any('\uac00' <= char <= '\ud7af' for char in text):
            return "ko"
        
        # 일본어 문자 체크  
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' 
               for char in text):
            return "ja"
        
        # 중국어 문자 체크
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return "zh"
        
        # 기본값은 영어
        return "en"


# 전역 인스턴스
translation_service = TranslationService()