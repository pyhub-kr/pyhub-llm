"""
LLM 모듈 설정 관리
"""

import os


class LLMSettings:
    """LLM 모듈의 설정을 관리하는 클래스"""

    def __init__(self):
        # Trace 관련 설정
        self.trace_enabled = self._parse_bool("PYHUB_LLM_TRACE", False)
        self.trace_function_calls = self._parse_bool("PYHUB_LLM_TRACE_FUNCTION_CALLS", False)
        self.trace_level = os.getenv("PYHUB_LLM_TRACE_LEVEL", "INFO").upper()

        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("PYHUB_LLM_OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("PYHUB_LLM_ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("PYHUB_LLM_GOOGLE_API_KEY")
        self.upstage_api_key = os.getenv("UPSTAGE_API_KEY") or os.getenv("PYHUB_LLM_UPSTAGE_API_KEY")

        # Base URLs
        self.openai_base_url = (
            os.getenv("OPENAI_BASE_URL") or os.getenv("PYHUB_LLM_OPENAI_BASE_URL") or "https://api.openai.com/v1"
        )
        self.anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("PYHUB_LLM_ANTHROPIC_BASE_URL")
        self.google_base_url = os.getenv("GOOGLE_BASE_URL") or os.getenv("PYHUB_LLM_GOOGLE_BASE_URL")
        self.upstage_base_url = (
            os.getenv("UPSTAGE_BASE_URL") or os.getenv("PYHUB_LLM_UPSTAGE_BASE_URL") or "https://api.upstage.ai/v1"
        )
        self.ollama_base_url = (
            os.getenv("OLLAMA_BASE_URL") or os.getenv("PYHUB_LLM_OLLAMA_BASE_URL") or "http://localhost:11434"
        )

    def _parse_bool(self, env_var: str, default: bool) -> bool:
        """환경변수를 bool 값으로 파싱"""
        value = os.getenv(env_var, str(default)).lower()
        return value in ("true", "1", "yes", "on")


# 전역 인스턴스
llm_settings = LLMSettings()


__all__ = ["LLMSettings", "llm_settings"]
