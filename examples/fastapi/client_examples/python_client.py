"""
Python 클라이언트 예제

pyhub-llm FastAPI 서비스를 사용하는 Python 클라이언트 예제입니다.
"""

import asyncio
import httpx
from typing import List, Optional, Dict, Any


class PyHubLLMClient:
    """pyhub-llm FastAPI 서비스 클라이언트"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def chat(
        self, 
        message: str, 
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        protected: bool = False
    ) -> Dict[str, Any]:
        """단일 채팅 요청"""
        endpoint = "/api/chat" if protected else "/chat"
        
        payload = {
            "message": message,
            "model": model
        }
        
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    async def batch(
        self,
        messages: List[str],
        model: str = "gpt-4o-mini",
        max_parallel: int = 3,
        history_mode: str = "independent",
        protected: bool = False
    ) -> Dict[str, Any]:
        """배치 처리 요청"""
        endpoint = "/api/batch" if protected else "/batch"
        
        payload = {
            "messages": messages,
            "model": model,
            "max_parallel": max_parallel,
            "history_mode": history_mode
        }
        
        async with httpx.AsyncClient(timeout=self.timeout * len(messages)) as client:
            response = await client.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """번역 요청"""
        payload = {
            "text": text,
            "target_language": target_language,
            "model": model
        }
        
        if source_language:
            payload["source_language"] = source_language
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/translate",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    async def summarize(
        self,
        text: str,
        max_length: int = 200,
        language: str = "ko",
        model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """요약 요청"""
        payload = {
            "text": text,
            "max_length": max_length,
            "language": language,
            "model": model
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/summarize",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    async def get_supported_languages(self) -> Dict[str, Any]:
        """지원하는 언어 목록 조회"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/api/supported-languages",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스체크"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()


# =============================================================================
# 사용 예제
# =============================================================================

async def basic_examples():
    """기본 사용 예제"""
    print("🚀 기본 사용 예제")
    
    # 클라이언트 생성 (인증 없이)
    client = PyHubLLMClient()
    
    # 헬스체크
    health = await client.health_check()
    print(f"✅ 서비스 상태: {health['status']}")
    
    # 단일 채팅
    print("\n💬 단일 채팅 예제:")
    chat_result = await client.chat(
        message="FastAPI와 pyhub-llm의 조합에 대해 간단히 설명해주세요.",
        model="gpt-4o-mini"
    )
    print(f"응답: {chat_result['response'][:100]}...")
    
    # 배치 처리
    print("\n📦 배치 처리 예제:")
    questions = [
        "Python의 주요 특징 3가지는?",
        "FastAPI를 사용하는 이유는?",
        "LLM API 서비스의 장점은?"
    ]
    
    batch_result = await client.batch(questions)
    print(f"✅ {batch_result['success_count']}/{batch_result['total_count']} 성공")
    print(f"⏱️ 실행 시간: {batch_result['execution_time']:.2f}초")
    
    for i, response in enumerate(batch_result['responses']):
        print(f"Q{i+1}: {questions[i]}")
        print(f"A{i+1}: {response['response'][:80]}...")
        print()


async def authenticated_examples():
    """인증이 필요한 예제"""
    print("🔐 인증 예제")
    
    # API 키가 필요한 클라이언트
    api_key = "demo-key-12345"  # 실제 환경에서는 환경변수 사용
    client = PyHubLLMClient(api_key=api_key)
    
    try:
        # 보호된 채팅
        print("\n🛡️ 보호된 채팅:")
        protected_result = await client.chat(
            message="인증이 필요한 보안 채팅입니다!",
            protected=True
        )
        print(f"응답: {protected_result['response'][:100]}...")
        
        # 번역 서비스
        print("\n🌍 번역 서비스:")
        translation = await client.translate(
            text="Hello, how are you today?",
            target_language="ko"
        )
        print(f"원문: {translation['original_text']}")
        print(f"번역: {translation['translated_text']}")
        
        # 요약 서비스
        print("\n📝 요약 서비스:")
        long_text = """
        FastAPI는 Python으로 API를 구축하기 위한 현대적이고 빠른 웹 프레임워크입니다.
        이 프레임워크는 표준 Python 타입 힌트를 기반으로 하며, 자동 문서 생성,
        데이터 검증, 직렬화 등의 기능을 제공합니다. 또한 높은 성능을 자랑하며
        NodeJS 및 Go와 견줄 만한 속도를 보여줍니다. pyhub-llm과 함께 사용하면
        강력한 LLM 기반 웹 서비스를 쉽게 구축할 수 있습니다.
        """
        
        summary = await client.summarize(
            text=long_text.strip(),
            max_length=100
        )
        print(f"원본 길이: {summary['original_length']}자")
        print(f"요약 길이: {summary['summary_length']}자")
        print(f"압축률: {summary['compression_ratio']:.2%}")
        print(f"요약: {summary['summary']}")
        
        # 지원 언어 목록
        print("\n🌐 지원 언어:")
        languages = await client.get_supported_languages()
        if languages.get('premium_user'):
            print(f"✨ 프리미엄 사용자: {languages['total_count']}개 언어 지원")
        else:
            print(f"📚 기본 사용자: {languages['total_count']}개 언어 지원")
        
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP 오류: {e.response.status_code}")
        print(f"상세: {e.response.text}")
    except Exception as e:
        print(f"❌ 오류: {str(e)}")


async def advanced_batch_examples():
    """고급 배치 처리 예제"""
    print("⚡ 고급 배치 처리 예제")
    
    client = PyHubLLMClient()
    
    # 순차적 대화 (각 응답이 다음 질문의 컨텍스트가 됨)
    print("\n🔗 순차적 대화:")
    sequential_questions = [
        "피보나치 수열에 대해 설명해주세요.",
        "이것을 Python으로 구현해주세요.",
        "이 구현의 시간 복잡도는 어떻게 되나요?",
        "더 효율적인 방법이 있을까요?"
    ]
    
    sequential_result = await client.batch(
        messages=sequential_questions,
        history_mode="sequential"
    )
    
    print("📈 순차적 대화 결과:")
    for i, (question, response) in enumerate(zip(sequential_questions, sequential_result['responses'])):
        print(f"\n단계 {i+1}: {question}")
        print(f"응답: {response['response'][:120]}...")
    
    # 공유 컨텍스트 (모든 질문이 동일한 초기 컨텍스트 공유)
    print("\n🤝 공유 컨텍스트 배치:")
    context_questions = [
        "FastAPI의 주요 특징은?",
        "pyhub-llm과의 연동 방법은?",
        "실제 서비스에서 고려할 점은?"
    ]
    
    shared_result = await client.batch(
        messages=context_questions,
        history_mode="shared"
    )
    
    print("🌟 공유 컨텍스트 결과:")
    for i, (question, response) in enumerate(zip(context_questions, shared_result['responses'])):
        print(f"\nQ{i+1}: {question}")
        print(f"A{i+1}: {response['response'][:100]}...")


async def main():
    """메인 실행 함수"""
    print("🌟 pyhub-llm FastAPI 클라이언트 예제")
    print("=" * 50)
    
    try:
        await basic_examples()
        print("\n" + "=" * 50)
        
        await authenticated_examples()
        print("\n" + "=" * 50)
        
        await advanced_batch_examples()
        
    except httpx.ConnectError:
        print("❌ 서버에 연결할 수 없습니다.")
        print("💡 먼저 FastAPI 서버를 실행해주세요:")
        print("   python main.py  또는  python advanced.py")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")
    
    print("\n✅ 모든 예제 완료!")


if __name__ == "__main__":
    asyncio.run(main())