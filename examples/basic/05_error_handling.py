#!/usr/bin/env python3
"""
예제: 기본 에러 처리
난이도: 초급
설명: LLM 사용 시 발생할 수 있는 에러를 처리하는 방법
요구사항: 
  - pyhub-llm (pip install pyhub-llm)
  - OPENAI_API_KEY 환경 변수
"""

import os
import time
from pyhub.llm import LLM
from pyhub.llm.exceptions import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError
)


def example_basic_error_handling():
    """기본 에러 처리 예제"""
    print("\n🛡️ 기본 에러 처리")
    print("-" * 50)
    
    # 잘못된 API 키로 시도
    print("1. 인증 에러 처리:")
    try:
        # 임시로 잘못된 API 키 설정
        old_key = os.environ.get("OPENAI_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = "invalid-key"
        
        llm = LLM.create("gpt-4o-mini")
        reply = llm.ask("안녕하세요")
        
    except AuthenticationError as e:
        print(f"❌ 인증 실패: API 키를 확인해주세요.")
        print(f"   에러 메시지: {str(e)[:100]}...")
    except Exception as e:
        print(f"❌ 예상치 못한 에러: {type(e).__name__}")
    finally:
        # API 키 복원
        os.environ["OPENAI_API_KEY"] = old_key
    
    # 정상 API 키로 복원 후 계속
    llm = LLM.create("gpt-4o-mini")
    
    # 2. 잘못된 요청 처리
    print("\n2. 잘못된 요청 처리:")
    try:
        # 너무 긴 입력 (토큰 제한 초과 시뮬레이션)
        very_long_text = "A" * 50000  # 매우 긴 텍스트
        reply = llm.ask(very_long_text)
    except InvalidRequestError as e:
        print(f"❌ 잘못된 요청: 입력이 너무 깁니다.")
        print(f"   에러 메시지: {str(e)[:100]}...")
    except Exception as e:
        print(f"❌ 에러 발생: {type(e).__name__}")
        print(f"   토큰 제한을 초과했을 수 있습니다.")


def example_retry_logic():
    """재시도 로직 예제"""
    print("\n🔄 재시도 로직")
    print("-" * 50)
    
    llm = LLM.create("gpt-4o-mini")
    
    def ask_with_retry(prompt: str, max_retries: int = 3) -> str:
        """재시도 로직이 포함된 LLM 호출"""
        for attempt in range(max_retries):
            try:
                print(f"시도 {attempt + 1}/{max_retries}...")
                reply = llm.ask(prompt)
                print("✅ 성공!")
                return reply.text
            
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 지수 백오프
                    print(f"⏳ 속도 제한. {wait_time}초 대기 중...")
                    time.sleep(wait_time)
                else:
                    print("❌ 최대 재시도 횟수 초과")
                    raise
            
            except Exception as e:
                print(f"❌ 에러 발생: {type(e).__name__}")
                if attempt < max_retries - 1:
                    print("재시도 중...")
                else:
                    raise
        
        return "재시도 실패"
    
    # 정상 요청
    result = ask_with_retry("파이썬의 장점을 한 문장으로 설명해주세요.")
    print(f"\n응답: {result}")


def example_fallback_handling():
    """폴백 처리 예제"""
    print("\n🔀 폴백 처리")
    print("-" * 50)
    
    def ask_with_fallback(prompt: str, models: list) -> str:
        """여러 모델을 시도하는 폴백 로직"""
        for i, model in enumerate(models):
            try:
                print(f"모델 시도: {model}")
                llm = LLM.create(model)
                reply = llm.ask(prompt)
                print(f"✅ {model} 성공!")
                return reply.text
            
            except Exception as e:
                print(f"❌ {model} 실패: {type(e).__name__}")
                if i < len(models) - 1:
                    print(f"다음 모델로 시도합니다...")
                else:
                    print("모든 모델이 실패했습니다.")
                    raise
        
        return "모든 모델 실패"
    
    # 여러 모델 시도 (일부는 존재하지 않을 수 있음)
    models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    prompt = "AI의 미래를 한 문장으로 예측해주세요."
    
    try:
        result = ask_with_fallback(prompt, models)
        print(f"\n최종 응답: {result}")
    except Exception as e:
        print(f"\n❌ 모든 시도 실패: {e}")


def example_timeout_handling():
    """타임아웃 처리 예제"""
    print("\n⏱️ 타임아웃 처리")
    print("-" * 50)
    
    import signal
    
    class TimeoutException(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutException("요청 시간 초과")
    
    def ask_with_timeout(llm, prompt: str, timeout_seconds: int = 10):
        """타임아웃이 있는 LLM 호출"""
        # 타임아웃 설정 (Unix 시스템에서만 작동)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        try:
            print(f"요청 중... (타임아웃: {timeout_seconds}초)")
            reply = llm.ask(prompt)
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # 타임아웃 해제
            return reply.text
        
        except TimeoutException:
            print(f"❌ {timeout_seconds}초 내에 응답을 받지 못했습니다.")
            return None
        except Exception as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # 타임아웃 해제
            raise
    
    llm = LLM.create("gpt-4o-mini")
    
    # 짧은 타임아웃 테스트
    result = ask_with_timeout(
        llm,
        "짧은 답변: 1+1은?",
        timeout_seconds=10
    )
    
    if result:
        print(f"✅ 응답: {result}")
    else:
        print("응답을 받지 못했습니다.")


def example_graceful_degradation():
    """우아한 성능 저하 예제"""
    print("\n📉 우아한 성능 저하")
    print("-" * 50)
    
    class SmartLLM:
        """에러 시 자동으로 성능을 낮추는 LLM 래퍼"""
        
        def __init__(self):
            self.quality_levels = [
                {"model": "gpt-4o", "max_tokens": 2000},
                {"model": "gpt-4o-mini", "max_tokens": 1000},
                {"model": "gpt-3.5-turbo", "max_tokens": 500}
            ]
            self.current_level = 0
        
        def ask(self, prompt: str) -> str:
            while self.current_level < len(self.quality_levels):
                config = self.quality_levels[self.current_level]
                
                try:
                    print(f"시도 중: {config['model']} (최대 토큰: {config['max_tokens']})")
                    llm = LLM.create(config['model'])
                    reply = llm.ask(prompt)
                    
                    # 성공하면 레벨 유지
                    print(f"✅ 성공!")
                    return reply.text
                
                except Exception as e:
                    print(f"❌ 실패: {type(e).__name__}")
                    self.current_level += 1
                    
                    if self.current_level < len(self.quality_levels):
                        print("품질 레벨을 낮춰서 재시도합니다...")
                    else:
                        print("모든 품질 레벨에서 실패했습니다.")
                        return "죄송합니다. 현재 서비스를 사용할 수 없습니다."
            
            return "서비스 이용 불가"
    
    # 사용 예
    smart_llm = SmartLLM()
    response = smart_llm.ask("인공지능의 역사를 간단히 설명해주세요.")
    print(f"\n최종 응답: {response[:100]}...")


def main():
    """에러 처리 예제 메인 함수"""
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        return
    
    print("🛡️ 에러 처리 예제")
    print("=" * 50)
    
    try:
        # 1. 기본 에러 처리
        example_basic_error_handling()
        
        # 2. 재시도 로직
        example_retry_logic()
        
        # 3. 폴백 처리
        example_fallback_handling()
        
        # 4. 타임아웃 처리
        example_timeout_handling()
        
        # 5. 우아한 성능 저하
        example_graceful_degradation()
        
        print("\n✅ 모든 에러 처리 예제 완료!")
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()