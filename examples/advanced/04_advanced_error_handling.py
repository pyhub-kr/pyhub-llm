#!/usr/bin/env python3
"""
예제: 고급 에러 처리
난이도: 고급
설명: 프로덕션 환경을 위한 고급 에러 처리 패턴
요구사항:
  - pyhub-llm (pip install "pyhub-llm[all]")
  - OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from pyhub.llm import LLM
from pyhub.llm.exceptions import RateLimitError

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """에러 심각도"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """에러 컨텍스트 정보"""

    error_type: str
    severity: ErrorSeverity
    timestamp: datetime
    retry_count: int
    context: Dict[str, Any]
    traceback: Optional[str] = None


class ErrorHandler:
    """중앙집중식 에러 핸들러"""

    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.error_callbacks: Dict[str, List[Callable]] = {}

    def register_callback(self, error_type: str, callback: Callable):
        """에러 타입별 콜백 등록"""
        if error_type not in self.error_callbacks:
            self.error_callbacks[error_type] = []
        self.error_callbacks[error_type].append(callback)

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """에러 처리"""
        error_context = ErrorContext(
            error_type=type(error).__name__,
            severity=self._determine_severity(error),
            timestamp=datetime.now(),
            retry_count=context.get("retry_count", 0) if context else 0,
            context=context or {},
            traceback=str(error),
        )

        # 에러 기록
        self.error_history.append(error_context)
        logger.error(f"Error occurred: {error_context}")

        # 콜백 실행
        callbacks = self.error_callbacks.get(error_context.error_type, [])
        callbacks.extend(self.error_callbacks.get("*", []))  # 와일드카드

        for callback in callbacks:
            try:
                callback(error, error_context)
            except Exception as e:
                logger.error(f"Error in callback: {e}")

        return error_context

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """에러 심각도 결정"""
        if "authentication" in str(error).lower() or "api key" in str(error).lower():
            return ErrorSeverity.CRITICAL
        elif isinstance(error, RateLimitError):
            return ErrorSeverity.MEDIUM
        elif "invalid" in str(error).lower() or "token" in str(error).lower():
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.HIGH

    def get_error_stats(self) -> Dict[str, Any]:
        """에러 통계"""
        if not self.error_history:
            return {"total": 0, "by_type": {}, "by_severity": {}}

        stats = {"total": len(self.error_history), "by_type": {}, "by_severity": {}, "recent_errors": []}

        for error in self.error_history:
            # 타입별 집계
            stats["by_type"][error.error_type] = stats["by_type"].get(error.error_type, 0) + 1
            # 심각도별 집계
            stats["by_severity"][error.severity.value] = stats["by_severity"].get(error.severity.value, 0) + 1

        # 최근 에러
        stats["recent_errors"] = [
            {
                "type": e.error_type,
                "severity": e.severity.value,
                "timestamp": e.timestamp.isoformat(),
                "message": e.traceback[:100],
            }
            for e in self.error_history[-5:]
        ]

        return stats


class RetryStrategy:
    """재시도 전략"""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, retry_count: int) -> float:
        """재시도 지연 시간 계산"""
        delay = min(self.initial_delay * (self.exponential_base**retry_count), self.max_delay)

        if self.jitter:
            import random

            delay *= 0.5 + random.random()

        return delay


def with_retry(strategy: RetryStrategy = None, error_handler: ErrorHandler = None):
    """재시도 데코레이터"""
    if strategy is None:
        strategy = RetryStrategy()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(strategy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if error_handler:
                        error_handler.handle_error(
                            e,
                            {
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "args": str(args)[:100],
                                "kwargs": str(kwargs)[:100],
                            },
                        )

                    if attempt < strategy.max_retries:
                        delay = strategy.get_delay(attempt)
                        logger.info(
                            f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1}/{strategy.max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}")
                        raise

            raise last_error

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(strategy.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if error_handler:
                        error_handler.handle_error(
                            e,
                            {
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "args": str(args)[:100],
                                "kwargs": str(kwargs)[:100],
                            },
                        )

                    if attempt < strategy.max_retries:
                        delay = strategy.get_delay(attempt)
                        logger.info(
                            f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1}/{strategy.max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}")
                        raise

            raise last_error

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


class CircuitBreaker:
    """서킷 브레이커 패턴"""

    def __init__(
        self, failure_threshold: int = 5, recovery_timeout: float = 60.0, expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise Exception(f"Circuit breaker is open for {func.__name__}")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception:
                self._on_failure()
                raise

        return wrapper

    def _should_attempt_reset(self) -> bool:
        """리셋 시도 여부"""
        return self.last_failure_time and datetime.now() - self.last_failure_time > timedelta(
            seconds=self.recovery_timeout
        )

    def _on_success(self):
        """성공 시 처리"""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None

    def _on_failure(self):
        """실패 시 처리"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RateLimiter:
    """속도 제한기"""

    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()

            # 오래된 호출 제거
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]

            if len(self.calls) >= self.max_calls:
                wait_time = self.time_window - (now - self.calls[0])
                raise RateLimitError(f"Rate limit exceeded. Please wait {wait_time:.2f} seconds.")

            self.calls.append(now)
            return func(*args, **kwargs)

        return wrapper


class FallbackLLM:
    """폴백 LLM 시스템"""

    def __init__(self, models: List[str], error_handler: ErrorHandler = None):
        self.models = models
        self.error_handler = error_handler or ErrorHandler()
        self.current_model_index = 0
        self.llm_instances = {}

    def _get_llm(self, model: str) -> LLM:
        """LLM 인스턴스 가져오기"""
        if model not in self.llm_instances:
            self.llm_instances[model] = LLM.create(model)
        return self.llm_instances[model]

    @with_retry(RetryStrategy(max_retries=2))
    def ask(self, prompt: str, **kwargs) -> Any:
        """폴백을 지원하는 ask 메서드"""
        errors = []

        for i, model in enumerate(self.models[self.current_model_index :], self.current_model_index):
            try:
                logger.info(f"Trying model: {model}")
                llm = self._get_llm(model)
                result = llm.ask(prompt, **kwargs)

                # 성공하면 현재 모델 인덱스 업데이트
                self.current_model_index = i
                return result

            except Exception as e:
                errors.append((model, e))
                self.error_handler.handle_error(e, {"model": model, "prompt": prompt[:100], "fallback_attempt": i + 1})

                if i < len(self.models) - 1:
                    logger.warning(f"Model {model} failed, trying next model...")
                else:
                    logger.error("All models failed")

        # 모든 모델이 실패한 경우
        raise Exception(f"All models failed: {errors}")


def example_error_handler_setup():
    """에러 핸들러 설정 예제"""
    print("\n🛡️ 에러 핸들러 설정")
    print("-" * 50)

    # 에러 핸들러 생성
    error_handler = ErrorHandler()

    # 에러별 콜백 등록
    def on_rate_limit(error, context):
        logger.warning("Rate limit hit! Waiting before retry...")
        print(f"⚠️  속도 제한 발생: {context.context}")

    def on_auth_error(error, context):
        logger.error("Authentication failed! Check API key")
        print("🔐 인증 실패: API 키를 확인하세요")

    def on_any_error(error, context):
        logger.info(f"Error logged: {context.error_type}")

    error_handler.register_callback("RateLimitError", on_rate_limit)
    error_handler.register_callback("Exception", on_auth_error)  # 인증 에러는 Exception으로 처리
    error_handler.register_callback("*", on_any_error)

    # 테스트
    try:
        raise RateLimitError("Too many requests")
    except Exception as e:
        error_handler.handle_error(e)

    # 통계 출력
    stats = error_handler.get_error_stats()
    print("\n📊 에러 통계:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def example_retry_strategy():
    """재시도 전략 예제"""
    print("\n🔄 재시도 전략")
    print("-" * 50)

    error_handler = ErrorHandler()
    strategy = RetryStrategy(max_retries=3, initial_delay=1.0, exponential_base=2.0, jitter=True)

    @with_retry(strategy=strategy, error_handler=error_handler)
    def unreliable_api_call():
        """불안정한 API 호출 시뮬레이션"""
        import random

        if random.random() < 0.7:  # 70% 실패율
            raise Exception("API call failed")
        return "Success!"

    try:
        result = unreliable_api_call()
        print(f"✅ 성공: {result}")
    except Exception as e:
        print(f"❌ 최종 실패: {e}")

    # 에러 통계
    stats = error_handler.get_error_stats()
    print(f"\n재시도 통계: 총 {stats['total']}회 시도")


def example_circuit_breaker():
    """서킷 브레이커 예제"""
    print("\n⚡ 서킷 브레이커")
    print("-" * 50)

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

    @breaker
    def flaky_service():
        """불안정한 서비스"""
        import random

        if random.random() < 0.8:  # 80% 실패율
            raise Exception("Service unavailable")
        return "Service response"

    # 여러 번 호출 시도
    for i in range(10):
        try:
            result = flaky_service()
            print(f"✅ 호출 {i+1}: {result}")
        except Exception as e:
            print(f"❌ 호출 {i+1}: {e}")

        if i == 5:
            print("\n⏳ 5초 대기 (회복 시간)...")
            time.sleep(5)


def example_rate_limiter():
    """속도 제한기 예제"""
    print("\n⏱️ 속도 제한기")
    print("-" * 50)

    # 5초에 3번까지만 허용
    limiter = RateLimiter(max_calls=3, time_window=5.0)

    @limiter
    def api_call(index):
        return f"API 호출 {index} 성공"

    # 빠른 연속 호출
    for i in range(5):
        try:
            result = api_call(i + 1)
            print(f"✅ {result}")
            time.sleep(1)
        except RateLimitError as e:
            print(f"⚠️  {e}")


def example_fallback_system():
    """폴백 시스템 예제"""
    print("\n🔀 폴백 LLM 시스템")
    print("-" * 50)

    # 여러 모델로 폴백 시스템 구성
    fallback_llm = FallbackLLM(models=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])

    try:
        # 정상 호출
        result = fallback_llm.ask("파이썬의 장점을 한 문장으로 설명하세요.")
        print(f"✅ 응답: {result.text}")

        # 에러 통계
        stats = fallback_llm.error_handler.get_error_stats()
        print(
            f"\n📊 폴백 통계: {stats['total']}회 에러, 현재 모델: {fallback_llm.models[fallback_llm.current_model_index]}"
        )

    except Exception as e:
        print(f"❌ 모든 모델 실패: {e}")


async def example_async_error_handling():
    """비동기 에러 처리 예제"""
    print("\n🔀 비동기 에러 처리")
    print("-" * 50)

    error_handler = ErrorHandler()

    @with_retry(error_handler=error_handler)
    async def async_llm_call(prompt: str):
        """비동기 LLM 호출"""
        llm = LLM.create("gpt-4o-mini")

        # 에러 시뮬레이션
        import random

        if random.random() < 0.3:
            raise Exception("Async operation failed")

        return await llm.ask_async(prompt)

    # 여러 비동기 호출
    prompts = ["AI의 미래는?", "기후 변화 해결책은?", "우주 탐사의 중요성은?"]

    tasks = [async_llm_call(prompt) for prompt in prompts]

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ 작업 {i+1} 실패: {result}")
            else:
                print(f"✅ 작업 {i+1} 성공: {result.text[:50]}...")

    except Exception as e:
        print(f"❌ 치명적 오류: {e}")


def example_production_setup():
    """프로덕션 환경 설정 예제"""
    print("\n🏭 프로덕션 환경 설정")
    print("-" * 50)

    class ProductionLLM:
        """프로덕션용 LLM 래퍼"""

        def __init__(self):
            # 에러 핸들러
            self.error_handler = ErrorHandler()

            # 재시도 전략
            self.retry_strategy = RetryStrategy(max_retries=3, initial_delay=1.0, max_delay=30.0)

            # 서킷 브레이커
            self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

            # 속도 제한기 (분당 60회)
            self.rate_limiter = RateLimiter(max_calls=60, time_window=60.0)

            # 폴백 모델
            self.fallback_llm = FallbackLLM(models=["gpt-4o-mini", "gpt-3.5-turbo"], error_handler=self.error_handler)

            # 메트릭 수집
            self.metrics = {"total_calls": 0, "successful_calls": 0, "failed_calls": 0, "total_tokens": 0}

        @with_retry()
        def ask(self, prompt: str, **kwargs):
            """프로덕션 ask 메서드"""
            self.metrics["total_calls"] += 1

            try:
                # 속도 제한 체크
                self.rate_limiter(lambda: None)()

                # 서킷 브레이커 체크
                @self.circuit_breaker
                def _ask():
                    return self.fallback_llm.ask(prompt, **kwargs)

                result = _ask()

                # 성공 메트릭
                self.metrics["successful_calls"] += 1
                if hasattr(result, "usage") and result.usage:
                    self.metrics["total_tokens"] += result.usage.total

                return result

            except Exception as e:
                self.metrics["failed_calls"] += 1
                self.error_handler.handle_error(e, {"prompt": prompt[:100], "kwargs": str(kwargs)})
                raise

        def get_health_status(self) -> Dict[str, Any]:
            """시스템 상태 확인"""
            error_stats = self.error_handler.get_error_stats()

            return {
                "status": "healthy" if error_stats["total"] < 10 else "degraded",
                "metrics": self.metrics,
                "error_stats": error_stats,
                "circuit_breaker": self.circuit_breaker.state,
                "current_model": self.fallback_llm.models[self.fallback_llm.current_model_index],
            }

    # 프로덕션 LLM 사용
    prod_llm = ProductionLLM()

    # 여러 요청 시뮬레이션
    requests = ["AI란 무엇인가?", "기계학습 설명", "딥러닝이란?", "자연어 처리란?", "컴퓨터 비전이란?"]

    for req in requests:
        try:
            result = prod_llm.ask(req)
            print(f"✅ '{req}' → {result.text[:50]}...")
        except Exception as e:
            print(f"❌ '{req}' → 실패: {e}")

    # 상태 확인
    health = prod_llm.get_health_status()
    print("\n🏥 시스템 상태:")
    print(json.dumps(health, indent=2, ensure_ascii=False))


def main():
    """고급 에러 처리 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("🛡️ 고급 에러 처리 예제")
    print("=" * 50)

    try:
        # 1. 에러 핸들러 설정
        example_error_handler_setup()

        # 2. 재시도 전략
        example_retry_strategy()

        # 3. 서킷 브레이커
        example_circuit_breaker()

        # 4. 속도 제한기
        example_rate_limiter()

        # 5. 폴백 시스템
        example_fallback_system()

        # 6. 비동기 에러 처리
        asyncio.run(example_async_error_handling())

        # 7. 프로덕션 설정
        example_production_setup()

        print("\n✅ 모든 고급 에러 처리 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
