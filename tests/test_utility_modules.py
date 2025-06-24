"""
유틸리티 모듈 테스트

현재 0% 커버리지인 유틸리티 모듈들을 테스트합니다.
- utils/mixins.py
- utils/pricing.py
- utils/retry.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time
from datetime import datetime

from pyhub.llm.types import Usage


class TestPricingUtils:
    """가격 계산 유틸리티 테스트"""
    
    def test_calculate_cost(self):
        """토큰 비용 계산 테스트"""
        from pyhub.llm.utils.pricing import calculate_cost
        
        # GPT-4o 가격 테스트
        result = calculate_cost(
            model="gpt-4o",
            input_tokens=1_000_000,  # 1M tokens
            output_tokens=500_000    # 0.5M tokens
        )
        
        assert result["model"] == "gpt-4o"
        assert result["input_tokens"] == 1_000_000
        assert result["output_tokens"] == 500_000
        assert result["input_cost"] == pytest.approx(2.50, rel=1e-3)  # $2.50 per 1M
        assert result["output_cost"] == pytest.approx(5.00, rel=1e-3)  # $10.00 per 1M * 0.5
        assert result["total_cost"] == pytest.approx(7.50, rel=1e-3)
    
    def test_calculate_cost_unknown_model(self):
        """알 수 없는 모델의 비용 계산 테스트"""
        from pyhub.llm.utils.pricing import calculate_cost
        
        # 알 수 없는 모델은 gpt-4o-mini 가격으로 추정
        result = calculate_cost(
            model="unknown-model",
            input_tokens=100_000,
            output_tokens=50_000
        )
        
        # gpt-4o-mini 가격: input $0.150/1M, output $0.600/1M
        assert result["input_cost"] == pytest.approx(0.015, rel=1e-3)
        assert result["output_cost"] == pytest.approx(0.030, rel=1e-3)
        assert result["total_cost"] == pytest.approx(0.045, rel=1e-3)
    
    def test_pricing_data_structure(self):
        """가격 데이터 구조 테스트"""
        from pyhub.llm.utils.pricing import PRICING
        
        # 주요 모델들이 포함되어 있는지 확인
        assert "gpt-4o" in PRICING
        assert "gpt-4o-mini" in PRICING
        assert "claude-3-5-sonnet-latest" in PRICING
        assert "gemini-1.5-pro" in PRICING
        
        # 각 모델에 input/output 가격이 있는지 확인
        for model, prices in PRICING.items():
            assert "input" in prices
            assert "output" in prices
            assert prices["input"] >= 0
            assert prices["output"] >= 0


class TestRetryUtils:
    """재시도 유틸리티 테스트"""
    
    def test_exponential_backoff_decorator(self):
        """지수 백오프 데코레이터 테스트"""
        from pyhub.llm.utils.retry import exponential_backoff
        
        call_count = 0
        
        @exponential_backoff(max_retries=2, initial_delay=0.01, verbose=False)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "Success"
        
        result = flaky_function()
        assert result == "Success"
        assert call_count == 3
    
    def test_retry_with_fallback(self):
        """폴백 함수 재시도 테스트"""
        from pyhub.llm.utils.retry import retry_with_fallback
        
        primary_calls = 0
        fallback_calls = 0
        
        def primary_func():
            nonlocal primary_calls
            primary_calls += 1
            raise Exception("Primary failed")
        
        def fallback_func():
            nonlocal fallback_calls
            fallback_calls += 1
            return "Fallback success"
        
        result = retry_with_fallback(
            primary_func=primary_func,
            fallback_func=fallback_func,
            max_retries=1,
            verbose=False
        )
        
        assert result == "Fallback success"
        assert primary_calls >= 1
        assert fallback_calls == 1
    
    def test_retry_specific_exceptions(self):
        """특정 예외만 재시도 테스트"""
        from pyhub.llm.utils.retry import exponential_backoff
        
        call_count = 0
        
        @exponential_backoff(
            max_retries=3,
            exceptions=(ValueError,),
            initial_delay=0.01,
            verbose=False
        )
        def selective_retry():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retry this")
            elif call_count == 2:
                raise TypeError("Don't retry this")
            return "Should not reach"
        
        with pytest.raises(TypeError):
            selective_retry()
        
        assert call_count == 2  # ValueError는 재시도, TypeError는 즉시 실패
    
    def test_handle_api_error(self):
        """API 에러 처리 테스트"""
        from pyhub.llm.utils.retry import (
            handle_api_error,
            RateLimitError,
            NetworkError,
            ServerError,
            AuthenticationError
        )
        
        # Test error pattern matching
        # Rate limit patterns
        with pytest.raises(RateLimitError):
            handle_api_error(Exception("rate limit exceeded"))
        
        with pytest.raises(RateLimitError):
            handle_api_error(Exception("quota exceeded"))
        
        # Network error patterns
        with pytest.raises(NetworkError):
            handle_api_error(Exception("connection refused"))
        
        with pytest.raises(NetworkError):
            handle_api_error(Exception("dns error"))
        
        # Server error patterns
        with pytest.raises(ServerError):
            handle_api_error(Exception("502 Bad Gateway"))
        
        with pytest.raises(ServerError):
            handle_api_error(Exception("internal error"))
        
        # Authentication error patterns
        with pytest.raises(AuthenticationError):
            handle_api_error(Exception("403 Forbidden"))
        
        with pytest.raises(AuthenticationError):
            handle_api_error(Exception("api key invalid"))
        
        # 기타 에러는 그대로 전파 - RuntimeError가 발생함 (raise without exception)
        with pytest.raises(RuntimeError):
            handle_api_error(ValueError("Unknown error"))
    
    def test_retry_api_call_decorator(self):
        """API 호출 재시도 데코레이터 테스트"""
        from pyhub.llm.utils.retry import retry_api_call, RateLimitError
        
        call_count = 0
        
        # Test with non-API error (should not be retried by this decorator)
        @retry_api_call(verbose=False)
        def normal_function():
            nonlocal call_count
            call_count += 1
            return "Success"
        
        result = normal_function()
        assert result == "Success"
        assert call_count == 1
        
        # Test that the decorator can handle general exceptions
        call_count = 0
        
        @retry_api_call(verbose=False)
        def exception_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("General error")
        
        with pytest.raises(ValueError):
            exception_function()
    
    def test_custom_exceptions(self):
        """커스톰 예외 클래스 테스트"""
        from pyhub.llm.utils.retry import (
            APIError,
            RateLimitError,
            NetworkError,
            ServerError,
            AuthenticationError,
            RetryError
        )
        
        # Exception hierarchy 테스트
        assert issubclass(RateLimitError, APIError)
        assert issubclass(NetworkError, APIError)
        assert issubclass(ServerError, APIError)
        assert issubclass(AuthenticationError, APIError)
        
        # Exception 인스턴스 생성 테스트
        rate_limit_err = RateLimitError("Rate limit exceeded")
        assert str(rate_limit_err) == "Rate limit exceeded"
        
        retry_err = RetryError("Max retries exceeded")
        assert str(retry_err) == "Max retries exceeded"


class TestMixins:
    """Mixin 클래스 테스트"""
    
    def test_retry_mixin(self):
        """재시도 믹스인 테스트"""
        from pyhub.llm.utils.mixins import RetryMixin
        
        # Test basic functionality (mixin initialization)
        llm_retry = RetryMixin(enable_retry=True, retry_verbose=False)
        assert llm_retry.enable_retry is True
        assert llm_retry.retry_verbose is False
        
        llm_no_retry = RetryMixin(enable_retry=False)
        assert llm_no_retry.enable_retry is False
        
        # Test _wrap_with_retry method
        def test_func():
            return "test_result"
        
        # With retry disabled, function should remain unchanged
        wrapped_no_retry = llm_no_retry._wrap_with_retry(test_func)
        assert wrapped_no_retry() == "test_result"
        
        # With retry enabled, function should be wrapped (but won't retry without API errors)
        wrapped_retry = llm_retry._wrap_with_retry(test_func)
        assert wrapped_retry() == "test_result"
    
    def test_validation_mixin(self):
        """검증 믹스인 테스트"""
        from pyhub.llm.utils.mixins import ValidationMixin
        from unittest.mock import patch
        
        # Create a simple mixin instance for testing
        class MockValidationMixin(ValidationMixin):
            def __init__(self, model="gpt-4o"):
                self.model = model
                self.max_tokens = 1000
                super().__init__()
        
        llm = MockValidationMixin()
        
        # Test model token limits are set
        assert llm.model_token_limits["gpt-4o"] == 128000
        assert llm.model_token_limits["claude-3-5-sonnet-latest"] == 200000
        assert llm.model_token_limits["gemini-1.5-pro"] == 1000000
        
        # Test token validation method
        short_text = "Hello"
        with patch("pyhub.llm.utils.mixins.console.print") as mock_print:
            llm._validate_token_limit(short_text)
            mock_print.assert_not_called()
        
        # Test with very long text that exceeds token limit
        long_text = "a" * 600000  # About 150k tokens
        with patch("pyhub.llm.utils.mixins.console.print") as mock_print:
            llm._validate_token_limit(long_text)
            mock_print.assert_called_once()
            warning_msg = mock_print.call_args[0][0]
            assert "경고" in warning_msg
            assert "토큰" in warning_msg
        
        # Test embed validation for long text (exceeds 8191 tokens)
        embed_long_text = "a" * 40000  # About 10k tokens, exceeds 8191 limit
        with patch("pyhub.llm.utils.mixins.console.print") as mock_print:
            # Mock the super().embed to avoid AttributeError
            with patch('builtins.super') as mock_super:
                mock_super.return_value.embed.return_value = "embedding"
                result = llm.embed(embed_long_text)
                assert result == "embedding"
                mock_print.assert_called_once()
                warning_msg = mock_print.call_args[0][0]
                assert "경고" in warning_msg
                assert "임베딩" in warning_msg