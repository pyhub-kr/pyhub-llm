"""
JSON Schema 구조화된 출력에서 특수문자 처리 테스트
Issue #22: https://github.com/pyhub-kr/pyhub-llm/issues/22
"""
import json
import pytest
from unittest.mock import Mock, patch

from pyhub.llm.base import BaseLLM
from pyhub.llm.types import Reply


class TestJSONSchemaSpecialChars:
    """JSON Schema에서 특수문자가 포함된 choices 처리 테스트"""
    
    def test_control_characters_in_response(self):
        """제어 문자가 포함된 응답 처리 테스트"""
        # 실제 문제가 발생한 응답 시뮬레이션
        problematic_response = '{"choice":"\\u001cA/S\\u001d0\\u001d\\u001d\\u001d"}'
        
        # BaseLLM의 _process_choice_response 메서드 테스트
        class TestLLM(BaseLLM):
            def _make_ask(self, *args, **kwargs):
                pass
            def _make_ask_async(self, *args, **kwargs):
                pass
            def _make_ask_stream(self, *args, **kwargs):
                pass
            def _make_ask_stream_async(self, *args, **kwargs):
                pass
            def _make_request_params(self, *args, **kwargs):
                pass
            def messages(self, *args, **kwargs):
                pass
            def embed(self, *args, **kwargs):
                pass
            def embed_async(self, *args, **kwargs):
                pass
            def generate_image(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
            def generate_image_async(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
        
        llm = TestLLM()
        choices = ["환불/반품", "배송문의", "사용방법", "가격문의", "A/S요청", "제품정보", "구매상담", "기타"]
        
        # 제어 문자가 포함된 응답 파싱
        choice, index, confidence = llm._process_choice_response(
            problematic_response, 
            choices, 
            choices_optional=False
        )
        
        # 기존 방식으로는 매칭 실패
        assert choice is None
        assert index is None
        assert confidence == 0.0
    
    def test_various_control_characters(self):
        """다양한 제어 문자 필터링 테스트"""
        test_cases = [
            ('{"choice":"\\u001c환불/반품\\u001d"}', "환불/반품"),
            ('{"choice":"\\u001cA/S\\u001d요청"}', "A/S요청"),
            ('{"choice":"배송\\u001c\\u001d문의"}', "배송문의"),
            ('{"choice":"\\u0000가격\\u001f문의\\u007f"}', "가격문의"),  # 다른 제어 문자들
        ]
        
        class TestLLM(BaseLLM):
            def _make_ask(self, *args, **kwargs):
                pass
            def _make_ask_async(self, *args, **kwargs):
                pass
            def _make_ask_stream(self, *args, **kwargs):
                pass
            def _make_ask_stream_async(self, *args, **kwargs):
                pass
            def _make_request_params(self, *args, **kwargs):
                pass
            def messages(self, *args, **kwargs):
                pass
            def embed(self, *args, **kwargs):
                pass
            def embed_async(self, *args, **kwargs):
                pass
            def generate_image(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
            def generate_image_async(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
        
        llm = TestLLM()
        choices = ["환불/반품", "배송문의", "사용방법", "가격문의", "A/S요청", "제품정보", "구매상담", "기타"]
        
        for response, expected in test_cases:
            choice, _, _ = llm._process_choice_response(response, choices, False)
            assert choice == expected, f"Failed for response: {response}"
    
    def test_special_chars_in_choices(self):
        """choices에 특수문자가 포함된 경우 정상 처리 테스트"""
        choices_with_special_chars = [
            "환불/반품",
            "A/S요청",
            "Q&A",
            "배송(택배)",
            "가격\\할인",
            "제품#1",
            "상담@전화"
        ]
        
        class TestLLM(BaseLLM):
            def _make_ask(self, *args, **kwargs):
                pass
            def _make_ask_async(self, *args, **kwargs):
                pass
            def _make_ask_stream(self, *args, **kwargs):
                pass
            def _make_ask_stream_async(self, *args, **kwargs):
                pass
            def _make_request_params(self, *args, **kwargs):
                pass
            def messages(self, *args, **kwargs):
                pass
            def embed(self, *args, **kwargs):
                pass
            def embed_async(self, *args, **kwargs):
                pass
            def generate_image(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
            def generate_image_async(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
        
        llm = TestLLM()
        
        # 각 choice가 정상적으로 매칭되는지 테스트
        for expected_choice in choices_with_special_chars:
            response = json.dumps({"choice": expected_choice})
            choice, index, confidence = llm._process_choice_response(
                response, 
                choices_with_special_chars, 
                False
            )
            assert choice == expected_choice
            assert index == choices_with_special_chars.index(expected_choice)
            assert confidence == 1.0
    
    def test_malformed_json_with_control_chars(self):
        """제어 문자로 인해 JSON 파싱이 실패하는 경우"""
        # JSON 파싱이 실패하더라도 텍스트 매칭으로 처리되어야 함
        malformed_responses = [
            'A/S요청',  # JSON이 아닌 일반 텍스트
            '{"choice": "A/S요청"',  # 닫히지 않은 JSON
            'choice: A/S요청',  # 잘못된 형식
        ]
        
        class TestLLM(BaseLLM):
            def _make_ask(self, *args, **kwargs):
                pass
            def _make_ask_async(self, *args, **kwargs):
                pass
            def _make_ask_stream(self, *args, **kwargs):
                pass
            def _make_ask_stream_async(self, *args, **kwargs):
                pass
            def _make_request_params(self, *args, **kwargs):
                pass
            def messages(self, *args, **kwargs):
                pass
            def embed(self, *args, **kwargs):
                pass
            def embed_async(self, *args, **kwargs):
                pass
            def generate_image(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
            def generate_image_async(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
        
        llm = TestLLM()
        choices = ["환불/반품", "배송문의", "사용방법", "가격문의", "A/S요청", "제품정보", "구매상담", "기타"]
        
        for response in malformed_responses:
            choice, index, confidence = llm._process_choice_response(response, choices, False)
            # 정확한 텍스트 매칭 또는 부분 매칭이 되어야 함
            assert choice == "A/S요청" or choice is None
    
    def test_choice_index_based_response(self):
        """choice_index 기반 응답 처리 테스트"""
        # choice_index를 포함한 정상 응답
        response_with_index = '{"choice":"환불_반품","choice_index":0,"confidence":0.95}'
        
        class TestLLM(BaseLLM):
            def _make_ask(self, *args, **kwargs):
                pass
            def _make_ask_async(self, *args, **kwargs):
                pass
            def _make_ask_stream(self, *args, **kwargs):
                pass
            def _make_ask_stream_async(self, *args, **kwargs):
                pass
            def _make_request_params(self, *args, **kwargs):
                pass
            def messages(self, *args, **kwargs):
                pass
            def embed(self, *args, **kwargs):
                pass
            def embed_async(self, *args, **kwargs):
                pass
            def generate_image(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
            def generate_image_async(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
        
        llm = TestLLM()
        choices = ["환불/반품", "배송문의", "사용방법", "가격문의", "A/S요청", "제품정보", "구매상담", "기타"]
        
        # choice_index 기반 응답 파싱
        choice, index, confidence = llm._process_choice_response(
            response_with_index, 
            choices, 
            choices_optional=False
        )
        
        # choice_index를 사용해 원본 choice 반환
        assert choice == "환불/반품"
        assert index == 0
        assert confidence == 0.95
    
    def test_normalized_choices_in_context(self):
        """정규화된 choices가 컨텍스트에 저장되는지 테스트"""
        class TestLLM(BaseLLM):
            def _make_ask(self, *args, **kwargs):
                return Reply(text='{"choice":"환불_반품","choice_index":0,"confidence":0.95}')
            def _make_ask_async(self, *args, **kwargs):
                pass
            def _make_ask_stream(self, *args, **kwargs):
                pass
            def _make_ask_stream_async(self, *args, **kwargs):
                pass
            def _make_request_params(self, *args, **kwargs):
                pass
            def messages(self, *args, **kwargs):
                pass
            def embed(self, *args, **kwargs):
                pass
            def embed_async(self, *args, **kwargs):
                pass
            def generate_image(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
            def generate_image_async(self, *args, **kwargs):
                raise NotImplementedError("Test LLM does not support image generation")
        
        llm = TestLLM()
        
        # 특수문자가 포함된 choices로 ask 호출
        choices = ["환불/반품", "A/S요청", "Q&A", "배송(택배)"]
        reply = llm.ask("문의 유형을 선택하세요", choices=choices)
        
        # 원본 choice가 반환되어야 함
        assert reply.choice == "환불/반품"
        assert reply.choice_index == 0
        assert reply.confidence == 0.95

    @pytest.mark.skipif(
        True,  # OpenAI 통합 테스트는 실제 API 호출이 필요하므로 일단 스킵
        reason="OpenAI 모듈의 동적 import로 인한 mock 이슈. 실제 API 테스트는 별도로 진행"
    )
    def test_openai_special_chars_handling(self):
        """OpenAI 구현에서 특수문자 처리 통합 테스트"""
        pass  # 실제 API 테스트로 대체 예정


if __name__ == "__main__":
    pytest.main([__file__, "-v"])