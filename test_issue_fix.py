#!/usr/bin/env python3
"""
Issue #22 수정 확인 테스트
"""

import os
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

# 현재 디렉토리의 src를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pyhub.llm import OpenAILLM

# 환경변수 확인
if not os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY 환경 변수를 설정해주세요.")
    sys.exit(1)

# 테스트
llm = OpenAILLM(system_prompt="You are a helpful assistant.")

print("테스트 1: 슬래시가 포함된 choices")
reply = llm.ask(
    "다음 고객 문의의 의도를 분류 : 이 제품 환불하고 싶은데요.", 
    use_history=False, 
    choices=["환불/반품", "배송문의", "사용방법", "가격문의", "A/S요청", "제품정보", "구매상담", "기타"]
)

print(f"응답 텍스트: {reply.text}")
print(f"선택된 choice: {reply.choice}")
print(f"choice 인덱스: {reply.choice_index}")
print(f"신뢰도: {reply.confidence}")

print("\n테스트 2: 특수문자가 없는 choices") 
reply2 = llm.ask(
    "다음 고객 문의의 의도를 분류 : 이 제품 환불하고 싶은데요.",
    use_history=False,
    choices=["환불반품", "배송문의", "사용방법", "가격문의", "AS요청", "제품정보", "구매상담", "기타"]
)

print(f"응답 텍스트: {reply2.text}")
print(f"선택된 choice: {reply2.choice}")
print(f"choice 인덱스: {reply2.choice_index}")
print(f"신뢰도: {reply2.confidence}")