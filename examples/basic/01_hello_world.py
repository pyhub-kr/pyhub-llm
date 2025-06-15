#!/usr/bin/env python3
"""
예제: pyhub-llm으로 Hello World
난이도: 초급
설명: pyhub-llm을 사용한 첫 번째 프로그램
요구사항:
  - pyhub-llm (pip install "pyhub-llm[all]")
  - OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import os

from pyhub.llm import LLM


def main():
    """기본 LLM 사용 예제"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        print("예: export OPENAI_API_KEY='your-api-key'")
        return

    print("🚀 pyhub-llm Hello World 예제\n")

    # LLM 생성
    llm = LLM.create("gpt-4o-mini")
    print("✅ LLM 인스턴스 생성 완료\n")

    # 간단한 질문
    question = "안녕하세요! 당신은 누구인가요?"
    print(f"질문: {question}")

    # 응답 받기
    reply = llm.ask(question)
    print(f"응답: {reply.text}")

    # 사용 통계 출력 (있는 경우)
    if reply.usage:
        print("\n📊 토큰 사용량:")
        print(f"  - 입력: {reply.usage.input}")
        print(f"  - 출력: {reply.usage.output}")
        print(f"  - 총합: {reply.usage.total}")


if __name__ == "__main__":
    main()
