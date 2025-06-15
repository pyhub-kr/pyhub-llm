#!/usr/bin/env python3
"""
예제: 스트리밍 응답
난이도: 초급
설명: 실시간으로 응답을 받아 처리하는 방법
요구사항: OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import os
import sys

from pyhub.llm import LLM


def main():
    """스트리밍 응답 예제"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("🌊 스트리밍 응답 예제\n")

    # LLM 생성
    llm = LLM.create("gpt-4o-mini")

    # 스트리밍으로 이야기 생성
    prompt = "파이썬의 장점을 3가지 설명해주세요."
    print(f"프롬프트: {prompt}\n")
    print("응답 (스트리밍):\n" + "-" * 50)

    # 스트리밍 응답 받기
    full_response = ""
    for chunk in llm.ask(prompt, stream=True):
        # 실시간으로 출력
        print(chunk.text, end="", flush=True)
        full_response += chunk.text

    print("\n" + "-" * 50)
    print(f"\n✅ 전체 응답 길이: {len(full_response)}자")

    # 스트리밍 with 진행 표시
    print("\n\n🎯 진행 표시와 함께 스트리밍:")
    prompt2 = "짧은 시를 하나 작성해주세요."
    print(f"프롬프트: {prompt2}\n")

    char_count = 0
    for chunk in llm.ask(prompt2, stream=True):
        print(chunk.text, end="", flush=True)
        char_count += len(chunk.text)

        # 진행 상태를 터미널 제목에 표시 (선택사항)
        sys.stdout.write(f"\033]0;생성 중... ({char_count}자)\007")

    print(f"\n\n✅ 완료! 총 {char_count}자 생성")


if __name__ == "__main__":
    main()
