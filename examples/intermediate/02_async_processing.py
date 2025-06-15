#!/usr/bin/env python3
"""
예제: 비동기 처리
난이도: 중급
설명: 비동기로 여러 작업을 동시에 처리하는 방법
요구사항: OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import asyncio
import os
import sys
import time

from pyhub.llm import LLM


async def async_question(llm, question: str, index: int) -> tuple:
    """비동기로 질문 처리"""
    start_time = time.time()
    print(f"🚀 작업 {index} 시작: {question[:30]}...")

    # 비동기 ask 호출
    reply = await llm.ask_async(question)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"✅ 작업 {index} 완료 ({elapsed:.2f}초)")
    return index, question, reply.text, elapsed


async def parallel_processing():
    """병렬 처리 예제"""
    print("\n⚡ 병렬 처리 예제")
    print("-" * 50)

    # LLM 인스턴스 생성
    llm = LLM.create("gpt-4o-mini")

    # 여러 질문 준비
    questions = [
        "파이썬의 장점을 3가지 설명해주세요.",
        "머신러닝과 딥러닝의 차이점은 무엇인가요?",
        "REST API의 주요 특징을 설명해주세요.",
        "Git과 GitHub의 차이점은 무엇인가요?",
        "Docker의 장점을 설명해주세요.",
    ]

    print(f"총 {len(questions)}개의 질문을 병렬로 처리합니다.\n")

    # 순차 처리 시간 측정
    print("1️⃣ 순차 처리 테스트")
    sequential_start = time.time()

    for i, question in enumerate(questions, 1):
        _ = llm.ask(question)
        print(f"  - 질문 {i} 완료")

    sequential_time = time.time() - sequential_start
    print(f"순차 처리 총 시간: {sequential_time:.2f}초\n")

    # 병렬 처리 시간 측정
    print("2️⃣ 병렬 처리 테스트")
    parallel_start = time.time()

    # 모든 작업을 동시에 실행
    tasks = [async_question(llm, question, i) for i, question in enumerate(questions, 1)]

    results = await asyncio.gather(*tasks)

    parallel_time = time.time() - parallel_start
    print(f"\n병렬 처리 총 시간: {parallel_time:.2f}초")
    print(f"🎯 속도 향상: {sequential_time/parallel_time:.2f}배 빠름!")

    # 결과 출력
    print("\n📊 처리 결과:")
    for index, question, answer, elapsed in results:
        print(f"\nQ{index}: {question}")
        print(f"A: {answer[:100]}...")
        print(f"⏱️  소요 시간: {elapsed:.2f}초")


async def streaming_async():
    """비동기 스트리밍 예제"""
    print("\n🌊 비동기 스트리밍 예제")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")
    prompt = "비동기 프로그래밍의 장점을 자세히 설명해주세요."

    print(f"프롬프트: {prompt}\n")
    print("스트리밍 응답:")
    print("-" * 30)

    # 비동기 스트리밍
    full_response = ""
    async for chunk in llm.ask_async(prompt, stream=True):
        print(chunk.text, end="", flush=True)
        full_response += chunk.text

    print("\n" + "-" * 30)
    print(f"\n총 {len(full_response)}자 생성됨")


async def concurrent_conversations():
    """동시 대화 처리 예제"""
    print("\n💬 동시 대화 처리 예제")
    print("-" * 50)

    # 여러 LLM 인스턴스로 다른 페르소나 생성
    poet_llm = LLM.create("gpt-4o-mini", system_prompt="당신은 감성적인 시인입니다.")
    scientist_llm = LLM.create("gpt-4o-mini", system_prompt="당신은 논리적인 과학자입니다.")
    philosopher_llm = LLM.create("gpt-4o-mini", system_prompt="당신은 깊이 있는 철학자입니다.")

    topic = "인공지능의 미래"
    print(f"주제: {topic}\n")

    # 동시에 세 관점에서 답변 생성
    tasks = [
        poet_llm.ask_async(f"{topic}에 대해 시적으로 표현해주세요."),
        scientist_llm.ask_async(f"{topic}에 대해 과학적으로 분석해주세요."),
        philosopher_llm.ask_async(f"{topic}에 대해 철학적으로 고찰해주세요."),
    ]

    print("세 가지 관점에서 동시에 답변 생성 중...\n")

    poet_reply, scientist_reply, philosopher_reply = await asyncio.gather(*tasks)

    print("🎭 시인의 관점:")
    print(poet_reply.text)
    print("\n🔬 과학자의 관점:")
    print(scientist_reply.text)
    print("\n🤔 철학자의 관점:")
    print(philosopher_reply.text)


async def error_handling_async():
    """비동기 에러 처리 예제"""
    print("\n⚠️  비동기 에러 처리 예제")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")

    # 여러 작업 중 일부가 실패할 수 있는 상황
    tasks = [
        llm.ask_async("정상적인 질문입니다."),
        llm.ask_async("이것도 정상적인 질문입니다."),
        # 의도적으로 매우 긴 프롬프트로 에러 유발 가능성
        llm.ask_async("A" * 100000),  # 토큰 제한 초과 가능
    ]

    # gather with return_exceptions=True
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"❌ 작업 {i} 실패: {type(result).__name__}: {str(result)[:100]}")
        else:
            print(f"✅ 작업 {i} 성공: {result.text[:50]}...")


async def main():
    """비동기 처리 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("🔄 비동기 처리 예제")
    print("=" * 50)

    try:
        # 1. 병렬 처리 예제
        await parallel_processing()

        # 2. 비동기 스트리밍 예제
        await streaming_async()

        # 3. 동시 대화 처리 예제
        await concurrent_conversations()

        # 4. 에러 처리 예제
        await error_handling_async()

        print("\n✅ 모든 비동기 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    # 비동기 메인 함수 실행
    asyncio.run(main())
