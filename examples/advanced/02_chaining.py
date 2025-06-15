#!/usr/bin/env python3
"""
예제: 체이닝
난이도: 고급
설명: 여러 LLM을 연결하여 복잡한 작업 수행
요구사항:
  - pyhub-llm (pip install "pyhub-llm[all]")
  - OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

from pyhub.llm import LLM


@dataclass
class ChainResult:
    """체인 실행 결과"""

    final_output: str
    intermediate_results: List[Dict[str, Any]]
    total_tokens: int = 0
    execution_time: float = 0.0


class SimpleChain:
    """간단한 체인 구현"""

    def __init__(self, llms: List[LLM]):
        self.llms = llms
        self.results = []

    def run(self, initial_input: str) -> ChainResult:
        """체인 실행"""
        import time

        start_time = time.time()

        current_input = initial_input
        intermediate_results = []
        total_tokens = 0

        for i, llm in enumerate(self.llms):
            print(f"\n🔗 체인 단계 {i+1}/{len(self.llms)}")

            # LLM 호출
            reply = llm.ask(current_input)

            # 결과 저장
            result = {
                "step": i + 1,
                "input": current_input[:100] + "..." if len(current_input) > 100 else current_input,
                "output": reply.text,
                "tokens": reply.usage.total if reply.usage else 0,
            }
            intermediate_results.append(result)

            # 다음 단계 입력으로 사용
            current_input = reply.text
            total_tokens += result["tokens"]

            print(f"   입력: {result['input']}")
            print(f"   출력: {result['output'][:100]}...")

        execution_time = time.time() - start_time

        return ChainResult(
            final_output=current_input,
            intermediate_results=intermediate_results,
            total_tokens=total_tokens,
            execution_time=execution_time,
        )


def example_basic_chaining():
    """기본 체이닝 예제"""
    print("\n🔗 기본 체이닝")
    print("-" * 50)

    # 여러 LLM을 연결
    chain = SimpleChain(
        [
            LLM.create("gpt-4o-mini", system_prompt="한국어를 영어로 번역하세요. 번역문만 출력하세요."),
            LLM.create("gpt-4o-mini", system_prompt="영어를 일본어로 번역하세요. 번역문만 출력하세요."),
            LLM.create("gpt-4o-mini", system_prompt="일본어를 다시 한국어로 번역하세요. 번역문만 출력하세요."),
        ]
    )

    # 체인 실행
    original_text = "인공지능은 인류의 미래를 바꿀 혁신적인 기술입니다."
    print(f"원본 텍스트: {original_text}")

    result = chain.run(original_text)

    print(f"\n최종 결과: {result.final_output}")
    print(f"총 토큰 사용: {result.total_tokens}")
    print(f"실행 시간: {result.execution_time:.2f}초")

    # 각 단계 결과
    print("\n📊 각 단계별 결과:")
    for step in result.intermediate_results:
        print(f"  단계 {step['step']}: {step['output']}")


def example_conditional_chaining():
    """조건부 체이닝 예제"""
    print("\n🔀 조건부 체이닝")
    print("-" * 50)

    class ConditionalChain:
        """조건에 따라 다른 체인을 실행"""

        def __init__(self):
            # 분류기
            self.classifier = LLM.create(
                "gpt-4o-mini", system_prompt="텍스트의 주제를 '기술', '비즈니스', '일반' 중 하나로]만 답하세요."
            )

            # 주제별 전문 LLM
            self.tech_expert = LLM.create(
                "gpt-4o-mini", system_prompt="당신은 기술 전문가입니다. 기술적이고 상세한 답변을 제공하세요."
            )

            self.business_expert = LLM.create(
                "gpt-4o-mini", system_prompt="당신은 비즈니스 전문가입니다. 비즈니스 관점에서 답변하세요."
            )

            self.general_expert = LLM.create(
                "gpt-4o-mini", system_prompt="당신은 친절한 어시스턴트입니다. 쉽고 친근하게 설명하세요."
            )

        def process(self, text: str) -> Dict[str, Any]:
            # 1. 주제 분류
            classification = self.classifier.ask(text, choices=["기술", "비즈니스", "일반"])
            topic = classification.choice

            print(f"📝 주제 분류: {topic}")

            # 2. 주제에 따른 처리
            if topic == "기술":
                expert = self.tech_expert
            elif topic == "비즈니스":
                expert = self.business_expert
            else:
                expert = self.general_expert

            # 3. 전문가 답변
            response = expert.ask(text)

            return {"topic": topic, "question": text, "answer": response.text, "expert_used": topic + "_expert"}

    # 조건부 체인 사용
    conditional_chain = ConditionalChain()

    questions = [
        "블록체인 기술의 작동 원리를 설명해주세요.",
        "스타트업의 자금 조달 방법에는 어떤 것들이 있나요?",
        "오늘 저녁 메뉴 추천해주세요.",
    ]

    for question in questions:
        print(f"\n💬 질문: {question}")
        result = conditional_chain.process(question)
        print(f"🏷️ 분류: {result['topic']}")
        print(f"👨‍💼 전문가: {result['expert_used']}")
        print(f"💡 답변: {result['answer'][:200]}...")


def example_parallel_chaining():
    """병렬 처리 체인 예제"""
    print("\n⚡ 병렬 처리 체인")
    print("-" * 50)

    async def parallel_analysis(text: str) -> Dict[str, Any]:
        """여러 분석을 병렬로 수행"""

        # 서로 다른 분석을 수행하는 LLM들
        sentiment_llm = LLM.create("gpt-4o-mini", system_prompt="텍스트의 감정을 분석하세요.")
        keyword_llm = LLM.create("gpt-4o-mini", system_prompt="핵심 키워드 3개를 추출하세요.")
        summary_llm = LLM.create("gpt-4o-mini", system_prompt="한 문장으로 요약하세요.")
        category_llm = LLM.create("gpt-4o-mini", system_prompt="적절한 카테고리를 제안하세요.")

        # 병렬 실행
        tasks = [
            sentiment_llm.ask_async(text),
            keyword_llm.ask_async(text),
            summary_llm.ask_async(text),
            category_llm.ask_async(text),
        ]

        import time

        start_time = time.time()

        results = await asyncio.gather(*tasks)

        execution_time = time.time() - start_time

        return {
            "original_text": text,
            "sentiment": results[0].text,
            "keywords": results[1].text,
            "summary": results[2].text,
            "category": results[3].text,
            "execution_time": execution_time,
        }

    # 분석할 텍스트
    text = """
    최근 인공지능 기술의 발전으로 많은 산업이 혁신적으로 변화하고 있습니다.
    특히 자연어 처리 분야에서 큰 발전이 있었고, 이는 고객 서비스, 콘텐츠 생성,
    번역 등 다양한 분야에 적용되고 있습니다. 하지만 동시에 일자리 대체, 
    개인정보 보호, 윤리적 문제 등 해결해야 할 과제들도 있습니다.
    """

    print(f"분석할 텍스트: {text[:100]}...")

    # 동기 실행을 위한 래퍼
    result = asyncio.run(parallel_analysis(text))

    print("\n📊 병렬 분석 결과:")
    print(f"  😊 감정: {result['sentiment']}")
    print(f"  🔑 키워드: {result['keywords']}")
    print(f"  📝 요약: {result['summary']}")
    print(f"  📁 카테고리: {result['category']}")
    print(f"  ⏱️  실행 시간: {result['execution_time']:.2f}초")


def example_complex_pipeline():
    """복잡한 파이프라인 예제"""
    print("\n🏭 복잡한 처리 파이프라인")
    print("-" * 50)

    class DataProcessingPipeline:
        """데이터 처리 파이프라인"""

        def __init__(self):
            # 각 단계별 LLM
            self.extractor = LLM.create(
                "gpt-4o-mini", system_prompt="텍스트에서 주요 정보를 추출하여 JSON 형식으로 출력하세요."
            )

            self.validator = LLM.create("gpt-4o-mini", system_prompt="데이터의 정확성을 검증하고 문제점을 지적하세요.")

            self.enricher = LLM.create(
                "gpt-4o-mini", system_prompt="주어진 정보를 보강하고 추가 컨텍스트를 제공하세요."
            )

            self.formatter = LLM.create(
                "gpt-4o-mini", system_prompt="정보를 보기 좋게 포맷팅하여 최종 보고서를 작성하세요."
            )

        def process(self, raw_text: str) -> Dict[str, Any]:
            """파이프라인 실행"""
            results = {"stages": []}

            # 1단계: 정보 추출
            print("1️⃣ 정보 추출 중...")
            extract_prompt = f"""
다음 텍스트에서 주요 정보를 추출하세요:
{raw_text}

JSON 형식으로 출력:
{{
    "main_topic": "주제",
    "key_points": ["포인트1", "포인트2"],
    "entities": ["개체1", "개체2"],
    "sentiment": "긍정/중립/부정"
}}
"""
            extract_result = self.extractor.ask(extract_prompt)
            results["stages"].append({"stage": "extraction", "output": extract_result.text})

            # 2단계: 검증
            print("2️⃣ 데이터 검증 중...")
            validate_prompt = f"""
추출된 데이터를 검증하세요:
{extract_result.text}

원본 텍스트:
{raw_text}

정확성과 완전성을 평가하고 개선점을 제시하세요.
"""
            validate_result = self.validator.ask(validate_prompt)
            results["stages"].append({"stage": "validation", "output": validate_result.text})

            # 3단계: 보강
            print("3️⃣ 정보 보강 중...")
            enrich_prompt = f"""
다음 정보를 보강하고 추가 컨텍스트를 제공하세요:
{extract_result.text}

검증 결과:
{validate_result.text}
"""
            enrich_result = self.enricher.ask(enrich_prompt)
            results["stages"].append({"stage": "enrichment", "output": enrich_result.text})

            # 4단계: 최종 포맷팅
            print("4️⃣ 최종 보고서 작성 중...")
            format_prompt = f"""
다음 정보를 바탕으로 읽기 쉬운 최종 보고서를 작성하세요:

원본 데이터: {extract_result.text}
검증 결과: {validate_result.text}
보강된 정보: {enrich_result.text}

보고서는 다음 형식을 따르세요:
1. 요약
2. 주요 발견사항
3. 상세 분석
4. 권장사항
"""
            format_result = self.formatter.ask(format_prompt)
            results["stages"].append({"stage": "formatting", "output": format_result.text})

            results["final_report"] = format_result.text
            return results

    # 파이프라인 실행
    pipeline = DataProcessingPipeline()

    sample_text = """
    ABC 회사는 2024년 1분기에 매출 150억원을 기록했습니다. 
    이는 전년 동기 대비 23% 증가한 수치입니다. 
    주요 성장 동력은 신제품 출시와 해외 시장 확대였습니다. 
    특히 동남아시아 지역에서 45%의 성장을 보였습니다.
    """

    print(f"입력 텍스트: {sample_text}\n")

    result = pipeline.process(sample_text)

    print("\n📋 최종 보고서:")
    print("-" * 50)
    print(result["final_report"])


def example_feedback_loop_chain():
    """피드백 루프 체인 예제"""
    print("\n🔄 피드백 루프 체인")
    print("-" * 50)

    class FeedbackChain:
        """자기 개선 피드백 루프"""

        def __init__(self, max_iterations: int = 3):
            self.max_iterations = max_iterations
            self.generator = LLM.create("gpt-4o-mini", system_prompt="주어진 주제에 대한 설명을 작성하세요.")
            self.critic = LLM.create("gpt-4o-mini", system_prompt="텍스트를 평가하고 구체적인 개선점을 제시하세요.")
            self.improver = LLM.create("gpt-4o-mini", system_prompt="피드백을 바탕으로 텍스트를 개선하세요.")

        def generate_with_feedback(self, topic: str) -> Dict[str, Any]:
            """피드백을 통한 반복 개선"""
            results = {"topic": topic, "iterations": []}

            # 초기 생성
            print(f"🎯 주제: {topic}")
            current_text = self.generator.ask(f"주제: {topic}").text

            for i in range(self.max_iterations):
                print(f"\n🔄 반복 {i+1}/{self.max_iterations}")

                iteration_result = {"iteration": i + 1, "text": current_text}

                # 비평
                critique_prompt = f"""
다음 텍스트를 평가하고 개선점을 제시하세요:

텍스트:
{current_text}

평가 기준:
1. 정확성
2. 명확성
3. 완전성
4. 가독성
"""
                critique = self.critic.ask(critique_prompt).text
                iteration_result["critique"] = critique

                print(f"   📝 현재 텍스트: {current_text[:100]}...")
                print(f"   🔍 비평: {critique[:100]}...")

                # 개선
                improve_prompt = f"""
다음 피드백을 바탕으로 텍스트를 개선하세요:

원본 텍스트:
{current_text}

피드백:
{critique}
"""
                improved_text = self.improver.ask(improve_prompt).text
                iteration_result["improved_text"] = improved_text

                results["iterations"].append(iteration_result)
                current_text = improved_text

            results["final_text"] = current_text
            return results

    # 피드백 체인 실행
    feedback_chain = FeedbackChain(max_iterations=2)
    result = feedback_chain.generate_with_feedback("양자 컴퓨팅의 원리와 응용")

    print("\n📈 개선 과정:")
    for iteration in result["iterations"]:
        print(f"\n반복 {iteration['iteration']}:")
        print(f"  텍스트: {iteration['text'][:150]}...")
        print(f"  비평: {iteration['critique'][:150]}...")

    print("\n✅ 최종 결과:")
    print(result["final_text"])


def main():
    """체이닝 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("🔗 체이닝 예제")
    print("=" * 50)

    try:
        # 1. 기본 체이닝
        example_basic_chaining()

        # 2. 조건부 체이닝
        example_conditional_chaining()

        # 3. 병렬 처리 체인
        example_parallel_chaining()

        # 4. 복잡한 파이프라인
        example_complex_pipeline()

        # 5. 피드백 루프 체인
        example_feedback_loop_chain()

        print("\n✅ 모든 체이닝 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
