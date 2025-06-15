#!/usr/bin/env python3
"""
예제: 구조화된 출력
난이도: 중급
설명: Pydantic 모델을 사용한 구조화된 응답 받기
요구사항: OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import os
from typing import List

from pydantic import BaseModel, Field

from pyhub.llm import LLM


# Pydantic 모델 정의
class BookReview(BaseModel):
    """책 리뷰 구조"""

    title: str = Field(description="책 제목")
    author: str = Field(description="저자")
    rating: float = Field(description="평점 (0-5)", ge=0, le=5)
    summary: str = Field(description="한 줄 요약")
    pros: List[str] = Field(description="장점 목록")
    cons: List[str] = Field(description="단점 목록")
    recommended_for: List[str] = Field(description="추천 대상")


class WeatherInfo(BaseModel):
    """날씨 정보 구조"""

    location: str = Field(description="위치")
    temperature: float = Field(description="온도 (섭씨)")
    condition: str = Field(description="날씨 상태")
    humidity: int = Field(description="습도 (%)", ge=0, le=100)
    wind_speed: float = Field(description="풍속 (m/s)")
    forecast: List[str] = Field(description="향후 3일 예보")


class TaskPlan(BaseModel):
    """작업 계획 구조"""

    goal: str = Field(description="목표")
    steps: List[str] = Field(description="단계별 작업")
    estimated_time: str = Field(description="예상 소요 시간")
    required_resources: List[str] = Field(description="필요한 리소스")
    potential_challenges: List[str] = Field(description="예상되는 어려움")


def example_book_review(llm):
    """책 리뷰 예제"""
    print("\n📚 책 리뷰 생성 예제")
    print("-" * 50)

    book = "클린 코드 (로버트 마틴)"

    # 구조화된 출력을 위한 프롬프트 작성
    prompt = f"""
{book}에 대한 리뷰를 다음 JSON 형식으로 작성해주세요:
{{
    "title": "책 제목",
    "author": "저자",
    "rating": 평점 (0-5),
    "summary": "한 줄 요약",
    "pros": ["장점1", "장점2", ...],
    "cons": ["단점1", "단점2", ...],
    "recommended_for": ["추천 대상1", "추천 대상2", ...]
}}
JSON만 출력하고 다른 설명은 하지 마세요.
"""

    print(f"책: {book}")
    print("구조화된 리뷰 생성 중...")

    reply = llm.ask(prompt)

    # JSON 파싱 및 Pydantic 모델로 변환
    import json

    try:
        json_data = json.loads(reply.text)
        review = BookReview(**json_data)
    except Exception as e:
        print(f"⚠️  JSON 파싱 오류: {e}")
        # 기본값으로 처리
        review = BookReview(
            title="클린 코드",
            author="로버트 마틴",
            rating=4.5,
            summary="깨끗한 코드 작성을 위한 필독서",
            pros=["실용적인 예제", "명확한 설명"],
            cons=["다소 장황한 부분"],
            recommended_for=["주니어 개발자", "코드 품질 개선을 원하는 개발자"],
        )

    print(f"\n제목: {review.title}")
    print(f"저자: {review.author}")
    print(f"평점: {'⭐' * int(review.rating)} ({review.rating}/5)")
    print(f"요약: {review.summary}")
    print("\n장점:")
    for pro in review.pros:
        print(f"  ✅ {pro}")
    print("\n단점:")
    for con in review.cons:
        print(f"  ⚠️  {con}")
    print("\n추천 대상:")
    for target in review.recommended_for:
        print(f"  👤 {target}")


def example_weather_info(llm):
    """날씨 정보 예제"""
    print("\n🌤️  날씨 정보 생성 예제")
    print("-" * 50)

    location = "서울"
    prompt = f"{location}의 현재 날씨와 향후 3일 예보를 알려주세요."

    print(f"위치: {location}")
    print("날씨 정보 생성 중...")

    reply = llm.ask(prompt, schema=WeatherInfo)
    weather = reply.structured_data

    print(f"\n📍 위치: {weather.location}")
    print(f"🌡️  온도: {weather.temperature}°C")
    print(f"☁️  상태: {weather.condition}")
    print(f"💧 습도: {weather.humidity}%")
    print(f"💨 풍속: {weather.wind_speed}m/s")
    print("\n📅 3일 예보:")
    for i, forecast in enumerate(weather.forecast, 1):
        print(f"  Day {i}: {forecast}")


def example_task_planning(llm):
    """작업 계획 예제"""
    print("\n📋 작업 계획 생성 예제")
    print("-" * 50)

    task = "웹 스크래핑 프로그램 개발"
    prompt = f"{task}을 위한 상세한 작업 계획을 만들어주세요."

    print(f"작업: {task}")
    print("계획 생성 중...")

    reply = llm.ask(prompt, schema=TaskPlan)
    plan = reply.structured_data

    print(f"\n🎯 목표: {plan.goal}")
    print(f"⏱️  예상 시간: {plan.estimated_time}")
    print("\n📝 작업 단계:")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {step}")
    print("\n🔧 필요 리소스:")
    for resource in plan.required_resources:
        print(f"  • {resource}")
    print("\n⚠️  예상 어려움:")
    for challenge in plan.potential_challenges:
        print(f"  • {challenge}")


def example_multiple_structures(llm):
    """여러 구조 동시 처리 예제"""
    print("\n🔄 여러 구조 동시 처리 예제")
    print("-" * 50)

    # 여러 작업을 한 번에 처리
    class MultiTaskResponse(BaseModel):
        book_recommendation: BookReview
        weather_report: WeatherInfo
        weekly_plan: TaskPlan

    prompt = """
    다음 세 가지를 생성해주세요:
    1. 파이썬 초보자를 위한 책 추천 및 리뷰
    2. 제주도의 날씨 정보
    3. 파이썬 학습 1주일 계획
    """

    print("복합 구조 응답 생성 중...")
    reply = llm.ask(prompt, schema=MultiTaskResponse)
    response = reply.structured_data

    print("\n=== 1. 책 추천 ===")
    print(f"📖 {response.book_recommendation.title}")
    print(f"⭐ 평점: {response.book_recommendation.rating}/5")

    print("\n=== 2. 날씨 정보 ===")
    print(f"📍 {response.weather_report.location}")
    print(f"🌡️  {response.weather_report.temperature}°C, {response.weather_report.condition}")

    print("\n=== 3. 학습 계획 ===")
    print(f"🎯 {response.weekly_plan.goal}")
    print(f"⏱️  {response.weekly_plan.estimated_time}")


def main():
    """구조화된 출력 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        return

    print("🏗️  구조화된 출력 예제")
    print("=" * 50)

    # LLM 생성
    llm = LLM.create("gpt-4o-mini")

    try:
        # 1. 책 리뷰 예제
        example_book_review(llm)

        # 2. 날씨 정보 예제
        example_weather_info(llm)

        # 3. 작업 계획 예제
        example_task_planning(llm)

        # 4. 여러 구조 동시 처리
        example_multiple_structures(llm)

        print("\n✅ 모든 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("💡 구조화된 출력은 모델이 지정된 형식을 따르도록 하는 기능입니다.")


if __name__ == "__main__":
    main()
