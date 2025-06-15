#!/usr/bin/env python3
"""
예제: 분류 및 선택
난이도: 중급
설명: LLM을 사용한 텍스트 분류, 감정 분석, 의도 파악
요구사항:
  - pyhub-llm (pip install "pyhub-llm[all]")
  - OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import json
import os
import sys

from pyhub.llm import LLM


def example_sentiment_analysis():
    """감정 분석 예제"""
    print("\n😊 감정 분석")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")

    # 분석할 텍스트들
    texts = [
        "오늘 정말 기분이 좋아요! 최고의 하루였습니다.",
        "이 제품 정말 실망스럽네요. 돈이 아깝습니다.",
        "그냥 평범한 하루였어요. 특별한 일은 없었습니다.",
        "처음엔 별로였는데 쓰다보니 괜찮네요. 나쁘지 않아요.",
        "와! 대박! 이런 건 처음 봐요! 완전 강추!",
    ]

    print("텍스트별 감정 분석 결과:\n")

    for text in texts:
        # choices 파라미터로 감정 분류
        prompt = f"다음 텍스트의 감정을 분석하세요: '{text}'"

        # 단순 분류
        reply = llm.ask(prompt, choices=["긍정", "부정", "중립"])

        print(f"텍스트: {text}")
        print(f"감정: {reply.choice}")

        # 상세 분석 (JSON 형식)
        detailed_prompt = f"""
다음 텍스트의 감정을 상세히 분석하여 JSON 형식으로 출력하세요:
텍스트: {text}

형식:
{{
    "sentiment": "긍정/부정/중립",
    "confidence": 0.0-1.0,
    "emotions": ["감정1", "감정2"],
    "intensity": "약함/보통/강함"
}}
"""

        detailed_reply = llm.ask(detailed_prompt)
        try:
            analysis = json.loads(detailed_reply.text)
            print(f"상세 분석: {analysis}")
        except json.JSONDecodeError:
            print(f"상세 분석: {detailed_reply.text[:100]}")

        print("-" * 30)


def example_intent_classification():
    """의도 분류 예제"""
    print("\n🎯 의도 분류")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")

    # 고객 문의 예시
    customer_queries = [
        "이 제품 환불하고 싶은데요",
        "배송이 언제쯤 도착하나요?",
        "제품 사용법을 알려주세요",
        "가격 할인은 안 되나요?",
        "A/S 신청하려고 하는데요",
        "다른 색상도 있나요?",
        "대량 구매 시 할인 가능한가요?",
    ]

    # 의도 카테고리
    intent_categories = ["환불/반품", "배송문의", "사용방법", "가격문의", "A/S요청", "제품정보", "구매상담", "기타"]

    print("고객 문의 의도 분류:\n")

    for query in customer_queries:
        reply = llm.ask(f"다음 고객 문의의 의도를 분류하세요: '{query}'", choices=intent_categories)

        print(f"문의: {query}")
        print(f"의도: {reply.choice}")

        # 추가 정보 추출
        info_prompt = f"""
고객 문의: {query}

다음 정보를 추출하여 JSON으로 출력하세요:
{{
    "intent": "주요 의도",
    "urgency": "높음/중간/낮음",
    "sentiment": "긍정/중립/부정",
    "keywords": ["키워드1", "키워드2"]
}}
"""

        info_reply = llm.ask(info_prompt)
        try:
            info = json.loads(info_reply.text)
            print(
                f"추가 정보: 긴급도={info.get('urgency', 'N/A')}, "
                f"감정={info.get('sentiment', 'N/A')}, "
                f"키워드={info.get('keywords', [])}"
            )
        except json.JSONDecodeError:
            pass

        print("-" * 30)


def example_topic_classification():
    """주제 분류 예제"""
    print("\n📚 주제 분류")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")

    # 뉴스 헤드라인
    headlines = [
        "코스피 3000 돌파, 외국인 매수세 지속",
        "파이썬 3.13 릴리즈, 성능 50% 향상",
        "손흥민 시즌 15호골, 팀 승리 견인",
        "서울 아파트값 0.5% 상승, 전세는 하락",
        "애플 비전프로 국내 출시 임박",
        "기후변화로 북극곰 서식지 30% 감소",
    ]

    # 주제 카테고리
    topics = ["경제", "기술", "스포츠", "부동산", "환경", "정치", "사회", "문화"]

    print("뉴스 헤드라인 주제 분류:\n")

    results = []
    for headline in headlines:
        # 단일 주제 분류
        reply = llm.ask(f"다음 뉴스 헤드라인의 주제를 분류하세요: '{headline}'", choices=topics)

        results.append({"headline": headline, "topic": reply.choice})

        print(f"📰 {headline}")
        print(f"   → 주제: {reply.choice}")

    # 주제별 통계
    print("\n📊 주제별 분포:")
    topic_count = {}
    for result in results:
        topic = result["topic"]
        topic_count[topic] = topic_count.get(topic, 0) + 1

    for topic, count in sorted(topic_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {topic}: {count}건")


def example_multi_label_classification():
    """다중 레이블 분류 예제"""
    print("\n🏷️ 다중 레이블 분류")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")

    # 영화 설명
    movies = [
        {
            "title": "인터스텔라",
            "description": "지구의 미래를 위해 우주로 떠나는 과학자들의 이야기. 가족애와 시간의 의미를 다룬다.",
        },
        {
            "title": "기생충",
            "description": "반지하에 사는 가난한 가족이 부유한 가족의 삶에 침투하는 블랙 코미디 스릴러.",
        },
        {
            "title": "라라랜드",
            "description": "LA에서 꿈을 쫓는 재즈 피아니스트와 배우 지망생의 사랑 이야기를 그린 뮤지컬.",
        },
    ]

    # 가능한 장르들
    all_genres = [
        "액션",
        "드라마",
        "코미디",
        "로맨스",
        "SF",
        "판타지",
        "스릴러",
        "공포",
        "뮤지컬",
        "다큐멘터리",
        "가족",
        "모험",
    ]

    print("영화 장르 분류 (복수 선택):\n")

    for movie in movies:
        prompt = f"""
다음 영화의 장르를 모두 선택하여 JSON 배열로 출력하세요.
가능한 장르: {', '.join(all_genres)}

영화: {movie['title']}
설명: {movie['description']}

선택된 장르만 JSON 배열로 출력하세요. 예: ["드라마", "로맨스"]
"""

        reply = llm.ask(prompt)

        try:
            genres = json.loads(reply.text)
            print(f"🎬 {movie['title']}")
            print(f"   장르: {', '.join(genres)}")

            # 주요 장르 결정
            main_genre_prompt = f"영화 '{movie['title']}'의 가장 주요한 장르 하나만 선택하세요."
            main_reply = llm.ask(main_genre_prompt, choices=genres)
            print(f"   주요 장르: {main_reply.choice}")

        except json.JSONDecodeError:
            print(f"🎬 {movie['title']}")
            print("   장르 분석 실패")

        print("-" * 30)


def example_confidence_scoring():
    """신뢰도 점수 포함 분류"""
    print("\n📊 신뢰도 점수 포함 분류")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")

    # 애매한 텍스트들
    ambiguous_texts = [
        "음... 글쎄요. 나쁘진 않은데 그렇다고 좋지도 않네요.",
        "완전 최고! 다시는 안 살 거예요!",  # 모순적
        "가격대비 훌륭합니다.",
        "조금 아쉽지만 만족합니다.",
        "?",  # 매우 애매한 경우
    ]

    print("애매한 텍스트의 감정 분석 (신뢰도 포함):\n")

    for text in ambiguous_texts:
        prompt = f"""
다음 텍스트의 감정을 분석하고 신뢰도를 포함하여 JSON으로 출력하세요:
텍스트: "{text}"

형식:
{{
    "sentiment": "긍정/부정/중립",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거",
    "alternative": "대안 해석 (있는 경우)"
}}
"""

        reply = llm.ask(prompt)

        try:
            result = json.loads(reply.text)
            print(f"텍스트: {text}")
            print(f"분류: {result['sentiment']} (신뢰도: {result['confidence']})")
            print(f"근거: {result['reasoning']}")
            if result.get("alternative"):
                print(f"대안: {result['alternative']}")
        except json.JSONDecodeError:
            print(f"텍스트: {text}")
            print("분석 실패")

        print("-" * 30)


def example_hierarchical_classification():
    """계층적 분류 예제"""
    print("\n🌳 계층적 분류")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")

    # 제품 설명
    products = [
        "삼성 갤럭시 S24 울트라 256GB 블랙",
        "나이키 에어맥스 270 런닝화 화이트",
        "LG 올레드 TV 65인치 4K",
        "아이패드 프로 12.9 M2 와이파이 모델",
        "다이슨 V15 무선청소기",
    ]

    print("제품 계층적 분류:\n")

    for product in products:
        # 1단계: 대분류
        major_categories = ["전자제품", "의류/신발", "가전제품", "가구", "식품"]
        reply1 = llm.ask(f"제품 '{product}'의 대분류를 선택하세요.", choices=major_categories)
        major = reply1.choice

        # 2단계: 중분류 (대분류에 따라 다름)
        if major == "전자제품":
            sub_categories = ["스마트폰", "태블릿", "노트북", "카메라", "기타"]
        elif major == "의류/신발":
            sub_categories = ["운동화", "구두", "의류", "액세서리", "기타"]
        elif major == "가전제품":
            sub_categories = ["TV", "냉장고", "세탁기", "청소기", "주방가전"]
        else:
            sub_categories = ["기타"]

        reply2 = llm.ask(f"제품 '{product}'의 중분류를 선택하세요.", choices=sub_categories)

        print(f"📦 {product}")
        print(f"   대분류: {major}")
        print(f"   중분류: {reply2.choice}")

        # 추가 속성 추출
        attr_prompt = f"""
제품 '{product}'의 주요 속성을 추출하여 JSON으로 출력하세요:
{{
    "brand": "브랜드명",
    "model": "모델명",
    "key_features": ["특징1", "특징2"]
}}
"""
        attr_reply = llm.ask(attr_prompt)
        try:
            attrs = json.loads(attr_reply.text)
            print(f"   브랜드: {attrs.get('brand', 'N/A')}")
            print(f"   특징: {', '.join(attrs.get('key_features', []))}")
        except json.JSONDecodeError:
            pass

        print("-" * 30)


def main():
    """분류 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("🏷️ 분류 및 선택 예제")
    print("=" * 50)

    try:
        # 1. 감정 분석
        example_sentiment_analysis()

        # 2. 의도 분류
        example_intent_classification()

        # 3. 주제 분류
        example_topic_classification()

        # 4. 다중 레이블 분류
        example_multi_label_classification()

        # 5. 신뢰도 점수
        example_confidence_scoring()

        # 6. 계층적 분류
        example_hierarchical_classification()

        print("\n✅ 모든 분류 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
