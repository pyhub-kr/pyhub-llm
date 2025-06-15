#!/usr/bin/env python3
"""
예제: 캐싱
난이도: 중급
설명: 응답 캐싱으로 성능 향상 및 비용 절감
요구사항: OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from pyhub.llm import LLM
from pyhub.llm.cache import BaseCache, FileCache, MemoryCache


class CustomJSONCache(BaseCache):
    """커스텀 JSON 캐시 구현 예제"""

    def __init__(self, cache_file="custom_cache.json"):
        self.cache_file = cache_file
        self.cache_data = {}
        self.load_cache()

    def load_cache(self):
        """캐시 파일 로드"""
        if Path(self.cache_file).exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache_data = json.load(f)

    def save_cache(self):
        """캐시 파일 저장"""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache_data, f, ensure_ascii=False, indent=2)

    def get(self, key: str):
        """캐시에서 값 가져오기"""
        if key in self.cache_data:
            entry = self.cache_data[key]
            # TTL 체크
            if "expires_at" in entry:
                if datetime.fromisoformat(entry["expires_at"]) > datetime.now():
                    print(f"💾 캐시 히트: {key[:50]}...")
                    return entry["value"]
                else:
                    print(f"⏰ 캐시 만료: {key[:50]}...")
                    del self.cache_data[key]
                    self.save_cache()
        return None

    def set(self, key: str, value, ttl: int = 3600):
        """캐시에 값 저장"""
        expires_at = datetime.now().timestamp() + ttl
        self.cache_data[key] = {
            "value": value,
            "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
            "created_at": datetime.now().isoformat(),
        }
        self.save_cache()
        print(f"💾 캐시 저장: {key[:50]}...")

    def clear(self):
        """캐시 클리어"""
        self.cache_data = {}
        if Path(self.cache_file).exists():
            Path(self.cache_file).unlink()
        print("🗑️  캐시 클리어 완료")


def example_memory_cache():
    """메모리 캐시 예제"""
    print("\n🧠 메모리 캐시 예제")
    print("-" * 50)

    # 메모리 캐시 사용
    cache = MemoryCache()
    llm = LLM.create("gpt-4o-mini", cache=cache)

    # 같은 질문을 여러 번 수행
    question = "파이썬의 데코레이터란 무엇인가요?"

    print(f"질문: {question}\n")

    # 첫 번째 호출 (캐시 미스)
    start = time.time()
    reply1 = llm.ask(question)
    time1 = time.time() - start
    print(f"1차 호출 (캐시 미스): {time1:.2f}초")
    print(f"응답 미리보기: {reply1.text[:100]}...\n")

    # 두 번째 호출 (캐시 히트)
    start = time.time()
    reply2 = llm.ask(question)
    time2 = time.time() - start
    print(f"2차 호출 (캐시 히트): {time2:.2f}초")
    print(f"속도 향상: {time1/time2:.0f}배 빠름!")

    # 응답이 동일한지 확인
    print(f"응답 동일: {reply1.text == reply2.text}")


def example_file_cache():
    """파일 캐시 예제"""
    print("\n📁 파일 캐시 예제")
    print("-" * 50)

    # 파일 캐시 사용 (캐시 디렉토리 지정)
    cache_dir = "./llm_cache"
    cache = FileCache(cache_dir=cache_dir)
    llm = LLM.create("gpt-4o-mini", cache=cache)

    print(f"캐시 디렉토리: {cache_dir}\n")

    # 여러 질문 처리
    questions = [
        "머신러닝이란 무엇인가요?",
        "딥러닝이란 무엇인가요?",
        "머신러닝이란 무엇인가요?",  # 중복 질문
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n질문 {i}: {question}")
        start = time.time()
        reply = llm.ask(question)
        elapsed = time.time() - start
        print(f"소요 시간: {elapsed:.2f}초")
        print(f"응답 길이: {len(reply.text)}자")

    # 캐시 파일 확인
    cache_files = list(Path(cache_dir).glob("*.cache"))
    print(f"\n📊 생성된 캐시 파일: {len(cache_files)}개")


def example_custom_cache():
    """커스텀 캐시 예제"""
    print("\n🛠️  커스텀 JSON 캐시 예제")
    print("-" * 50)

    # 커스텀 캐시 사용
    cache = CustomJSONCache("my_llm_cache.json")
    llm = LLM.create("gpt-4o-mini", cache=cache)

    # TTL 테스트를 위한 질문
    question = "캐싱의 장점은 무엇인가요?"

    print(f"질문: {question}\n")

    # 첫 번째 호출
    print("1️⃣ 첫 번째 호출")
    reply = llm.ask(question)
    print(f"응답: {reply.text[:100]}...\n")

    # 캐시 파일 내용 확인
    if Path("my_llm_cache.json").exists():
        with open("my_llm_cache.json", "r", encoding="utf-8") as f:
            cache_content = json.load(f)
        print("📄 캐시 파일 내용:")
        for key, value in list(cache_content.items())[:1]:  # 첫 번째 항목만
            print(f"  - 키: {key[:50]}...")
            print(f"  - 생성 시간: {value['created_at']}")
            print(f"  - 만료 시간: {value['expires_at']}")

    # 두 번째 호출 (캐시 히트)
    print("\n2️⃣ 두 번째 호출 (캐시에서 로드)")
    _ = llm.ask(question)

    # 캐시 클리어
    print("\n🗑️  캐시 클리어")
    cache.clear()


def example_cache_strategy():
    """캐시 전략 예제"""
    print("\n📈 캐시 전략 예제")
    print("-" * 50)

    # 짧은 TTL의 메모리 캐시
    short_cache = MemoryCache(ttl=5)  # 5초 TTL
    llm_short = LLM.create("gpt-4o-mini", cache=short_cache)

    question = "현재 시간은 몇 시인가요?"

    print(f"질문: {question}")
    print("(5초 TTL 캐시 사용)\n")

    # 첫 번째 호출
    print("1️⃣ 첫 번째 호출")
    reply1 = llm_short.ask(question)
    print(f"응답: {reply1.text}\n")

    # 3초 후 (캐시 유효)
    print("⏱️  3초 대기 중...")
    time.sleep(3)
    print("2️⃣ 3초 후 호출 (캐시 유효)")
    reply2 = llm_short.ask(question)
    print(f"응답: {reply2.text}")
    print(f"캐시 사용: {reply1.text == reply2.text}\n")

    # 3초 더 대기 (총 6초, 캐시 만료)
    print("⏱️  3초 더 대기 중...")
    time.sleep(3)
    print("3️⃣ 6초 후 호출 (캐시 만료)")
    reply3 = llm_short.ask(question)
    print(f"응답: {reply3.text}")
    print(f"새로운 응답: {reply1.text != reply3.text}")


def example_cache_statistics():
    """캐시 통계 예제"""
    print("\n📊 캐시 통계 예제")
    print("-" * 50)

    # 통계 기능이 있는 캐시
    class StatisticsCache(MemoryCache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stats = {"hits": 0, "misses": 0}

        def get(self, key: str):
            result = super().get(key)
            if result is not None:
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1
            return result

        def get_statistics(self):
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
            return {
                "total_requests": total,
                "cache_hits": self.stats["hits"],
                "cache_misses": self.stats["misses"],
                "hit_rate": f"{hit_rate:.1f}%",
            }

    # 통계 캐시 사용
    stats_cache = StatisticsCache()
    llm = LLM.create("gpt-4o-mini", cache=stats_cache)

    # 다양한 질문들
    questions = [
        "Python이란?",
        "JavaScript란?",
        "Python이란?",  # 중복
        "Java란?",
        "Python이란?",  # 중복
        "JavaScript란?",  # 중복
    ]

    print("여러 질문 처리 중...\n")
    for q in questions:
        _ = llm.ask(q)
        print(f"✓ {q}")

    # 통계 출력
    stats = stats_cache.get_statistics()
    print("\n📈 캐시 통계:")
    print(f"  - 전체 요청: {stats['total_requests']}")
    print(f"  - 캐시 히트: {stats['cache_hits']}")
    print(f"  - 캐시 미스: {stats['cache_misses']}")
    print(f"  - 히트율: {stats['hit_rate']}")


def main():
    """캐싱 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        return

    print("💾 캐싱 예제")
    print("=" * 50)

    try:
        # 1. 메모리 캐시 예제
        example_memory_cache()

        # 2. 파일 캐시 예제
        example_file_cache()

        # 3. 커스텀 캐시 예제
        example_custom_cache()

        # 4. 캐시 전략 예제
        example_cache_strategy()

        # 5. 캐시 통계 예제
        example_cache_statistics()

        print("\n✅ 모든 캐싱 예제 완료!")

        # 정리
        print("\n🧹 정리 중...")
        # 생성된 캐시 파일들 삭제 (선택사항)
        cache_files = ["my_llm_cache.json"]
        for file in cache_files:
            if Path(file).exists():
                Path(file).unlink()
                print(f"  - {file} 삭제됨")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
