#!/usr/bin/env python3
"""
예제: 구조화된 출력 (간단 버전)
난이도: 중급
설명: JSON 형식으로 구조화된 응답 받기
요구사항: OPENAI_API_KEY 환경 변수
"""

import os
import json
from typing import List
from pyhub.llm import LLM


def get_structured_response(llm, prompt: str) -> dict:
    """JSON 형식의 구조화된 응답 받기"""
    # JSON 응답을 요청하는 프롬프트
    json_prompt = f"{prompt}\n\nJSON 형식으로만 응답해주세요."
    
    reply = llm.ask(json_prompt)
    
    try:
        # JSON 파싱
        return json.loads(reply.text)
    except json.JSONDecodeError:
        # JSON이 아닌 경우 텍스트에서 JSON 부분 추출 시도
        text = reply.text
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except:
                pass
        return {"error": "JSON 파싱 실패", "raw_text": text}


def main():
    """구조화된 출력 간단 예제"""
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        return
    
    print("🏗️  구조화된 출력 (간단 버전)")
    print("=" * 50)
    
    # LLM 생성
    llm = LLM.create("gpt-4o-mini")
    
    # 1. 간단한 정보 추출
    print("\n1️⃣ 간단한 정보 추출")
    prompt = """
    "파이썬"에 대해 다음 형식으로 정보를 제공해주세요:
    {
        "name": "언어 이름",
        "creator": "만든 사람",
        "year": 출시 연도,
        "paradigms": ["프로그래밍 패러다임들"],
        "popular_uses": ["주요 사용 분야들"]
    }
    """
    
    result = get_structured_response(llm, prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 2. 할 일 목록 생성
    print("\n\n2️⃣ 할 일 목록 생성")
    prompt = """
    "웹사이트 만들기" 프로젝트의 할 일 목록을 다음 형식으로 만들어주세요:
    {
        "project": "프로젝트명",
        "tasks": [
            {"id": 1, "task": "할 일", "priority": "high/medium/low", "estimated_hours": 시간}
        ],
        "total_hours": 총 예상 시간
    }
    """
    
    result = get_structured_response(llm, prompt)
    if "tasks" in result:
        print(f"프로젝트: {result.get('project', 'N/A')}")
        print(f"총 예상 시간: {result.get('total_hours', 'N/A')}시간\n")
        print("작업 목록:")
        for task in result.get('tasks', []):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.get('priority', ''), "⚪")
            print(f"{priority_emoji} {task.get('id')}. {task.get('task')} ({task.get('estimated_hours')}시간)")
    else:
        print(result)
    
    # 3. 비교 분석
    print("\n\n3️⃣ 비교 분석")
    prompt = """
    Python과 JavaScript를 비교해서 다음 형식으로 정리해주세요:
    {
        "comparison": {
            "python": {
                "strengths": ["강점들"],
                "weaknesses": ["약점들"],
                "best_for": ["적합한 용도"]
            },
            "javascript": {
                "strengths": ["강점들"],
                "weaknesses": ["약점들"],
                "best_for": ["적합한 용도"]
            }
        },
        "recommendation": "상황별 추천"
    }
    """
    
    result = get_structured_response(llm, prompt)
    if "comparison" in result:
        for lang in ["python", "javascript"]:
            lang_data = result["comparison"].get(lang, {})
            print(f"\n{lang.upper()}:")
            print(f"  강점: {', '.join(lang_data.get('strengths', []))}")
            print(f"  약점: {', '.join(lang_data.get('weaknesses', []))}")
            print(f"  적합: {', '.join(lang_data.get('best_for', []))}")
        print(f"\n추천: {result.get('recommendation', 'N/A')}")
    else:
        print(result)
    
    print("\n✅ 구조화된 출력 예제 완료!")


if __name__ == "__main__":
    main()