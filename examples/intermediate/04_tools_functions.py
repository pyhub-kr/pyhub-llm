#!/usr/bin/env python3
"""
예제: 도구/함수 호출
난이도: 중급
설명: LLM이 외부 도구나 함수를 호출하는 방법
요구사항: OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict

from pyhub.llm import LLM


# 도구 정의
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """현재 날씨 정보를 가져오는 함수 (시뮬레이션)"""
    # 실제로는 날씨 API를 호출하겠지만, 여기서는 시뮬레이션
    weather_data = {
        "서울": {"temp": 25, "condition": "맑음", "humidity": 60},
        "부산": {"temp": 28, "condition": "구름 조금", "humidity": 70},
        "제주": {"temp": 30, "condition": "흐림", "humidity": 80},
        "뉴욕": {"temp": 20, "condition": "비", "humidity": 85},
    }

    data = weather_data.get(location, {"temp": 20, "condition": "알 수 없음", "humidity": 50})

    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9 / 5 + 32
        unit_str = "°F"
    else:
        unit_str = "°C"

    return json.dumps(
        {
            "location": location,
            "temperature": f"{data['temp']}{unit_str}",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%",
        },
        ensure_ascii=False,
    )


def calculate(expression: str) -> str:
    """수학 계산을 수행하는 함수"""
    try:
        # simpleeval을 사용한 안전한 계산
        try:
            import simpleeval
            evaluator = simpleeval.SimpleEval()
            # 추가 수학 함수들 허용
            evaluator.functions.update({
                'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'sqrt': math.sqrt,
                'log': math.log, 'log10': math.log10, 'exp': math.exp,
                'floor': math.floor, 'ceil': math.ceil, 'pow': pow
            })
            evaluator.names.update({
                'pi': math.pi, 'e': math.e
            })
            result = evaluator.eval(expression)
        except ImportError:
            # simpleeval이 없으면 제한된 eval 사용 (보안 강화)
            import re
            # 위험한 키워드 검사
            dangerous_patterns = [
                r'__\w+__', r'import', r'exec', r'eval', r'open', r'file',
                r'globals', r'locals', r'vars', r'dir', r'getattr', r'setattr'
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, expression, re.IGNORECASE):
                    raise ValueError(f"위험한 키워드가 포함됨: {pattern}")

            # 허용된 문자만 확인
            if not re.match(r'^[0-9+\-*/().,\s_a-zA-Z]+$', expression):
                raise ValueError("허용되지 않은 문자가 포함됨")

            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            allowed_names.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow})
            result = eval(expression, {"__builtins__": {}}, allowed_names)

        return json.dumps({"result": result, "expression": expression})
    except Exception as e:
        return json.dumps({"error": str(e), "expression": expression})


def search_web(query: str, max_results: int = 3) -> str:
    """웹 검색 함수 (시뮬레이션)"""
    # 실제로는 검색 API를 사용하겠지만, 여기서는 시뮬레이션
    mock_results = {
        "파이썬": [
            {"title": "Python.org", "url": "https://python.org", "snippet": "Python 공식 웹사이트"},
            {
                "title": "Python 튜토리얼",
                "url": "https://docs.python.org/ko/3/tutorial/",
                "snippet": "파이썬 공식 튜토리얼",
            },
        ],
        "머신러닝": [
            {"title": "머신러닝이란?", "url": "https://example.com/ml", "snippet": "머신러닝 기초 개념 설명"},
            {"title": "scikit-learn", "url": "https://scikit-learn.org", "snippet": "파이썬 머신러닝 라이브러리"},
        ],
    }

    # 쿼리에 포함된 키워드로 결과 반환
    for keyword, results in mock_results.items():
        if keyword in query:
            return json.dumps({"query": query, "results": results[:max_results]}, ensure_ascii=False)

    return json.dumps({"query": query, "results": []}, ensure_ascii=False)


def get_current_time(timezone: str = "Asia/Seoul") -> str:
    """현재 시간을 반환하는 함수"""
    # 간단한 시뮬레이션 (실제로는 pytz 등을 사용)
    now = datetime.now()
    return json.dumps({"timezone": timezone, "time": now.strftime("%Y-%m-%d %H:%M:%S"), "timestamp": now.timestamp()})


# 도구 스키마 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "특정 지역의 현재 날씨 정보를 가져옵니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "날씨를 확인할 지역 (예: 서울, 부산, 뉴욕)"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "온도 단위"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "수학 계산을 수행합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "계산할 수식 (예: 2+2, sqrt(16), sin(3.14/2))"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "웹에서 정보를 검색합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색할 쿼리"},
                    "max_results": {"type": "integer", "description": "최대 결과 수", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "현재 시간을 가져옵니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "시간대 (예: Asia/Seoul, America/New_York)",
                        "default": "Asia/Seoul",
                    }
                },
            },
        },
    },
]


def example_basic_tool_use():
    """기본 도구 사용 예제"""
    print("\n🔧 기본 도구 사용 예제")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")

    # 날씨 정보 요청
    question = "서울과 부산의 현재 날씨를 알려주세요."
    print(f"질문: {question}\n")

    reply = llm.ask_with_tools(
        question,
        tools=tools,
        tool_functions={
            "get_current_weather": get_current_weather,
            "calculate": calculate,
            "search_web": search_web,
            "get_current_time": get_current_time,
        },
    )

    print(f"응답: {reply.text}")

    # 도구 호출 내역 확인
    if reply.tool_calls:
        print("\n📋 도구 호출 내역:")
        for call in reply.tool_calls:
            print(f"  - {call.name}({call.arguments})")


def example_multi_tool_use():
    """여러 도구 조합 사용 예제"""
    print("\n🔀 여러 도구 조합 사용 예제")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")

    # 복잡한 요청
    question = """
    다음을 수행해주세요:
    1. 현재 시간을 알려주세요
    2. 서울의 날씨를 확인해주세요
    3. 만약 온도가 25도 이상이면, 화씨로 변환해주세요
    4. 파이썬에 대해 웹 검색을 해주세요
    """

    print(f"요청:\n{question}\n")

    reply = llm.ask_with_tools(
        question,
        tools=tools,
        tool_functions={
            "get_current_weather": get_current_weather,
            "calculate": calculate,
            "search_web": search_web,
            "get_current_time": get_current_time,
        },
    )

    print("응답:")
    print(reply.text)

    # 사용된 도구 분석
    if reply.tool_calls:
        print(f"\n📊 총 {len(reply.tool_calls)}개의 도구가 사용되었습니다:")
        tool_usage = {}
        for call in reply.tool_calls:
            tool_usage[call.name] = tool_usage.get(call.name, 0) + 1

        for tool_name, count in tool_usage.items():
            print(f"  - {tool_name}: {count}회")


def example_custom_tool_handler():
    """커스텀 도구 핸들러 예제"""
    print("\n🎯 커스텀 도구 핸들러 예제")
    print("-" * 50)

    # 데이터베이스 시뮬레이션
    mock_database = {
        "users": [
            {"id": 1, "name": "김철수", "age": 30, "city": "서울"},
            {"id": 2, "name": "이영희", "age": 25, "city": "부산"},
            {"id": 3, "name": "박민수", "age": 35, "city": "대전"},
        ],
        "products": [
            {"id": 1, "name": "노트북", "price": 1500000, "stock": 10},
            {"id": 2, "name": "마우스", "price": 50000, "stock": 100},
            {"id": 3, "name": "키보드", "price": 100000, "stock": 50},
        ],
    }

    def query_database(table: str, condition: Dict[str, Any] = None) -> str:
        """데이터베이스 쿼리 함수"""
        if table not in mock_database:
            return json.dumps({"error": f"Table '{table}' not found"})

        results = mock_database[table]

        # 조건 필터링
        if condition:
            filtered = []
            for item in results:
                match = True
                for key, value in condition.items():
                    if key not in item or item[key] != value:
                        match = False
                        break
                if match:
                    filtered.append(item)
            results = filtered

        return json.dumps({"table": table, "count": len(results), "data": results}, ensure_ascii=False)

    # 데이터베이스 도구 추가
    db_tool = {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "데이터베이스에서 정보를 조회합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "enum": ["users", "products"], "description": "조회할 테이블"},
                    "condition": {"type": "object", "description": "조회 조건 (선택사항)"},
                },
                "required": ["table"],
            },
        },
    }

    llm = LLM.create("gpt-4o-mini")

    # 데이터베이스 관련 질문
    questions = [
        "모든 사용자 목록을 보여주세요",
        "서울에 사는 사용자를 찾아주세요",
        "가격이 100만원 이상인 제품을 찾아주세요",
    ]

    for question in questions:
        print(f"\n💬 {question}")

        reply = llm.ask_with_tools(question, tools=[db_tool], tool_functions={"query_database": query_database})

        print(f"답변: {reply.text}")


def example_error_handling():
    """도구 에러 처리 예제"""
    print("\n⚠️  도구 에러 처리 예제")
    print("-" * 50)

    def risky_function(operation: str, value: float) -> str:
        """에러가 발생할 수 있는 함수"""
        if operation == "divide":
            if value == 0:
                raise ValueError("0으로 나눌 수 없습니다")
            return json.dumps({"result": 100 / value})
        elif operation == "sqrt":
            if value < 0:
                raise ValueError("음수의 제곱근은 계산할 수 없습니다")
            return json.dumps({"result": math.sqrt(value)})
        else:
            raise ValueError(f"알 수 없는 연산: {operation}")

    risky_tool = {
        "type": "function",
        "function": {
            "name": "risky_function",
            "description": "위험한 수학 연산을 수행합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["divide", "sqrt"], "description": "수행할 연산"},
                    "value": {"type": "number", "description": "연산에 사용할 값"},
                },
                "required": ["operation", "value"],
            },
        },
    }

    llm = LLM.create("gpt-4o-mini")

    # 에러가 발생하는 요청들
    requests = [
        "100을 5로 나눈 값은?",
        "100을 0으로 나눈 값은?",  # 에러 발생
        "16의 제곱근은?",
        "-16의 제곱근은?",  # 에러 발생
    ]

    for request in requests:
        print(f"\n🔍 {request}")

        try:
            reply = llm.ask_with_tools(request, tools=[risky_tool], tool_functions={"risky_function": risky_function})
            print(f"✅ 결과: {reply.text}")
        except Exception as e:
            print(f"❌ 에러: {e}")


def main():
    """도구/함수 호출 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("🛠️  도구/함수 호출 예제")
    print("=" * 50)

    try:
        # 1. 기본 도구 사용
        example_basic_tool_use()

        # 2. 여러 도구 조합 사용
        example_multi_tool_use()

        # 3. 커스텀 도구 핸들러
        example_custom_tool_handler()

        # 4. 에러 처리
        example_error_handling()

        print("\n✅ 모든 도구 호출 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
