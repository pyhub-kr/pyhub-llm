#!/usr/bin/env python3
"""
예제: 고급 챗봇 구현
난이도: 고급
설명: 컨텍스트 관리, 메모리, 도구 사용을 포함한 챗봇
요구사항:
  - pyhub-llm (pip install "pyhub-llm[all]")
  - OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

from pyhub.llm import LLM
from pyhub.llm.types import Message


class DateTimeEncoder(json.JSONEncoder):
    """datetime 객체를 처리하는 JSON 인코더"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# from pyhub.llm.cache import MemoryCache  # 캐시 기능은 현재 예제에서 사용하지 않음


@dataclass
class ConversationContext:
    """대화 컨텍스트"""

    user_info: Dict[str, Any]
    conversation_id: str
    started_at: datetime
    messages_count: int
    topics: List[str]
    preferences: Dict[str, Any]


class AdvancedChatbot:
    """고급 챗봇 클래스"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = LLM.create(model, system_prompt=self._get_system_prompt())
        # self.cache = MemoryCache(ttl=3600)  # 캐시 기능은 현재 예제에서 사용하지 않음
        self.conversations: Dict[str, List[Message]] = {}
        self.contexts: Dict[str, ConversationContext] = {}

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """당신은 도움이 되고 친절한 AI 어시스턴트입니다.
사용자와의 대화 내역을 기억하고, 맥락을 이해하며, 개인화된 응답을 제공합니다.
필요한 경우 도구를 사용하여 더 정확한 정보를 제공합니다."""

    def create_conversation(self, user_info: Dict[str, Any]) -> str:
        """새 대화 생성"""
        import uuid

        conversation_id = str(uuid.uuid4())

        self.conversations[conversation_id] = []
        self.contexts[conversation_id] = ConversationContext(
            user_info=user_info,
            conversation_id=conversation_id,
            started_at=datetime.now(),
            messages_count=0,
            topics=[],
            preferences={},
        )

        return conversation_id

    def _get_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 정의"""
        return [
            {
                "name": "get_weather",
                "description": "특정 지역의 날씨 정보를 가져옵니다",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "위치 (예: 서울, 부산)"}},
                    "required": ["location"],
                },
            },
            {
                "name": "search_web",
                "description": "웹에서 정보를 검색합니다",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "검색어"}},
                    "required": ["query"],
                },
            },
            {
                "name": "remember_preference",
                "description": "사용자의 선호도를 기억합니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "선호도 키"},
                        "value": {"type": "string", "description": "선호도 값"},
                    },
                    "required": ["key", "value"],
                },
            },
        ]

    def _execute_tool(self, tool_call: Dict[str, Any], conversation_id: str) -> str:
        """도구 실행"""
        tool_name = tool_call.get("name", "")
        parameters = tool_call.get("parameters", {})

        if tool_name == "get_weather":
            location = parameters.get("location", "서울")
            # 실제 구현에서는 날씨 API 호출
            return f"{location}의 현재 날씨는 맑고 기온은 20도입니다."

        elif tool_name == "search_web":
            query = parameters.get("query", "")
            # 실제 구현에서는 검색 API 호출
            return f"'{query}'에 대한 검색 결과: 관련 정보를 찾았습니다."

        elif tool_name == "remember_preference":
            key = parameters.get("key", "")
            value = parameters.get("value", "")
            context = self.contexts.get(conversation_id)
            if context:
                context.preferences[key] = value
            return f"선호도를 기억했습니다: {key} = {value}"

        return "도구를 실행할 수 없습니다."

    def _build_context_prompt(self, conversation_id: str, message: str) -> str:
        """컨텍스트를 포함한 프롬프트 생성"""
        context = self.contexts.get(conversation_id)
        messages = self.conversations.get(conversation_id, [])

        # 컨텍스트 정보 구성
        context_info = []

        if context:
            if context.user_info.get("name"):
                context_info.append(f"사용자 이름: {context.user_info['name']}")

            if context.preferences:
                prefs = ", ".join([f"{k}: {v}" for k, v in context.preferences.items()])
                context_info.append(f"사용자 선호도: {prefs}")

            if context.topics:
                context_info.append(f"대화 주제: {', '.join(context.topics[-3:])}")

        # 최근 대화 내역
        recent_messages = []
        for msg in messages[-5:]:  # 최근 5개 메시지
            if msg.role != "system":
                recent_messages.append(f"{msg.role}: {msg.content}")

        # 전체 프롬프트 구성
        prompt_parts = []

        if context_info:
            prompt_parts.append("=== 컨텍스트 정보 ===")
            prompt_parts.extend(context_info)
            prompt_parts.append("")

        if recent_messages:
            prompt_parts.append("=== 최근 대화 ===")
            prompt_parts.extend(recent_messages)
            prompt_parts.append("")

        prompt_parts.append(f"user: {message}")
        prompt_parts.append("assistant:")

        return "\n".join(prompt_parts)

    def _extract_topics(self, message: str, response: str) -> List[str]:
        """대화에서 주제 추출"""
        # 간단한 구현 - 실제로는 NLP 사용
        topics = []

        keywords = ["날씨", "음식", "여행", "프로그래밍", "AI", "건강", "운동", "영화", "음악"]

        combined_text = f"{message} {response}".lower()
        for keyword in keywords:
            if keyword in combined_text:
                topics.append(keyword)

        return topics

    def chat(self, conversation_id: str, message: str, use_tools: bool = True, stream: bool = False) -> Dict[str, Any]:
        """채팅 메시지 처리"""
        # 컨텍스트 확인
        if conversation_id not in self.conversations:
            raise ValueError(f"대화 ID {conversation_id}를 찾을 수 없습니다.")

        # 캐시 확인 (캐시 기능은 현재 비활성화)
        # cache_key = f"{conversation_id}:{message}"
        # cached_response = self.cache.get(cache_key)
        # if cached_response:
        #     return cached_response

        # 사용자 메시지 저장
        user_message = Message(role="user", content=message)
        self.conversations[conversation_id].append(user_message)

        # 컨텍스트 업데이트
        context = self.contexts[conversation_id]
        context.messages_count += 1

        # 프롬프트 생성
        prompt = self._build_context_prompt(conversation_id, message)

        try:
            if use_tools:
                # 도구 사용 (현재는 시뮬레이션)
                reply = self.llm.ask(prompt)
                response_text = reply.text

                # 도구 호출 시뮬레이션
                tool_results = []
                if "날씨" in message:
                    # 날씨 도구 호출 시뮬레이션
                    location = "서울"  # 실제로는 메시지에서 추출
                    if "부산" in message:
                        location = "부산"
                    elif "제주" in message:
                        location = "제주"

                    weather_result = self._execute_tool(
                        {"name": "get_weather", "parameters": {"location": location}}, conversation_id
                    )
                    tool_results.append({"tool": "get_weather", "result": weather_result})
                    response_text = f"{weather_result}\n\n{response_text}"

            else:
                # 일반 응답
                if stream:
                    # 스트리밍 모드
                    response_text = ""
                    for chunk in self.llm.ask(prompt, stream=True):
                        response_text += chunk.text
                        # 실시간 처리를 위해 yield 사용 가능
                else:
                    reply = self.llm.ask(prompt)
                    response_text = reply.text

            # 어시스턴트 메시지 저장
            assistant_message = Message(role="assistant", content=response_text)
            self.conversations[conversation_id].append(assistant_message)

            # 주제 추출 및 저장
            topics = self._extract_topics(message, response_text)
            context.topics.extend(topics)

            # 응답 구성
            response = {
                "response": response_text,
                "conversation_id": conversation_id,
                "message_count": context.messages_count,
                "topics": list(set(context.topics)),
                "timestamp": datetime.now().isoformat(),
            }

            if use_tools and "tool_results" in locals():
                response["tool_results"] = tool_results

            # 캐시 저장 (캐시 기능은 현재 비활성화)
            # self.cache.set(cache_key, response)

            return response

        except Exception as e:
            print(f"오류 발생: {e}")
            return {
                "response": f"죄송합니다. 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
            }

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """대화 요약"""
        context = self.contexts.get(conversation_id)
        messages = self.conversations.get(conversation_id, [])

        if not context:
            return {"error": "대화를 찾을 수 없습니다."}

        # 대화 내용 요약 생성
        if messages:
            conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages if msg.role != "system"])

            reply = self.llm.ask(f"다음 대화를 한국어로 요약하세요:\n\n{conversation_text}")
            summary = reply.text
        else:
            summary = "대화 내용이 없습니다."

        return {
            "conversation_id": conversation_id,
            "user_info": context.user_info,
            "started_at": context.started_at.isoformat(),
            "message_count": context.messages_count,
            "topics": list(set(context.topics)),
            "preferences": context.preferences,
            "summary": summary,
            "duration": (datetime.now() - context.started_at).total_seconds(),
        }

    def export_conversation(self, conversation_id: str, format: str = "json") -> str:
        """대화 내보내기"""
        messages = self.conversations.get(conversation_id, [])
        context = self.contexts.get(conversation_id)

        if format == "json":
            data = {
                "conversation_id": conversation_id,
                "context": asdict(context) if context else None,
                "messages": [
                    {"role": msg.role, "content": msg.content, "timestamp": datetime.now()}  # datetime 객체로 유지
                    for msg in messages
                ],
            }
            # DateTimeEncoder를 사용하여 datetime 객체를 자동으로 변환
            return json.dumps(data, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

        elif format == "text":
            lines = []
            if context:
                lines.append(f"대화 ID: {conversation_id}")
                lines.append(f"시작 시간: {context.started_at}")
                lines.append(f"사용자: {context.user_info.get('name', '알 수 없음')}")
                lines.append("-" * 50)

            for msg in messages:
                if msg.role != "system":
                    lines.append(f"{msg.role.upper()}: {msg.content}")
                    lines.append("")

            return "\n".join(lines)

        else:
            raise ValueError(f"지원하지 않는 형식: {format}")


def example_basic_chat():
    """기본 채팅 예제"""
    print("\n💬 기본 채팅")
    print("-" * 50)

    # 챗봇 생성
    bot = AdvancedChatbot()

    # 대화 시작
    user_info = {"name": "김철수", "age": 30, "location": "서울"}
    conv_id = bot.create_conversation(user_info)

    print(f"대화 시작 (ID: {conv_id[:8]}...)")

    # 대화
    messages = ["안녕하세요! 저는 김철수입니다.", "오늘 날씨가 어떤가요?", "주말에 가볼만한 곳을 추천해주세요."]

    for msg in messages:
        print(f"\n👤 사용자: {msg}")
        response = bot.chat(conv_id, msg, use_tools=False)
        print(f"🤖 봇: {response['response']}")

    # 대화 요약
    summary = bot.get_conversation_summary(conv_id)
    print("\n📊 대화 요약:")
    print(f"  - 메시지 수: {summary['message_count']}")
    print(f"  - 주제: {', '.join(summary['topics'])}")
    print(f"  - 요약: {summary['summary'][:100]}...")


def example_with_tools():
    """도구 사용 예제"""
    print("\n🔧 도구 사용 채팅")
    print("-" * 50)

    bot = AdvancedChatbot()
    conv_id = bot.create_conversation({"name": "이영희"})

    messages = ["서울 날씨 알려줘", "파이썬 async/await에 대해 검색해줘", "나는 매운 음식을 좋아해. 기억해줘."]

    for msg in messages:
        print(f"\n👤 사용자: {msg}")
        response = bot.chat(conv_id, msg, use_tools=True)
        print(f"🤖 봇: {response['response']}")

        if "tool_results" in response:
            print("📌 사용된 도구:")
            for tool in response["tool_results"]:
                print(f"  - {tool['tool']}: {tool['result']}")


def example_streaming():
    """스트리밍 채팅 예제"""
    print("\n🌊 스트리밍 채팅")
    print("-" * 50)

    bot = AdvancedChatbot()
    conv_id = bot.create_conversation({"name": "박민수"})

    message = "인공지능의 미래에 대해 자세히 설명해주세요."
    print(f"👤 사용자: {message}")
    print("🤖 봇: ", end="", flush=True)

    # 실제 스트리밍 구현
    # 여기서는 시뮬레이션
    response = bot.chat(conv_id, message, use_tools=False, stream=True)
    print(response["response"])


def example_export():
    """대화 내보내기 예제"""
    print("\n💾 대화 내보내기")
    print("-" * 50)

    bot = AdvancedChatbot()
    conv_id = bot.create_conversation({"name": "정수진"})

    # 대화 진행
    bot.chat(conv_id, "안녕하세요!")
    bot.chat(conv_id, "AI에 대해 알고 싶어요")
    bot.chat(conv_id, "감사합니다!")

    # JSON 내보내기
    json_export = bot.export_conversation(conv_id, format="json")
    print("JSON 형식:")
    print(json_export[:200] + "...")

    # 텍스트 내보내기
    text_export = bot.export_conversation(conv_id, format="text")
    print("\n텍스트 형식:")
    print(text_export)


def main():
    """고급 챗봇 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("🤖 고급 챗봇 예제")
    print("=" * 50)

    try:
        # 1. 기본 채팅
        example_basic_chat()

        # 2. 도구 사용
        example_with_tools()

        # 3. 스트리밍
        example_streaming()

        # 4. 내보내기
        example_export()

        print("\n✅ 모든 챗봇 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
