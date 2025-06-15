#!/usr/bin/env python3
"""
예제: 대화 관리
난이도: 초급
설명: 대화 내역을 유지하며 채팅하는 방법
요구사항: OPENAI_API_KEY 환경 변수
"""

import os
from pyhub.llm import LLM
from pyhub.llm.types import Message


def print_conversation(messages):
    """대화 내역을 보기 좋게 출력"""
    print("\n📝 현재 대화 내역:")
    print("-" * 50)
    for msg in messages:
        role_emoji = "👤" if msg.role == "user" else "🤖"
        print(f"{role_emoji} {msg.role}: {msg.content}")
    print("-" * 50 + "\n")


def main():
    """대화 관리 예제"""
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        return
    
    print("💬 대화 관리 예제")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    # LLM 생성
    llm = LLM.create("gpt-4o-mini")
    
    # 대화 내역 초기화
    messages = []
    
    # 시스템 메시지 추가 (선택사항)
    system_message = Message(
        role="system",
        content="당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 한국어로 대화합니다."
    )
    messages.append(system_message)
    
    # 대화 루프
    while True:
        # 사용자 입력
        user_input = input("👤 You: ").strip()
        
        # 종료 조건
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("👋 대화를 종료합니다.")
            break
        
        # 빈 입력 처리
        if not user_input:
            continue
        
        # 사용자 메시지 추가
        user_message = Message(role="user", content=user_input)
        messages.append(user_message)
        
        try:
            # AI 응답 받기
            reply = llm.messages(messages)
            print(f"🤖 AI: {reply.text}")
            
            # AI 응답을 대화 내역에 추가
            assistant_message = Message(role="assistant", content=reply.text)
            messages.append(assistant_message)
            
            # 대화가 너무 길어지면 오래된 메시지 제거 (시스템 메시지는 유지)
            if len(messages) > 10:
                messages = [messages[0]] + messages[-9:]  # 시스템 + 최근 9개
                print("\n💡 대화가 길어져 오래된 메시지를 일부 제거했습니다.")
        
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    # 최종 대화 내역 출력
    if len(messages) > 1:  # 시스템 메시지만 있는 경우 제외
        print_conversation(messages[1:])  # 시스템 메시지 제외하고 출력
        print(f"✅ 총 {len(messages)-1}개의 메시지가 교환되었습니다.")


if __name__ == "__main__":
    main()