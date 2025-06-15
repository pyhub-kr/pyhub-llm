#!/usr/bin/env python3
"""
예제: History Backup
난이도: 중급
설명: 대화 내역을 백업하고 복원하는 방법
요구사항: OPENAI_API_KEY 환경 변수
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from pyhub.llm import LLM
from pyhub.llm.types import Message
from pyhub.llm.history import InInMemoryHistoryBackup
try:
    from pyhub.llm.history import SQLAlchemyHistoryBackup
except ImportError:
    SQLAlchemyHistoryBackup = None


def example_memory_backup():
    """메모리 기반 백업 예제"""
    print("\n🧠 메모리 기반 History Backup 예제")
    print("-" * 50)
    
    # 메모리 백업 사용
    backup = InMemoryHistoryBackup()
    llm = LLM.create("gpt-4o-mini", history_backup=backup)
    
    # 대화 시작
    conversation_id = "chat_001"
    print(f"대화 ID: {conversation_id}\n")
    
    # 여러 번의 대화
    exchanges = [
        "안녕하세요! 파이썬에 대해 배우고 싶습니다.",
        "파이썬의 주요 특징은 무엇인가요?",
        "파이썬으로 웹 개발을 하려면 어떻게 시작해야 하나요?"
    ]
    
    messages = []
    for i, user_input in enumerate(exchanges, 1):
        print(f"\n라운드 {i}:")
        print(f"👤 User: {user_input}")
        
        # 사용자 메시지 추가
        user_msg = Message(role="user", content=user_input)
        messages.append(user_msg)
        
        # AI 응답
        reply = llm.messages(messages)
        print(f"🤖 AI: {reply.text[:100]}...")
        
        # AI 메시지 추가
        ai_msg = Message(role="assistant", content=reply.text)
        messages.append(ai_msg)
        
        # 백업에 저장
        backup.save_messages(conversation_id, messages)
    
    # 백업에서 복원
    print("\n\n📥 백업에서 대화 복원:")
    restored_messages = backup.load_messages(conversation_id)
    
    print(f"복원된 메시지 수: {len(restored_messages)}")
    for msg in restored_messages[-2:]:  # 마지막 2개만 표시
        print(f"{msg.role}: {msg.content[:50]}...")
    
    # 통계
    print(f"\n📊 백업 통계:")
    print(f"저장된 대화 수: {len(backup.storage)}")
    print(f"총 메시지 수: {sum(len(msgs) for msgs in backup.storage.values())}")


def example_sqlite_backup():
    """SQLite 파일 백업 예제"""
    print("\n💾 SQLite History Backup 예제")
    print("-" * 50)
    
    # SQLite 백업을 위한 간단한 구현
    class SQLiteHistoryBackup:
        def __init__(self, db_path="chat_history.db"):
            self.db_path = db_path
            self._init_db()
        
        def _init_db(self):
            """데이터베이스 초기화"""
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            conn.commit()
            conn.close()
        
        def save_messages(self, conversation_id: str, messages: List[Message]):
            """메시지 저장"""
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 대화 생성/업데이트
            cursor.execute("""
                INSERT OR REPLACE INTO conversations (id, created_at, updated_at)
                VALUES (?, ?, ?)
            """, (conversation_id, datetime.now(), datetime.now()))
            
            # 기존 메시지 삭제 (간단한 구현)
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            
            # 새 메시지 저장
            for msg in messages:
                metadata = {}
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    metadata['tool_calls'] = [tc.dict() for tc in msg.tool_calls]
                
                cursor.execute("""
                    INSERT INTO messages (conversation_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    conversation_id,
                    msg.role,
                    msg.content,
                    datetime.now(),
                    json.dumps(metadata) if metadata else None
                ))
            
            conn.commit()
            conn.close()
        
        def load_messages(self, conversation_id: str) -> List[Message]:
            """메시지 로드"""
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT role, content, metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id
            """, (conversation_id,))
            
            messages = []
            for role, content, metadata in cursor.fetchall():
                msg = Message(role=role, content=content)
                if metadata:
                    # 메타데이터 처리 (필요시)
                    pass
                messages.append(msg)
            
            conn.close()
            return messages
        
        def list_conversations(self):
            """모든 대화 목록"""
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT c.id, c.created_at, COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
            """)
            
            results = cursor.fetchall()
            conn.close()
            return results
    
    # SQLite 백업 사용
    backup = SQLiteHistoryBackup()
    llm = LLM.create("gpt-4o-mini")
    
    # 여러 대화 세션 시뮬레이션
    sessions = [
        {
            "id": "tech_discussion",
            "messages": [
                "파이썬과 자바의 차이점은?",
                "웹 개발에는 어떤 언어가 좋을까요?"
            ]
        },
        {
            "id": "ai_learning",
            "messages": [
                "머신러닝을 시작하려면?",
                "딥러닝과 머신러닝의 차이는?"
            ]
        }
    ]
    
    print("💬 여러 대화 세션 생성 중...\n")
    
    for session in sessions:
        print(f"\n세션: {session['id']}")
        messages = []
        
        for user_input in session['messages']:
            # 사용자 메시지
            user_msg = Message(role="user", content=user_input)
            messages.append(user_msg)
            
            # AI 응답
            reply = llm.messages(messages)
            ai_msg = Message(role="assistant", content=reply.text)
            messages.append(ai_msg)
            
            print(f"  Q: {user_input}")
            print(f"  A: {reply.text[:50]}...")
        
        # 백업에 저장
        backup.save_messages(session['id'], messages)
    
    # 저장된 대화 목록
    print("\n\n📋 저장된 대화 목록:")
    conversations = backup.list_conversations()
    for conv_id, created_at, msg_count in conversations:
        print(f"  - {conv_id}: {msg_count}개 메시지 (생성: {created_at})")
    
    # 특정 대화 복원
    print("\n\n🔄 'ai_learning' 대화 복원:")
    restored = backup.load_messages("ai_learning")
    for msg in restored:
        role_emoji = "👤" if msg.role == "user" else "🤖"
        print(f"{role_emoji} {msg.role}: {msg.content[:80]}...")


def example_advanced_backup():
    """고급 백업 기능 예제"""
    print("\n🚀 고급 History Backup 예제")
    print("-" * 50)
    
    # 고급 기능이 있는 백업 클래스
    class AdvancedHistoryBackup(InMemoryHistoryBackup):
        def __init__(self):
            super().__init__()
            self.metadata = {}  # 대화별 메타데이터
        
        def save_with_metadata(self, conversation_id: str, messages: List[Message], metadata: dict):
            """메타데이터와 함께 저장"""
            self.save_messages(conversation_id, messages)
            self.metadata[conversation_id] = {
                **metadata,
                'message_count': len(messages),
                'last_updated': datetime.now().isoformat()
            }
        
        def search_conversations(self, keyword: str):
            """키워드로 대화 검색"""
            results = []
            for conv_id, messages in self.storage.items():
                for msg in messages:
                    if keyword.lower() in msg.content.lower():
                        results.append({
                            'conversation_id': conv_id,
                            'message': msg,
                            'metadata': self.metadata.get(conv_id, {})
                        })
                        break
            return results
        
        def export_to_json(self, filepath: str):
            """JSON으로 내보내기"""
            export_data = {
                'conversations': {},
                'metadata': self.metadata,
                'export_date': datetime.now().isoformat()
            }
            
            for conv_id, messages in self.storage.items():
                export_data['conversations'][conv_id] = [
                    {'role': msg.role, 'content': msg.content}
                    for msg in messages
                ]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return filepath
        
        def import_from_json(self, filepath: str):
            """JSON에서 가져오기"""
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for conv_id, messages in data['conversations'].items():
                msg_objects = [
                    Message(role=msg['role'], content=msg['content'])
                    for msg in messages
                ]
                self.save_messages(conv_id, msg_objects)
            
            self.metadata.update(data.get('metadata', {}))
    
    # 고급 백업 사용
    backup = AdvancedHistoryBackup()
    llm = LLM.create("gpt-4o-mini", history_backup=backup)
    
    # 메타데이터와 함께 대화 저장
    print("💾 메타데이터와 함께 대화 저장\n")
    
    # 고객 지원 대화
    support_messages = [
        Message(role="system", content="당신은 친절한 고객 지원 담당자입니다."),
        Message(role="user", content="제품이 작동하지 않습니다."),
        Message(role="assistant", content="불편을 드려 죄송합니다. 어떤 문제가 발생했는지 자세히 알려주시겠어요?"),
        Message(role="user", content="전원 버튼을 눌러도 켜지지 않아요."),
        Message(role="assistant", content="전원 코드가 올바르게 연결되어 있는지 확인해 주시겠어요?")
    ]
    
    backup.save_with_metadata(
        "support_001",
        support_messages,
        {
            'customer_id': 'CUST123',
            'issue_type': 'technical',
            'priority': 'high',
            'status': 'ongoing'
        }
    )
    
    # 학습 상담 대화
    learning_messages = [
        Message(role="user", content="파이썬을 배우고 싶은데 어디서부터 시작해야 할까요?"),
        Message(role="assistant", content="파이썬을 배우기 시작하신다니 훌륭한 선택입니다! 초보자를 위한 단계별 학습 방법을 안내해드리겠습니다."),
        Message(role="user", content="프로그래밍은 처음입니다."),
        Message(role="assistant", content="걱정하지 마세요! 파이썬은 초보자에게 가장 적합한 언어입니다.")
    ]
    
    backup.save_with_metadata(
        "learning_001",
        learning_messages,
        {
            'topic': 'python',
            'level': 'beginner',
            'session_type': 'consultation'
        }
    )
    
    # 대화 검색
    print("\n🔍 '파이썬' 키워드로 대화 검색:")
    search_results = backup.search_conversations("파이썬")
    for result in search_results:
        print(f"\n대화 ID: {result['conversation_id']}")
        print(f"메타데이터: {result['metadata']}")
        print(f"관련 메시지: {result['message'].content[:50]}...")
    
    # JSON 내보내기/가져오기
    print("\n\n📤 JSON으로 내보내기:")
    export_path = backup.export_to_json("chat_backup.json")
    print(f"내보내기 완료: {export_path}")
    
    # 파일 내용 미리보기
    with open(export_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        print(f"내보낸 대화 수: {len(content['conversations'])}")
        print(f"메타데이터 포함: {bool(content['metadata'])}")
    
    # 새 백업 인스턴스에서 가져오기
    print("\n📥 JSON에서 가져오기:")
    new_backup = AdvancedHistoryBackup()
    new_backup.import_from_json(export_path)
    print(f"가져온 대화 수: {len(new_backup.storage)}")
    
    # 정리
    if Path(export_path).exists():
        Path(export_path).unlink()
        print("\n🧹 임시 파일 정리 완료")


def example_tool_interaction_backup():
    """도구 상호작용 백업 예제"""
    print("\n🛠️  도구 상호작용 백업 예제")
    print("-" * 50)
    
    # 도구 정의
    def get_time():
        """현재 시간 반환"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def calculate(expression: str):
        """수식 계산"""
        try:
            return str(eval(expression))
        except:
            return "계산 오류"
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "현재 시간을 가져옵니다"
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "수식을 계산합니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "계산할 수식"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    # 백업과 함께 LLM 생성
    backup = InMemoryHistoryBackup()
    llm = LLM.create("gpt-4o-mini", history_backup=backup)
    
    # 도구를 사용하는 대화
    print("💬 도구를 사용하는 대화 시작\n")
    
    conversation_id = "tool_demo"
    questions = [
        "현재 시간이 몇 시인가요?",
        "1234 * 5678을 계산해주세요.",
        "방금 계산한 결과를 1000으로 나누면?"
    ]
    
    messages = []
    for question in questions:
        print(f"\n👤 User: {question}")
        
        user_msg = Message(role="user", content=question)
        messages.append(user_msg)
        
        # 도구와 함께 응답
        reply = llm.ask_with_tools(
            question,
            tools=tools,
            tool_functions={
                "get_time": get_time,
                "calculate": calculate
            }
        )
        
        print(f"🤖 AI: {reply.text}")
        
        if reply.tool_calls:
            print("📋 사용된 도구:")
            for tc in reply.tool_calls:
                print(f"  - {tc.name}({tc.arguments})")
        
        # 응답 메시지 생성 (도구 호출 정보 포함)
        ai_msg = Message(
            role="assistant",
            content=reply.text,
            tool_calls=reply.tool_calls  # 도구 호출 정보 저장
        )
        messages.append(ai_msg)
        
        # 백업에 저장
        backup.save_messages(conversation_id, messages)
    
    # 복원 및 도구 호출 정보 확인
    print("\n\n📥 백업에서 복원 및 도구 사용 분석:")
    restored = backup.load_messages(conversation_id)
    
    tool_usage_count = 0
    for msg in restored:
        if msg.role == "assistant" and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_usage_count += len(msg.tool_calls)
            print(f"\n메시지: {msg.content[:50]}...")
            print("사용된 도구:")
            for tc in msg.tool_calls:
                print(f"  - {tc.name}")
    
    print(f"\n📊 총 도구 사용 횟수: {tool_usage_count}")


def main():
    """History Backup 예제 메인 함수"""
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        return
    
    print("💾 History Backup 예제")
    print("=" * 50)
    
    try:
        # 1. 메모리 백업
        example_memory_backup()
        
        # 2. SQLite 백업
        example_sqlite_backup()
        
        # 3. 고급 백업 기능
        example_advanced_backup()
        
        # 4. 도구 상호작용 백업
        example_tool_interaction_backup()
        
        print("\n✅ 모든 History Backup 예제 완료!")
        
        # 정리
        print("\n🧹 정리 중...")
        # SQLite 파일 삭제
        if Path("chat_history.db").exists():
            Path("chat_history.db").unlink()
            print("  - chat_history.db 삭제됨")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()