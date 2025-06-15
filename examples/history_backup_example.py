"""히스토리 백업 시스템 사용 예제
예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

from pyhub.llm import LLM
from pyhub.llm.history import InMemoryHistoryBackup


def main():
    """메모리 기반 히스토리 백업 예제"""
    
    # 1. 백업 스토리지 생성
    backup = InMemoryHistoryBackup(
        user_id="user123",
        session_id="session456"
    )
    
    # 2. 백업과 함께 LLM 생성
    llm = LLM.create("gpt-4o-mini", history_backup=backup)
    
    # 3. 대화 (자동으로 백업됨)
    print("첫 번째 대화:")
    reply1 = llm.ask("안녕하세요! 제 이름은 김철수입니다.")
    print("User: 안녕하세요! 제 이름은 김철수입니다.")
    print(f"Assistant: {reply1.text}\n")
    
    print("두 번째 대화:")
    reply2 = llm.ask("제 이름이 뭐라고 했죠?")
    print("User: 제 이름이 뭐라고 했죠?")
    print(f"Assistant: {reply2.text}\n")
    
    # 4. 백업된 히스토리 확인
    print("=== 백업된 히스토리 ===")
    messages = backup.load_history()
    for i, msg in enumerate(messages):
        print(f"{i+1}. [{msg.role}] {msg.content[:50]}...")
    
    # 5. 세션 정보 확인
    print("\n=== 세션 정보 ===")
    info = backup.get_session_info()
    print(f"User ID: {info['user_id']}")
    print(f"Session ID: {info['session_id']}")
    print(f"총 메시지 수: {info['message_count']}")
    print(f"총 사용량: {info['total_usage']}")


def sqlalchemy_example():
    """SQLAlchemy 백업 예제 (SQLAlchemy 설치 필요)"""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from pyhub.llm.history import SQLAlchemyHistoryBackup
        from pyhub.llm.history.sqlalchemy_backup import Base
        
        # 1. 데이터베이스 설정
        engine = create_engine('sqlite:///chat_history.db')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # 2. SQLAlchemy 백업 생성
        backup = SQLAlchemyHistoryBackup(
            session=session,
            user_id="user123",
            session_id="session789"
        )
        
        # 3. 백업과 함께 LLM 생성
        llm = LLM.create("gpt-4o-mini", history_backup=backup)
        
        # 4. 대화
        reply = llm.ask("SQLAlchemy 백업 테스트입니다.")
        print(f"Reply: {reply.text}")
        
        # 5. 영구 저장된 데이터 확인
        print(f"저장된 메시지 수: {backup.get_session_messages_count()}")
        
        session.close()
        
    except ImportError:
        print("SQLAlchemy가 설치되지 않았습니다.")
        print("설치: pip install sqlalchemy")


def session_restoration_example():
    """세션 복원 예제"""
    
    # 1. 첫 번째 세션
    print("=== 첫 번째 세션 ===")
    backup1 = InMemoryHistoryBackup(user_id="user123", session_id="session001")
    llm1 = LLM.create("gpt-4o-mini", history_backup=backup1)
    
    llm1.ask("저는 파이썬을 배우고 있습니다.")
    llm1.ask("특히 FastAPI에 관심이 많아요.")
    
    # 2. 동일한 백업으로 새 LLM 생성 (세션 복원)
    print("\n=== 세션 복원 ===")
    llm2 = LLM.create("gpt-4o-mini", history_backup=backup1)
    
    # 이전 대화 내용이 복원됨
    print(f"복원된 메시지 수: {len(llm2.history)}")
    
    # 이전 대화를 기억하고 있음
    reply = llm2.ask("제가 어떤 프레임워크에 관심이 있다고 했죠?")
    print(f"Assistant: {reply.text}")


if __name__ == "__main__":
    print("1. 메모리 백업 예제")
    print("-" * 50)
    main()
    
    print("\n\n2. 세션 복원 예제")
    print("-" * 50)
    session_restoration_example()
    
    print("\n\n3. SQLAlchemy 백업 예제")
    print("-" * 50)
    sqlalchemy_example()