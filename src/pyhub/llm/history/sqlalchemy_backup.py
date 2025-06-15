"""SQLAlchemy 기반 히스토리 백업 구현"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from pyhub.llm.history.base import HistoryBackup
from pyhub.llm.types import Message, Usage

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# SQLAlchemy는 선택적 의존성이므로 try-except로 처리
try:
    from sqlalchemy import (  # create_engine imported but unused
        JSON,
        Column,
        DateTime,
        Integer,
        String,
        Text,
    )
    from sqlalchemy.ext.declarative import declarative_base

    # from sqlalchemy.orm import Session, sessionmaker  # imported but unused

    Base = declarative_base()

    class ChatHistory(Base):
        """채팅 히스토리 테이블 모델"""

        __tablename__ = "chat_history"

        id = Column(Integer, primary_key=True)
        user_id = Column(String, nullable=False, index=True)
        session_id = Column(String, nullable=False, index=True)
        timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

        # Message fields
        role = Column(String, nullable=False)
        content = Column(Text, nullable=False)
        file_paths = Column(JSON, nullable=True)  # 파일 경로 목록
        tool_interactions = Column(JSON, nullable=True)  # 도구 호출 내역

        # Usage fields (assistant 메시지용)
        usage_input = Column(Integer, nullable=True)
        usage_output = Column(Integer, nullable=True)

        # Model info
        model = Column(String, nullable=True)

except ImportError:
    Base = None
    ChatHistory = None
    Session = Any  # Fallback type for when SQLAlchemy is not installed


class SQLAlchemyHistoryBackup(HistoryBackup):
    """SQLAlchemy를 사용한 히스토리 백업 구현

    사용 예시:
        # 엔진과 세션 생성
        engine = create_engine('sqlite:///chat_history.db')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # 백업 인스턴스 생성
        backup = SQLAlchemyHistoryBackup(session, user_id="user123", session_id="session456")

        # LLM과 함께 사용
        llm = LLM.create("gpt-4o-mini", history_backup=backup)
    """

    def __init__(self, session: "Session", user_id: str, session_id: str):
        """
        Args:
            session: SQLAlchemy 세션
            user_id: 사용자 ID
            session_id: 세션 ID
        """
        if Base is None:
            raise ImportError("SQLAlchemy is not installed. Install with: pip install sqlalchemy")

        self.session = session
        self.user_id = user_id
        self.session_id = session_id

    def save_exchange(
        self, user_msg: Message, assistant_msg: Message, usage: Optional[Usage] = None, model: Optional[str] = None
    ) -> None:
        """대화 쌍을 데이터베이스에 저장"""
        timestamp = datetime.utcnow()

        # User 메시지 저장
        user_record = ChatHistory(
            user_id=self.user_id,
            session_id=self.session_id,
            timestamp=timestamp,
            role=user_msg.role,
            content=user_msg.content,
            file_paths=self._serialize_files(user_msg.files) if user_msg.files else None,
        )
        self.session.add(user_record)

        # Assistant 메시지 저장
        assistant_record = ChatHistory(
            user_id=self.user_id,
            session_id=self.session_id,
            timestamp=timestamp,
            role=assistant_msg.role,
            content=assistant_msg.content,
            tool_interactions=assistant_msg.tool_interactions,  # 이미 dict 형태
            usage_input=usage.input if usage else None,
            usage_output=usage.output if usage else None,
            model=model,
        )
        self.session.add(assistant_record)

        # 커밋
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

    def load_history(self, limit: Optional[int] = None) -> list[Message]:
        """데이터베이스에서 히스토리 로드"""
        query = (
            self.session.query(ChatHistory)
            .filter_by(user_id=self.user_id, session_id=self.session_id)
            .order_by(ChatHistory.timestamp, ChatHistory.id)
        )

        if limit:
            # 최근 N개 메시지 (user-assistant 쌍으로 계산)
            # limit이 10이면 최근 20개 메시지 (10쌍)
            query = query.limit(limit * 2)

        messages = []
        for record in query:
            msg = Message(
                role=record.role,
                content=record.content,
                files=self._deserialize_files(record.file_paths) if record.file_paths else None,
                tool_interactions=record.tool_interactions,
            )
            messages.append(msg)

        return messages

    def get_usage_summary(self) -> Usage:
        """세션의 총 사용량 계산"""
        records = (
            self.session.query(ChatHistory)
            .filter_by(user_id=self.user_id, session_id=self.session_id, role="assistant")
            .all()
        )

        total_usage = Usage(input=0, output=0)
        for record in records:
            if record.usage_input:
                total_usage.input += record.usage_input
            if record.usage_output:
                total_usage.output += record.usage_output

        return total_usage

    def clear(self) -> int:
        """세션의 모든 메시지 삭제"""
        count = self.session.query(ChatHistory).filter_by(user_id=self.user_id, session_id=self.session_id).count()

        self.session.query(ChatHistory).filter_by(user_id=self.user_id, session_id=self.session_id).delete()

        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

        return count

    def _serialize_files(self, files: list) -> list[str]:
        """파일 객체를 저장 가능한 경로 목록으로 변환"""
        paths = []
        for file in files:
            if isinstance(file, str):
                paths.append(file)
            elif hasattr(file, "name"):
                # Path 객체나 파일 객체
                paths.append(str(file))
            else:
                # IO 객체 등은 이름만 저장
                paths.append(f"<IO: {type(file).__name__}>")
        return paths

    def _deserialize_files(self, file_paths: list[str]) -> list[str]:
        """저장된 경로 목록을 반환 (단순히 문자열 목록으로)"""
        return file_paths

    # 추가 유틸리티 메서드
    def get_session_messages_count(self) -> int:
        """세션의 메시지 개수"""
        return self.session.query(ChatHistory).filter_by(user_id=self.user_id, session_id=self.session_id).count()

    def get_all_sessions(self, user_id: Optional[str] = None) -> list[str]:
        """사용자의 모든 세션 ID 목록"""
        query = self.session.query(ChatHistory.session_id).distinct()
        if user_id:
            query = query.filter_by(user_id=user_id)
        else:
            query = query.filter_by(user_id=self.user_id)

        return [row[0] for row in query.all()]
