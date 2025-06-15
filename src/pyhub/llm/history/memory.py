"""메모리 기반 히스토리 백업 구현 (테스트용)"""

from datetime import datetime
from typing import Optional

from pyhub.llm.history.base import HistoryBackup
from pyhub.llm.types import Message, Usage


class InMemoryHistoryBackup(HistoryBackup):
    """메모리에 히스토리를 저장하는 백업 구현

    테스트 및 개발 용도로 사용됩니다.
    """

    def __init__(self, user_id: str, session_id: str):
        """
        Args:
            user_id: 사용자 ID
            session_id: 세션 ID
        """
        self.user_id = user_id
        self.session_id = session_id
        self.messages: list[tuple[Message, dict]] = []  # (메시지, 메타데이터) 쌍

    def save_exchange(
        self, user_msg: Message, assistant_msg: Message, usage: Optional[Usage] = None, model: Optional[str] = None
    ) -> None:
        """대화 쌍을 메모리에 저장"""
        timestamp = datetime.utcnow()

        # User 메시지 저장
        user_metadata = {"timestamp": timestamp, "user_id": self.user_id, "session_id": self.session_id, "role": "user"}
        self.messages.append((user_msg, user_metadata))

        # Assistant 메시지 저장
        assistant_metadata = {
            "timestamp": timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": "assistant",
            "usage": usage,
            "model": model,
        }
        self.messages.append((assistant_msg, assistant_metadata))

    def load_history(self, limit: Optional[int] = None) -> list[Message]:
        """저장된 메시지 목록 반환"""
        messages = [msg for msg, _ in self.messages]

        if limit is not None:
            # 최근 메시지부터 limit개 반환
            return messages[-limit:] if len(messages) > limit else messages

        return messages

    def get_usage_summary(self) -> Usage:
        """총 사용량 계산"""
        total_usage = Usage(input=0, output=0)

        for _, metadata in self.messages:
            if metadata.get("role") == "assistant" and metadata.get("usage"):
                usage = metadata["usage"]
                total_usage.input += usage.input
                total_usage.output += usage.output

        return total_usage

    def clear(self) -> int:
        """모든 메시지 삭제"""
        count = len(self.messages)
        self.messages.clear()
        return count

    # 추가 유틸리티 메서드
    def get_messages_with_metadata(self) -> list[tuple[Message, dict]]:
        """메타데이터와 함께 메시지 반환 (테스트용)"""
        return self.messages.copy()

    def get_session_info(self) -> dict:
        """세션 정보 반환"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "total_usage": self.get_usage_summary(),
        }
