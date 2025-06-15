"""히스토리 백업 인터페이스"""

from abc import ABC, abstractmethod
from typing import Optional

from pyhub.llm.types import Message, Usage


class HistoryBackup(ABC):
    """채팅 히스토리 백업/복원 인터페이스

    메모리의 history를 외부 저장소에 백업하고 복원하는 기능을 제공합니다.
    user_id와 session_id는 생성자에서 받아 인스턴스에 바인딩됩니다.
    """

    @abstractmethod
    def save_exchange(
        self, user_msg: Message, assistant_msg: Message, usage: Optional[Usage] = None, model: Optional[str] = None
    ) -> None:
        """User-Assistant 대화 쌍을 백업합니다.

        Args:
            user_msg: 사용자 메시지
            assistant_msg: 어시스턴트 응답 (tool_interactions 포함 가능)
            usage: 토큰 사용량 정보
            model: 사용된 모델명
        """
        pass

    @abstractmethod
    def load_history(self, limit: Optional[int] = None) -> list[Message]:
        """백업된 히스토리를 복원합니다.

        Args:
            limit: 최근 N개의 메시지만 로드 (None이면 전체)

        Returns:
            Message 객체 리스트 (시간순 정렬)
        """
        pass

    @abstractmethod
    def get_usage_summary(self) -> Usage:
        """현재 세션의 총 토큰 사용량을 조회합니다.

        Returns:
            총 사용량 (input + output)
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """현재 세션의 모든 메시지를 삭제합니다.

        Returns:
            삭제된 메시지 수
        """
        pass

    # Async 버전 (기본 구현은 동기 버전 호출)
    async def save_exchange_async(
        self, user_msg: Message, assistant_msg: Message, usage: Optional[Usage] = None, model: Optional[str] = None
    ) -> None:
        """save_exchange의 비동기 버전"""
        return self.save_exchange(user_msg, assistant_msg, usage, model)

    async def load_history_async(self, limit: Optional[int] = None) -> list[Message]:
        """load_history의 비동기 버전"""
        return self.load_history(limit)

    async def get_usage_summary_async(self) -> Usage:
        """get_usage_summary의 비동기 버전"""
        return self.get_usage_summary()

    async def clear_async(self) -> int:
        """clear의 비동기 버전"""
        return self.clear()
