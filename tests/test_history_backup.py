"""히스토리 백업 시스템 테스트"""

from unittest.mock import MagicMock, patch

import pytest

from pyhub.llm import LLM
from pyhub.llm.history import InMemoryHistoryBackup
from pyhub.llm.types import Message, Usage


class TestInMemoryHistoryBackup:
    """메모리 기반 히스토리 백업 테스트"""

    def test_save_and_load_exchange(self):
        """대화 저장 및 로드 테스트"""
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")

        # 대화 저장
        user_msg = Message(role="user", content="안녕하세요")
        assistant_msg = Message(role="assistant", content="안녕하세요! 무엇을 도와드릴까요?")
        usage = Usage(input=10, output=20)

        backup.save_exchange(user_msg, assistant_msg, usage=usage, model="gpt-4o-mini")

        # 로드 확인
        messages = backup.load_history()
        assert len(messages) == 2
        assert messages[0].content == "안녕하세요"
        assert messages[1].content == "안녕하세요! 무엇을 도와드릴까요?"

    def test_load_history_with_limit(self):
        """제한된 개수만 로드하는 테스트"""
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")

        # 여러 대화 저장
        for i in range(5):
            user_msg = Message(role="user", content=f"질문 {i}")
            assistant_msg = Message(role="assistant", content=f"답변 {i}")
            backup.save_exchange(user_msg, assistant_msg)

        # 최근 2개 대화만 로드 (4개 메시지)
        messages = backup.load_history(limit=4)
        assert len(messages) == 4
        assert messages[0].content == "질문 3"
        assert messages[-1].content == "답변 4"

    def test_usage_summary(self):
        """사용량 합계 테스트"""
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")

        # 여러 대화 저장
        for i in range(3):
            user_msg = Message(role="user", content=f"질문 {i}")
            assistant_msg = Message(role="assistant", content=f"답변 {i}")
            usage = Usage(input=10 + i, output=20 + i)
            backup.save_exchange(user_msg, assistant_msg, usage=usage)

        # 총 사용량 확인
        total_usage = backup.get_usage_summary()
        assert total_usage.input == 33  # 10 + 11 + 12
        assert total_usage.output == 63  # 20 + 21 + 22

    def test_clear(self):
        """메시지 삭제 테스트"""
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")

        # 대화 저장
        user_msg = Message(role="user", content="테스트")
        assistant_msg = Message(role="assistant", content="응답")
        backup.save_exchange(user_msg, assistant_msg)

        # 삭제
        count = backup.clear()
        assert count == 2
        assert len(backup.load_history()) == 0

    def test_file_paths_handling(self):
        """파일 경로 처리 테스트"""
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")

        # 파일이 포함된 메시지
        user_msg = Message(role="user", content="이 파일을 봐주세요", files=["image.png", "/path/to/document.pdf"])
        assistant_msg = Message(role="assistant", content="파일을 확인했습니다.")

        backup.save_exchange(user_msg, assistant_msg)

        # 로드 확인
        messages = backup.load_history()
        assert messages[0].files == ["image.png", "/path/to/document.pdf"]

    def test_tool_interactions_handling(self):
        """도구 호출 정보 처리 테스트"""
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")

        # 도구 호출이 포함된 응답
        user_msg = Message(role="user", content="25도를 화씨로 변환해줘")
        assistant_msg = Message(
            role="assistant",
            content="25°C는 77°F입니다.",
            tool_interactions=[
                {"tool": "convert_temperature", "arguments": {"value": 25, "from": "C", "to": "F"}, "result": "77°F"}
            ],
        )

        backup.save_exchange(user_msg, assistant_msg)

        # 로드 확인
        messages = backup.load_history()
        assert messages[1].tool_interactions is not None
        assert len(messages[1].tool_interactions) == 1
        assert messages[1].tool_interactions[0]["tool"] == "convert_temperature"

    def test_multiple_tool_interactions(self):
        """여러 도구 호출이 포함된 경우 테스트"""
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")

        # 여러 도구 호출이 포함된 응답
        user_msg = Message(role="user", content="서울과 도쿄의 날씨와 환율을 알려줘")
        assistant_msg = Message(
            role="assistant",
            content="서울은 맑음 25°C, 도쿄는 흐림 22°C입니다. 현재 환율은 1달러 = 1300원, 1달러 = 150엔입니다.",
            tool_interactions=[
                {"tool": "get_weather", "arguments": {"city": "Seoul"}, "result": "맑음, 25°C"},
                {"tool": "get_weather", "arguments": {"city": "Tokyo"}, "result": "흐림, 22°C"},
                {"tool": "get_exchange_rate", "arguments": {"from": "USD", "to": "KRW"}, "result": 1300},
                {"tool": "get_exchange_rate", "arguments": {"from": "USD", "to": "JPY"}, "result": 150},
            ],
        )

        backup.save_exchange(user_msg, assistant_msg)

        # 로드 확인
        messages = backup.load_history()
        assert messages[1].tool_interactions is not None
        assert len(messages[1].tool_interactions) == 4
        assert messages[1].tool_interactions[0]["tool"] == "get_weather"
        assert messages[1].tool_interactions[2]["tool"] == "get_exchange_rate"


class TestLLMWithHistoryBackup:
    """LLM과 히스토리 백업 통합 테스트"""

    @patch("pyhub.llm.openai.OpenAILLM._make_ask")
    def test_llm_with_backup(self, mock_ask):
        """백업이 설정된 LLM 테스트"""
        # Mock 설정
        mock_ask.return_value = MagicMock(text="안녕하세요! 무엇을 도와드릴까요?", usage=Usage(input=10, output=20))

        # 백업과 함께 LLM 생성
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")
        llm = LLM.create("gpt-4o-mini", history_backup=backup)

        # 대화
        _reply = llm.ask("안녕하세요")

        # 백업 확인
        messages = backup.load_history()
        assert len(messages) == 2
        assert messages[0].content == "안녕하세요"
        assert messages[1].content == "안녕하세요! 무엇을 도와드릴까요?"

        # usage가 저장되지 않는 것은 현재 구현의 한계
        # TODO: _update_history에서 usage 정보를 전달받도록 개선 필요

    @patch("pyhub.llm.openai.OpenAILLM._make_ask")
    def test_llm_history_restoration(self, mock_ask):
        """히스토리 복원 테스트"""
        # 기존 대화가 있는 백업
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")

        # 미리 대화 저장
        old_user = Message(role="user", content="이전 질문")
        old_assistant = Message(role="assistant", content="이전 답변")
        backup.save_exchange(old_user, old_assistant)

        # 백업에서 복원하여 LLM 생성
        llm = LLM.create("gpt-4o-mini", history_backup=backup)

        # 히스토리가 복원되었는지 확인
        assert len(llm.history) == 2
        assert llm.history[0].content == "이전 질문"
        assert llm.history[1].content == "이전 답변"

    @patch("pyhub.llm.openai.OpenAILLM._make_ask")
    def test_backup_failure_handling(self, mock_ask):
        """백업 실패 시에도 정상 동작하는지 테스트"""
        # Mock 설정
        mock_ask.return_value = MagicMock(text="응답", usage=Usage(input=5, output=10))

        # 실패하는 백업 생성
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")
        backup.save_exchange = MagicMock(side_effect=Exception("Backup failed"))

        # 백업과 함께 LLM 생성
        with patch("pyhub.llm.base.logger") as mock_logger:
            llm = LLM.create("gpt-4o-mini", history_backup=backup)

            # 대화 (백업은 실패하지만 정상 동작해야 함)
            reply = llm.ask("테스트")

            # 응답은 정상
            assert reply.text == "응답"

            # 경고 로그 확인
            mock_logger.warning.assert_called()
            assert "History backup failed" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    @patch("pyhub.llm.openai.OpenAILLM._make_ask_async")
    async def test_async_llm_with_backup(self, mock_ask_async):
        """비동기 LLM과 백업 테스트"""
        # Mock 설정
        mock_ask_async.return_value = MagicMock(text="비동기 응답", usage=Usage(input=15, output=25))

        # 백업과 함께 LLM 생성
        backup = InMemoryHistoryBackup(user_id="test_user", session_id="test_session")
        llm = await LLM.create_async("gpt-4o-mini", history_backup=backup)

        # 비동기 대화
        _reply = await llm.ask_async("비동기 테스트")

        # 백업 확인
        messages = backup.load_history()
        assert len(messages) == 2
        assert messages[0].content == "비동기 테스트"
        assert messages[1].content == "비동기 응답"


class TestSQLAlchemyHistoryBackup:
    """SQLAlchemy 백업 테스트 (SQLAlchemy 설치 시에만)"""

    @pytest.mark.skipif("not HAS_SQLALCHEMY", reason="SQLAlchemy not installed")
    def test_sqlalchemy_backup(self):
        """SQLAlchemy 백업 기본 동작 테스트"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from pyhub.llm.history import SQLAlchemyHistoryBackup

        # 메모리 DB 생성
        engine = create_engine("sqlite:///:memory:")
        from pyhub.llm.history.sqlalchemy_backup import Base

        Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        # 백업 생성
        backup = SQLAlchemyHistoryBackup(session, user_id="test", session_id="test_session")

        # 대화 저장
        user_msg = Message(role="user", content="SQL 테스트")
        assistant_msg = Message(role="assistant", content="SQL 응답")
        backup.save_exchange(user_msg, assistant_msg, usage=Usage(input=5, output=10))

        # 로드 확인
        messages = backup.load_history()
        assert len(messages) == 2
        assert messages[0].content == "SQL 테스트"

        # 사용량 확인
        usage = backup.get_usage_summary()
        assert usage.input == 5
        assert usage.output == 10

        session.close()


# SQLAlchemy 설치 여부 확인
try:
    import sqlalchemy  # noqa: F401

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# pytest에 변수 등록
pytest.HAS_SQLALCHEMY = HAS_SQLALCHEMY
