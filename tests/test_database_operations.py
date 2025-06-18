"""
데이터베이스 작업 개선 테스트
타임스탬프 보존 등 DB 관련 개선사항 검증
"""

import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime

# 테스트를 위해 src 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pyhub.llm.types import Message


def test_conversation_timestamp_preservation():
    """대화 타임스탬프 보존 테스트"""

    # 임시 데이터베이스 생성
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # 테스트용 SQLite 백업 클래스 구현
        class TestSQLiteBackup:
            def __init__(self, db_path: str):
                self.db_path = db_path
                self._init_db()

            def _init_db(self):
                """데이터베이스 초기화"""
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # 테이블 생성
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                    )
                """
                )

                conn.commit()
                conn.close()

            def save_messages(self, conversation_id: str, messages):
                """메시지 저장 (개선된 버전)"""
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # 대화 생성/업데이트 (created_at 보존)
                current_time = datetime.now()
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO conversations (id, created_at, updated_at)
                    VALUES (?, ?, ?)
                """,
                    (conversation_id, current_time, current_time),
                )

                # 기존 대화가 있으면 updated_at만 업데이트
                cursor.execute(
                    """
                    UPDATE conversations 
                    SET updated_at = ?
                    WHERE id = ?
                """,
                    (current_time, conversation_id),
                )

                # 기존 메시지 삭제 (간단한 구현)
                cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))

                # 새 메시지 저장
                for msg in messages:
                    cursor.execute(
                        """
                        INSERT INTO messages (conversation_id, role, content, timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (conversation_id, msg.role, msg.content, current_time, "{}"),
                    )

                conn.commit()
                conn.close()

            def get_conversation_timestamps(self, conversation_id: str):
                """대화 타임스탬프 조회"""
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("SELECT created_at, updated_at FROM conversations WHERE id = ?", (conversation_id,))
                result = cursor.fetchone()
                conn.close()

                if result:
                    return {
                        "created_at": datetime.fromisoformat(result[0]),
                        "updated_at": datetime.fromisoformat(result[1]),
                    }
                return None

        # 테스트 시작
        backup = TestSQLiteBackup(db_path)
        conversation_id = "test_conversation"

        # 첫 번째 메시지 저장
        messages1 = [Message(role="user", content="안녕하세요")]
        backup.save_messages(conversation_id, messages1)

        # 첫 번째 타임스탬프 기록
        first_timestamps = backup.get_conversation_timestamps(conversation_id)
        assert first_timestamps is not None

        # 잠시 대기 (타임스탬프 차이를 위해)
        time.sleep(0.1)

        # 두 번째 메시지 추가
        messages2 = [
            Message(role="user", content="안녕하세요"),
            Message(role="assistant", content="안녕하세요! 도움이 필요하시면 말씀해주세요."),
        ]
        backup.save_messages(conversation_id, messages2)

        # 두 번째 타임스탬프 기록
        second_timestamps = backup.get_conversation_timestamps(conversation_id)
        assert second_timestamps is not None

        # created_at은 보존되어야 하고, updated_at은 업데이트되어야 함
        assert (
            first_timestamps["created_at"] == second_timestamps["created_at"]
        ), "created_at 타임스탬프가 보존되지 않았습니다"

        assert (
            second_timestamps["updated_at"] >= first_timestamps["updated_at"]
        ), "updated_at 타임스탬프가 업데이트되지 않았습니다"

        print("✅ 대화 타임스탬프 보존 테스트 통과")

    finally:
        # 임시 파일 정리
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_insert_or_ignore_behavior():
    """INSERT OR IGNORE + UPDATE 패턴 테스트"""

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 테스트 테이블 생성
        cursor.execute(
            """
            CREATE TABLE test_conversations (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """
        )

        conversation_id = "test_conv"

        # 첫 번째 삽입
        time1 = datetime.now()
        cursor.execute(
            "INSERT OR IGNORE INTO test_conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
            (conversation_id, time1, time1),
        )

        time.sleep(0.1)

        # 두 번째 시도 - INSERT OR IGNORE는 실행되지 않아야 함
        time2 = datetime.now()
        cursor.execute(
            "INSERT OR IGNORE INTO test_conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
            (conversation_id, time2, time2),
        )

        # UPDATE는 실행되어야 함
        cursor.execute("UPDATE test_conversations SET updated_at = ? WHERE id = ?", (time2, conversation_id))

        # 결과 확인
        cursor.execute("SELECT created_at, updated_at FROM test_conversations WHERE id = ?", (conversation_id,))
        result = cursor.fetchone()

        created_at = datetime.fromisoformat(result[0])
        updated_at = datetime.fromisoformat(result[1])

        # created_at은 첫 번째 시간, updated_at은 두 번째 시간이어야 함
        assert abs((created_at - time1).total_seconds()) < 0.01, "created_at이 첫 번째 시간과 일치하지 않습니다"

        assert abs((updated_at - time2).total_seconds()) < 0.01, "updated_at이 두 번째 시간과 일치하지 않습니다"

        conn.close()
        print("✅ INSERT OR IGNORE + UPDATE 패턴 테스트 통과")

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    print("🗄️ 데이터베이스 작업 개선 테스트 시작...")

    try:
        test_conversation_timestamp_preservation()
    except Exception as e:
        print(f"❌ 대화 타임스탬프 보존 테스트 실패: {e}")

    try:
        test_insert_or_ignore_behavior()
    except Exception as e:
        print(f"❌ INSERT OR IGNORE + UPDATE 패턴 테스트 실패: {e}")

    print("🗄️ 데이터베이스 작업 개선 테스트 완료")
