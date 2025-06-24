"""
FastAPI 기본 서비스 테스트

main.py의 기본 FastAPI 애플리케이션을 테스트합니다.
"""

import pytest
import os
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

# 환경변수 설정 (테스트 전에 필요)
os.environ["OPENAI_API_KEY"] = "test-key"

from main import app
from pyhub.llm.types import Reply, Usage


# =============================================================================
# 테스트 설정
# =============================================================================

@pytest.fixture
def client():
    """테스트 클라이언트 생성"""
    return TestClient(app)


@pytest.fixture
def mock_llm():
    """Mock LLM 인스턴스"""
    mock = AsyncMock()
    mock.ask_async.return_value = Reply(
        text="Test response",
        usage=Usage(input=10, output=20)
    )
    return mock


@pytest.fixture
def mock_batch_responses():
    """Mock 배치 응답"""
    return [
        Reply(text="Response 1", usage=Usage(input=5, output=10)),
        Reply(text="Response 2", usage=Usage(input=6, output=12)),
        Reply(text="Response 3", usage=Usage(input=7, output=14))
    ]


# =============================================================================
# 기본 엔드포인트 테스트
# =============================================================================

def test_root_endpoint(client):
    """루트 엔드포인트 테스트"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "service" in data
    assert "endpoints" in data
    assert data["service"] == "pyhub-llm FastAPI Integration"


def test_health_endpoint(client):
    """헬스체크 엔드포인트 테스트"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "active_sessions" in data


# =============================================================================
# 채팅 엔드포인트 테스트
# =============================================================================

@patch('main.get_llm_instance')
def test_chat_endpoint(mock_get_llm, client, mock_llm):
    """채팅 엔드포인트 테스트"""
    mock_get_llm.return_value = mock_llm
    
    response = client.post("/chat", json={
        "message": "Hello, world!",
        "model": "gpt-4o-mini"
    })
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["response"] == "Test response"
    assert data["model"] == "gpt-4o-mini"
    assert data["usage"]["input"] == 10
    assert data["usage"]["output"] == 20
    
    # Mock 호출 확인
    mock_llm.ask_async.assert_called_once()


def test_chat_endpoint_with_options(client, mock_llm):
    """옵션이 포함된 채팅 엔드포인트 테스트"""
    with patch('main.get_llm_instance', return_value=mock_llm):
        response = client.post("/chat", json={
            "message": "Test message",
            "model": "gpt-4o-mini",
            "system_prompt": "You are a helpful assistant",
            "temperature": 0.7,
            "max_tokens": 100
        })
        
        assert response.status_code == 200
        
        # Mock 호출 확인 (파라미터 포함)
        mock_llm.ask_async.assert_called_once()
        call_args = mock_llm.ask_async.call_args
        assert call_args.kwargs["input"] == "Test message"
        assert call_args.kwargs["system_prompt"] == "You are a helpful assistant"
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["max_tokens"] == 100


def test_chat_endpoint_validation_error(client):
    """채팅 엔드포인트 유효성 검사 오류 테스트"""
    # 빈 메시지
    response = client.post("/chat", json={
        "message": "",
        "model": "gpt-4o-mini"
    })
    assert response.status_code == 422
    
    # 필수 필드 누락
    response = client.post("/chat", json={
        "model": "gpt-4o-mini"
    })
    assert response.status_code == 422


# =============================================================================
# 배치 처리 엔드포인트 테스트
# =============================================================================

@patch('main.get_llm_instance')
def test_batch_endpoint(mock_get_llm, client, mock_llm, mock_batch_responses):
    """배치 처리 엔드포인트 테스트"""
    mock_get_llm.return_value = mock_llm
    mock_llm.batch.return_value = mock_batch_responses
    
    response = client.post("/batch", json={
        "messages": ["Message 1", "Message 2", "Message 3"],
        "model": "gpt-4o-mini",
        "max_parallel": 2
    })
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_count"] == 3
    assert data["success_count"] == 3
    assert len(data["responses"]) == 3
    assert "execution_time" in data
    
    # 첫 번째 응답 확인
    assert data["responses"][0]["response"] == "Response 1"
    assert data["responses"][0]["model"] == "gpt-4o-mini"


def test_batch_endpoint_empty_messages(client):
    """빈 메시지 배열로 배치 처리 테스트"""
    response = client.post("/batch", json={
        "messages": [],
        "model": "gpt-4o-mini"
    })
    assert response.status_code == 422


def test_batch_endpoint_too_many_messages(client):
    """너무 많은 메시지로 배치 처리 테스트"""
    response = client.post("/batch", json={
        "messages": ["Message"] * 15,  # 최대 10개 제한
        "model": "gpt-4o-mini"
    })
    assert response.status_code == 422


# =============================================================================
# 스트리밍 엔드포인트 테스트
# =============================================================================

@patch('main.get_llm_instance')
def test_stream_endpoint(mock_get_llm, client, mock_llm):
    """스트리밍 엔드포인트 테스트"""
    # 스트리밍 응답 Mock
    async def mock_stream():
        yield Reply(text="Hello ")
        yield Reply(text="World!")
        yield Reply(text="")
    
    mock_get_llm.return_value = mock_llm
    mock_llm.ask_stream_async.return_value = mock_stream()
    
    response = client.post("/stream", json={
        "message": "Hello",
        "model": "gpt-4o-mini"
    })
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    
    # 스트리밍 내용 확인
    content = response.text
    assert "data: Hello " in content
    assert "data: World!" in content
    assert "data: [DONE]" in content


# =============================================================================
# 세션 관리 테스트
# =============================================================================

@patch('main.LLM')
def test_session_chat_endpoint(mock_llm_class, client, mock_llm):
    """세션 기반 채팅 엔드포인트 테스트"""
    mock_llm_class.create.return_value = mock_llm
    
    # 첫 번째 세션 메시지
    response = client.post("/chat/session", json={
        "message": "Hello",
        "session_id": "test_session",
        "model": "gpt-4o-mini"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Test response"
    
    # 세션이 생성되었는지 확인
    mock_llm_class.create.assert_called_once_with("gpt-4o-mini")
    mock_llm.ask_async.assert_called_once()


def test_clear_session_endpoint(client):
    """세션 삭제 엔드포인트 테스트"""
    # 먼저 세션 생성
    with patch('main.LLM') as mock_llm_class:
        mock_llm_class.create.return_value = AsyncMock()
        client.post("/chat/session", json={
            "message": "Hello",
            "session_id": "test_session", 
            "model": "gpt-4o-mini"
        })
    
    # 세션 삭제
    response = client.delete("/chat/session/test_session")
    assert response.status_code == 200
    
    data = response.json()
    assert "test_session" in data["message"]


def test_clear_nonexistent_session(client):
    """존재하지 않는 세션 삭제 테스트"""
    response = client.delete("/chat/session/nonexistent_session")
    assert response.status_code == 404


def test_list_sessions_endpoint(client):
    """세션 목록 조회 테스트"""
    response = client.get("/sessions")
    assert response.status_code == 200
    
    data = response.json()
    assert "total_sessions" in data
    assert "sessions" in data
    assert isinstance(data["sessions"], dict)


# =============================================================================
# 오류 처리 테스트
# =============================================================================

@patch('main.get_llm_instance')
def test_chat_endpoint_llm_error(mock_get_llm, client):
    """LLM 오류 발생 시 테스트"""
    mock_llm = AsyncMock()
    mock_llm.ask_async.side_effect = Exception("LLM Error")
    mock_get_llm.return_value = mock_llm
    
    response = client.post("/chat", json={
        "message": "Test message",
        "model": "gpt-4o-mini"
    })
    
    assert response.status_code == 500
    data = response.json()
    assert "LLM 처리 중 오류가 발생했습니다" in data["detail"]


def test_invalid_model_error(client):
    """지원하지 않는 모델 오류 테스트"""
    with patch('main.get_llm_instance', side_effect=Exception("Unsupported model")):
        response = client.post("/chat", json={
            "message": "Test message", 
            "model": "invalid-model"
        })
        
        assert response.status_code == 500


# =============================================================================
# API 키 검증 테스트
# =============================================================================

def test_api_key_missing():
    """API 키 누락 테스트"""
    with patch.dict(os.environ, {}, clear=True):
        with patch('main.validate_api_key') as mock_validate:
            mock_validate.side_effect = Exception("API key not set")
            
            client = TestClient(app)
            response = client.post("/chat", json={
                "message": "Test",
                "model": "gpt-4o-mini"
            })
            
            assert response.status_code == 500