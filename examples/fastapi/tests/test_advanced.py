"""
FastAPI 고급 서비스 테스트

advanced.py의 고급 FastAPI 애플리케이션을 테스트합니다.
"""

import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

# 환경변수 설정 (테스트 전에 필요)
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["ALLOWED_API_KEYS"] = "test-key-1,test-key-2,demo-key-12345"

from advanced import app
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
def auth_headers():
    """인증 헤더"""
    return {"Authorization": "Bearer demo-key-12345"}


# =============================================================================
# 인증 테스트
# =============================================================================

def test_protected_endpoint_without_auth(client):
    """인증 없이 보호된 엔드포인트 접근 테스트"""
    response = client.post("/api/chat", json={
        "message": "Test message",
        "model": "gpt-4o-mini"
    })
    assert response.status_code == 401
    assert "Authentication required" in response.json()["detail"]


def test_protected_endpoint_with_invalid_auth(client):
    """잘못된 인증으로 보호된 엔드포인트 접근 테스트"""
    headers = {"Authorization": "Bearer invalid-key"}
    response = client.post("/api/chat", json={
        "message": "Test message",
        "model": "gpt-4o-mini"
    }, headers=headers)
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


@patch('advanced.get_llm_instance')
def test_protected_endpoint_with_valid_auth(mock_get_llm, client, mock_llm, auth_headers):
    """올바른 인증으로 보호된 엔드포인트 접근 테스트"""
    mock_get_llm.return_value = mock_llm
    
    response = client.post("/api/chat", json={
        "message": "Test message",
        "model": "gpt-4o-mini"
    }, headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Test response"


# =============================================================================
# 속도 제한 테스트
# =============================================================================

@patch('advanced.get_llm_instance')
def test_rate_limiting(mock_get_llm, client, mock_llm, auth_headers):
    """속도 제한 테스트"""
    mock_get_llm.return_value = mock_llm
    
    # 첫 번째 요청은 성공
    response = client.post("/api/chat", json={
        "message": "Test 1",
        "model": "gpt-4o-mini"
    }, headers=auth_headers)
    assert response.status_code == 200
    
    # 100개의 요청을 빠르게 보내서 제한에 걸리게 함
    for i in range(105):  # 기본 제한보다 많이
        response = client.post("/api/chat", json={
            "message": f"Test {i+2}",
            "model": "gpt-4o-mini"
        }, headers=auth_headers)
    
    # 마지막 요청은 제한에 걸려야 함
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]


# =============================================================================
# 번역 서비스 테스트
# =============================================================================

@patch('advanced.translation_service.translate')
def test_translate_endpoint(mock_translate, client, auth_headers):
    """번역 엔드포인트 테스트"""
    mock_translate.return_value = {
        "original_text": "Hello",
        "translated_text": "안녕하세요",
        "source_language": "en",
        "target_language": "ko",
        "model": "gpt-4o-mini"
    }
    
    response = client.post("/api/translate", json={
        "text": "Hello",
        "target_language": "ko",
        "model": "gpt-4o-mini"
    }, headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == "Hello"
    assert data["translated_text"] == "안녕하세요"
    assert data["target_language"] == "ko"


def test_translate_endpoint_validation(client, auth_headers):
    """번역 엔드포인트 유효성 검사 테스트"""
    # 빈 텍스트
    response = client.post("/api/translate", json={
        "text": "",
        "target_language": "ko"
    }, headers=auth_headers)
    assert response.status_code == 422
    
    # 긴 텍스트 (5000자 제한)
    response = client.post("/api/translate", json={
        "text": "a" * 5001,
        "target_language": "ko"
    }, headers=auth_headers)
    assert response.status_code == 422


# =============================================================================
# 요약 서비스 테스트
# =============================================================================

@patch('advanced.translation_service.summarize')
def test_summarize_endpoint(mock_summarize, client, auth_headers):
    """요약 엔드포인트 테스트"""
    mock_summarize.return_value = {
        "summary": "테스트 요약입니다.",
        "original_length": 100,
        "summary_length": 10,
        "compression_ratio": 0.1,
        "language": "ko",
        "model": "gpt-4o-mini"
    }
    
    response = client.post("/api/summarize", json={
        "text": "긴 텍스트입니다." * 20,
        "max_length": 100,
        "language": "ko",
        "model": "gpt-4o-mini"
    }, headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "테스트 요약입니다."
    assert data["compression_ratio"] == 0.1


def test_summarize_endpoint_validation(client, auth_headers):
    """요약 엔드포인트 유효성 검사 테스트"""
    # 짧은 텍스트 (50자 미만)
    response = client.post("/api/summarize", json={
        "text": "짧은 텍스트",
        "model": "gpt-4o-mini"
    }, headers=auth_headers)
    assert response.status_code == 422
    
    # 너무 긴 텍스트 (10000자 초과)
    response = client.post("/api/summarize", json={
        "text": "a" * 10001,
        "model": "gpt-4o-mini"
    }, headers=auth_headers)
    assert response.status_code == 422


# =============================================================================
# 지원 언어 테스트
# =============================================================================

def test_supported_languages_endpoint(client, auth_headers):
    """지원 언어 엔드포인트 테스트"""
    response = client.get("/api/supported-languages", headers=auth_headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "languages" in data
    assert "total_count" in data
    assert "premium_user" in data
    assert isinstance(data["languages"], list)
    assert len(data["languages"]) > 0


# =============================================================================
# 관리자 엔드포인트 테스트
# =============================================================================

def test_admin_stats_endpoint(client, auth_headers):
    """관리자 통계 엔드포인트 테스트"""
    response = client.get("/api/admin/stats", headers=auth_headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "total_requests" in data
    assert "active_users" in data
    assert "rate_limit_hits" in data
    assert "avg_response_time" in data


def test_admin_health_endpoint(client, auth_headers):
    """관리자 헬스체크 엔드포인트 테스트"""
    response = client.get("/api/admin/health", headers=auth_headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "llm_providers" in data["components"]
    assert "cache" in data["components"]
    assert "rate_limiter" in data["components"]


# =============================================================================
# 백그라운드 작업 테스트
# =============================================================================

@patch('advanced.get_llm_instance')
def test_background_batch_endpoint(mock_get_llm, client, mock_llm, auth_headers):
    """백그라운드 배치 처리 엔드포인트 테스트"""
    mock_get_llm.return_value = mock_llm
    mock_llm.batch.return_value = [
        Reply(text="Response 1", usage=Usage(input=5, output=10))
    ]
    
    response = client.post("/api/batch/background", json={
        "messages": ["Test message"],
        "model": "gpt-4o-mini",
        "callback_url": "https://example.com/callback"
    }, headers=auth_headers)
    
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "accepted"
    assert "estimated_completion" in data


# =============================================================================
# 오류 처리 테스트
# =============================================================================

@patch('advanced.translation_service.translate')
def test_translate_service_error(mock_translate, client, auth_headers):
    """번역 서비스 오류 처리 테스트"""
    mock_translate.side_effect = Exception("Translation service error")
    
    response = client.post("/api/translate", json={
        "text": "Hello",
        "target_language": "ko"
    }, headers=auth_headers)
    
    assert response.status_code == 500
    data = response.json()
    assert "번역 처리 중 오류가 발생했습니다" in data["detail"]


@patch('advanced.translation_service.summarize')
def test_summarize_service_error(mock_summarize, client, auth_headers):
    """요약 서비스 오류 처리 테스트"""
    mock_summarize.side_effect = Exception("Summarization service error")
    
    response = client.post("/api/summarize", json={
        "text": "긴 텍스트입니다." * 20,
        "model": "gpt-4o-mini"
    }, headers=auth_headers)
    
    assert response.status_code == 500
    data = response.json()
    assert "요약 처리 중 오류가 발생했습니다" in data["detail"]


# =============================================================================
# 성능 테스트
# =============================================================================

@patch('advanced.get_llm_instance')
def test_concurrent_requests(mock_get_llm, client, mock_llm, auth_headers):
    """동시 요청 처리 테스트"""
    import concurrent.futures
    import time
    
    mock_get_llm.return_value = mock_llm
    
    def make_request():
        return client.post("/api/chat", json={
            "message": "Test message",
            "model": "gpt-4o-mini"
        }, headers=auth_headers)
    
    start_time = time.time()
    
    # 10개의 동시 요청
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    end_time = time.time()
    
    # 모든 요청이 성공해야 함
    success_count = sum(1 for r in responses if r.status_code == 200)
    assert success_count >= 8  # 최소 80% 성공률
    
    # 전체 시간이 합리적이어야 함 (순차 처리보다 빨라야 함)
    assert end_time - start_time < 30  # 30초 이내