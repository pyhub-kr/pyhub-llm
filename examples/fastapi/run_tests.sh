#!/bin/bash
# FastAPI 테스트 실행 스크립트

# 현재 디렉토리 저장
ORIGINAL_DIR=$(pwd)

# FastAPI 예제 디렉토리로 이동
cd "$(dirname "$0")"

# Django 설정 환경 변수 해제
unset DJANGO_SETTINGS_MODULE

# 환경 변수 설정
export OPENAI_API_KEY="test-key"
export ALLOWED_API_KEYS="test-key-1,test-key-2,demo-key-12345"
export RATE_LIMIT_REQUESTS="10000"

# 테스트 실행
python -m pytest tests/ -v "$@"

# 원래 디렉토리로 돌아가기
cd "$ORIGINAL_DIR"