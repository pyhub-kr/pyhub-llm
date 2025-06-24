"""
FastAPI 테스트 공통 설정
"""

import pytest
import os
import sys

# Django가 로드되지 않도록 방지
if 'django' in sys.modules:
    del sys.modules['django']

# 환경변수 설정
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["ALLOWED_API_KEYS"] = "test-key-1,test-key-2,demo-key-12345"

# Rate limit을 테스트용으로 높게 설정
os.environ["RATE_LIMIT_REQUESTS"] = "10000"

# Django 설정 비활성화
os.environ.pop('DJANGO_SETTINGS_MODULE', None)