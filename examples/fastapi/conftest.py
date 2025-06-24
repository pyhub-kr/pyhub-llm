"""
FastAPI 예제 테스트 설정

FastAPI 모듈을 import 하기 전에 필요한 설정을 수행합니다.
"""

import sys
from pathlib import Path

# FastAPI 예제 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 테스트 시 필요한 환경변수 설정
import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ALLOWED_API_KEYS", "test-key-1,test-key-2,demo-key-12345")