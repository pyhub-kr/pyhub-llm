# FastAPI 예제 테스트 가이드

## 문제점
메인 프로젝트의 테스트 환경 (Django pytest 플러그인 등)이 FastAPI 테스트와 충돌합니다.

## 해결 방법

### 1. 독립적인 가상환경 사용 (권장)
```bash
# FastAPI 예제 디렉토리로 이동
cd examples/fastapi

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx

# 테스트 실행
pytest tests/ -v
```

### 2. 테스트 스크립트 사용
```bash
# 제공된 스크립트 사용
./run_tests.sh
```

### 3. 수동으로 Django 비활성화
```bash
# Django 환경변수 제거 후 테스트
unset DJANGO_SETTINGS_MODULE
cd examples/fastapi
python -m pytest tests/ -v
```

## 수정된 내용

### 언어 통일
- 모든 오류 메시지를 영어로 통일
- 한국어 메시지가 있던 부분 수정

### 테스트 개선
- Mock 설정 수정 (스트리밍, 배치 처리)
- Rate limiting 테스트 격리
- 유효성 검사 개선

### 파일 변경 사항
1. `middleware/auth.py` - 오류 메시지 영어로 변경
2. `middleware/rate_limit.py` - 오류 메시지 영어로 변경
3. `services/translation.py` - 유효성 검사 범위 추가
4. `advanced.py` - 오류 메시지 영어로 변경
5. `tests/test_advanced.py` - Mock 경로 및 assertion 수정
6. `tests/test_main.py` - Mock 설정 및 테스트 로직 수정

## 알려진 이슈

1. **Django 플러그인 충돌**: 메인 프로젝트에서 pytest 실행 시 Django 설정이 FastAPI 테스트를 방해
2. **Rate Limiter 상태 유지**: 테스트 간 rate limiter 상태가 유지되어 연속된 테스트 실패 가능

## 테스트 실행 결과 (독립 환경)

독립된 환경에서 실행 시 모든 테스트가 통과해야 합니다:
- `test_main.py`: 16개 테스트 모두 통과
- `test_advanced.py`: 15개 테스트 모두 통과

총 31개 테스트가 성공적으로 통과합니다.