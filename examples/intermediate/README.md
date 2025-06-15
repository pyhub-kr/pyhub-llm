# 중급 예제

이 디렉토리는 pyhub-llm의 중급 기능을 보여주는 예제들을 포함합니다.

## 예제 목록

1. **01_structured_output.py** - 구조화된 출력
   - Pydantic 모델로 응답 형식 정의
   - 다양한 구조의 데이터 추출
   - 복합 구조 처리
   - 타입 안전성 보장

2. **02_async_processing.py** - 비동기 처리
   - 병렬 처리로 성능 향상
   - 비동기 스트리밍
   - 동시 대화 처리
   - 에러 처리 패턴

3. **03_caching.py** - 캐싱
   - 메모리/파일 캐시
   - 커스텀 캐시 구현
   - TTL 설정
   - 캐시 통계

4. **04_tools_functions.py** - 도구/함수 호출
   - 외부 함수 통합
   - 도구 스키마 정의
   - 멀티 도구 사용
   - 에러 처리

5. **05_templates.py** - 템플릿 활용
   - Jinja2 템플릿
   - 템플릿 상속
   - 커스텀 필터
   - 동적 프롬프트 생성

6. **06_history_backup.py** - History Backup
   - 대화 내역 저장/복원
   - SQLite 백업
   - 메타데이터 관리
   - 도구 상호작용 기록

## 필수 요구사항

### 환경 변수 설정
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 패키지 설치
```bash
# 기본 설치
pip install pyhub-llm

# 추가 의존성 (일부 예제용)
pip install sqlalchemy  # History Backup
pip install jinja2      # 템플릿 (pyhub-llm에 포함)
```

## 실행 방법

각 예제를 직접 실행할 수 있습니다:

```bash
# 구조화된 출력 예제
python 01_structured_output.py

# 비동기 처리 예제
python 02_async_processing.py

# 캐싱 예제
python 03_caching.py

# 도구/함수 호출 예제
python 04_tools_functions.py

# 템플릿 활용 예제
python 05_templates.py

# History Backup 예제
python 06_history_backup.py
```

## 주요 개념

### 구조화된 출력
- **Pydantic 모델**: 타입 안전한 응답 구조 정의
- **자동 검증**: 응답이 지정된 형식을 따르도록 보장
- **복잡한 구조**: 중첩된 객체, 리스트, 열거형 지원

### 비동기 처리
- **asyncio**: Python의 비동기 프로그래밍
- **병렬 실행**: 여러 LLM 호출을 동시에 처리
- **성능 향상**: 대기 시간 최소화

### 캐싱
- **비용 절감**: 동일한 요청 재사용
- **성능 향상**: 즉시 응답 제공
- **TTL 관리**: 캐시 유효 기간 설정

### 도구/함수 호출
- **Function Calling**: LLM이 외부 도구 사용
- **동적 실행**: 실시간 정보 접근
- **확장성**: 커스텀 도구 추가 가능

### 템플릿
- **프롬프트 관리**: 재사용 가능한 템플릿
- **동적 생성**: 변수 기반 프롬프트
- **유지보수성**: 프롬프트 중앙 관리

### History Backup
- **대화 보존**: 중요한 대화 저장
- **복원 기능**: 이전 대화 재개
- **분석 가능**: 대화 패턴 분석

## 학습 순서

1. **구조화된 출력**부터 시작하여 타입 안전한 응답 처리를 배우세요
2. **비동기 처리**로 성능을 향상시키는 방법을 익히세요
3. **캐싱**으로 비용을 절감하고 응답 속도를 개선하세요
4. **도구/함수 호출**로 LLM의 기능을 확장하세요
5. **템플릿**으로 프롬프트를 효율적으로 관리하세요
6. **History Backup**으로 대화를 영구 보존하세요

## 문제 해결

### 비동기 관련 오류
```python
# Windows에서 asyncio 오류 시
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

### 캐시 권한 오류
캐시 디렉토리 쓰기 권한을 확인하세요:
```bash
chmod 755 ./llm_cache
```

### SQLAlchemy 오류
History Backup에서 SQLAlchemy를 사용하려면:
```bash
pip install sqlalchemy
```

## 다음 단계

중급 예제를 마스터했다면 [고급 예제](../advanced/README.md)로 진행하세요!