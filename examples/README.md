# pyhub-llm 예제

이 디렉토리는 pyhub-llm의 다양한 사용 예제를 포함합니다.

## 📁 디렉토리 구조

```text
examples/
├── basic/           # 초급 예제
├── intermediate/    # 중급 예제
└── advanced/        # 고급 예제
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# API 키 설정
export OPENAI_API_KEY="your-api-key"

# pyhub-llm 설치
pip install pyhub-llm
```

### 2. 예제 실행
```bash
# 특정 예제 실행
python examples/basic/01_hello_world.py

# 모든 예제 테스트
python examples/test_examples.py
```

## 📚 난이도별 가이드

### 🌱 [초급 (Basic)](./basic/README.md)
- Hello World
- 스트리밍 응답
- 대화 관리
- 파일 처리

### 🚀 [중급 (Intermediate)](./intermediate/README.md)
- 구조화된 출력
- 비동기 처리
- 캐싱
- 도구/함수 호출
- 템플릿 활용
- History Backup

### 🔥 [고급 (Advanced)](./advanced/README.md)
- MCP 통합
- 웹 프레임워크 통합
- 실용적인 예제

## 📦 필수 패키지

각 예제 파일 상단의 docstring에 필요한 패키지가 명시되어 있습니다.

기본 요구사항:
- `pyhub-llm`
- `python >= 3.8`

선택적 패키지:
- `sqlalchemy` - History Backup 예제
- `scikit-learn` - 임베딩 유사도 계산
- `numpy` - 벡터 연산

## 🔧 문제 해결

### API 키 오류
```bash
export OPENAI_API_KEY="sk-..."
```

### Import 오류
```bash
pip install -e ".[dev,all]"
```

### 타임아웃 오류
일부 예제는 여러 API 호출을 수행하므로 시간이 걸릴 수 있습니다.

## 📝 기여하기

새로운 예제를 추가하려면:
1. 적절한 난이도 디렉토리에 파일 생성
2. 파일 상단에 명확한 docstring 추가
3. 필요한 패키지 명시
4. 에러 처리 포함
5. 한글 주석으로 설명 추가