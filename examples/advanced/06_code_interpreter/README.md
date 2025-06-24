# Code Interpreter Tool Examples

이 디렉토리는 pyhub-llm의 Code Interpreter 도구 사용 예시를 포함합니다.

## 설치

Code Interpreter를 사용하려면 추가 의존성이 필요합니다:

```bash
# Code Interpreter와 필요한 데이터 분석 라이브러리 설치
pip install pyhub-llm[code-interpreter]

# Docker 백엔드도 사용하려면
pip install pyhub-llm[code-interpreter,docker]

# 또는 모든 기능 설치
pip install pyhub-llm[all]
```

## Code Interpreter란?

Code Interpreter는 LLM이 Python 코드를 안전하게 실행할 수 있게 해주는 도구입니다. 데이터 분석, 시각화, 계산 등 다양한 작업을 자연어로 요청하면 LLM이 자동으로 코드를 생성하고 실행합니다.

### 주요 기능

- 🔒 **안전한 실행 환경**: 위험한 코드는 자동으로 차단
- 📊 **데이터 분석**: pandas, numpy 등 주요 라이브러리 지원
- 📈 **시각화**: matplotlib, seaborn으로 차트 생성
- 💾 **세션 관리**: 변수와 상태를 유지하며 연속 작업 가능
- 🚀 **다중 백엔드**: Local, Docker, Remote Docker 실행 지원

## 예시 파일

### 1. `01_basic_analysis.py` - 기본 데이터 분석
- 기본 통계 계산
- 데이터셋 생성 및 분석
- 간단한 시각화

```python
from pyhub.llm import OpenAILLM
from pyhub.llm.agents.tools import CodeInterpreter

llm = OpenAILLM(model="gpt-4")
code_tool = CodeInterpreter(backend="local")

response = llm.ask(
    "Create a dataset and calculate basic statistics",
    tools=[code_tool]
)
```

### 2. `02_visualization.py` - 고급 시각화
- 다중 서브플롯 대시보드
- 시계열 데이터 시각화
- 통계적 시각화
- 인터랙티브 차트 (plotly)

### 3. `03_session_based.py` - 세션 기반 분석
- 연속적인 데이터 분석 워크플로우
- 고객 세그먼테이션 예시
- 머신러닝 파이프라인
- 세션 상태 관리

### 4. `04_multi_tool.py` - 다중 도구 활용
- Code Interpreter + Calculator 조합
- 컨텍스트 기반 분석
- 실무 시나리오 (CSV 분석)

### 5. `05_docker_backend.py` - Docker 백엔드 사용
- 완전히 격리된 Docker 컨테이너에서 코드 실행
- 보안 강화 및 리소스 제한
- 로컬/원격 Docker 지원
- Local vs Docker 백엔드 비교

## 사용법

### 기본 사용

```python
from pyhub.llm import OpenAILLM
from pyhub.llm.agents.tools import CodeInterpreter

# LLM과 도구 초기화
llm = OpenAILLM(model="gpt-4")
code_tool = CodeInterpreter(backend="local")  # Usage notes included by default

# 코드 실행 요청
response = llm.ask(
    "Analyze this data and create a visualization",
    tools=[code_tool]
)
```

### 주의사항 및 사용 가이드

Code Interpreter는 기본적으로 다음과 같은 사용 가이드를 LLM에게 제공합니다:

- **Matplotlib 플롯**: `plt.show()` 대신 `plt.savefig('filename.png')` 사용
- **파일 저장 확인**: 파일 저장 후 확인 메시지 출력
- **파일명**: 설명적인 파일명 사용

```python
# Usage notes를 비활성화하려면
code_tool = CodeInterpreter(backend="local", include_usage_notes=False)

# 사용자 정의 설명과 함께
code_tool = CodeInterpreter(
    backend="local",
    description="Custom Python executor for data science",
    include_usage_notes=True
)
```

### 세션 사용

```python
# 세션 ID로 상태 유지
session_id = "my_analysis"

# 첫 번째 실행
response1 = llm.ask(
    "Load data and create df",
    tools=[code_tool],
    tool_kwargs={"session_id": session_id}
)

# 같은 세션에서 계속 작업
response2 = llm.ask(
    "Using df from before, create visualizations",
    tools=[code_tool],
    tool_kwargs={"session_id": session_id}
)
```

### 파일 작업

```python
# 파일 업로드하여 분석
response = llm.ask(
    "Analyze the uploaded CSV file",
    tools=[code_tool],
    tool_kwargs={"files": ["data.csv"]}
)

# 생성된 파일 다운로드
output_file = code_tool.download_file(session_id, "results.png")
```

## 보안 고려사항

Code Interpreter는 다음과 같은 보안 제한이 있습니다:

- ❌ 시스템 명령 실행 불가 (`os.system`, `subprocess`)
- ❌ 네트워크 접근 차단 (`requests`, `urllib`)
- ❌ 파일 시스템 직접 접근 제한
- ✅ 안전한 데이터 분석 라이브러리만 허용

## 지원 라이브러리

- **데이터 분석**: pandas, numpy, scipy
- **시각화**: matplotlib, seaborn, plotly
- **머신러닝**: scikit-learn
- **기타**: datetime, json, math, collections

## 실행 환경

### Local 백엔드
제한된 로컬 Python 환경에서 실행합니다. 기본값이며 추가 설정이 필요 없습니다.

```python
code_tool = CodeInterpreter(backend="local")
```

### Docker 백엔드
완전히 격리된 Docker 컨테이너에서 실행합니다. Docker가 설치되어 있어야 합니다.

```bash
# Docker 패키지 설치
pip install pyhub-llm[docker]

# Docker Desktop이 실행 중이어야 함
```

```python
code_tool = CodeInterpreter(
    backend="docker",
    backend_config={
        "image_name": "python:3.9-slim",
        "memory_limit": "512m",
        "cpu_quota": 50000,  # 50% CPU
        "network_mode": "none"  # 네트워크 차단
    }
)
```

### Remote Docker 백엔드
원격 Docker 데몬에 연결하여 실행합니다.

```python
code_tool = CodeInterpreter(
    backend="docker",
    backend_config={
        "remote_docker_url": "tcp://remote-host:2376"
    }
)
```

## 문제 해결

### "Module not found" 오류
필요한 라이브러리가 설치되어 있지 않습니다:
```bash
# Code Interpreter 의존성 설치
pip install pyhub-llm[code-interpreter]

# 또는 개별 설치
pip install pandas matplotlib seaborn numpy scipy scikit-learn
```

### 보안 검증 실패
코드에 허용되지 않은 작업이 포함되어 있습니다. 에러 메시지를 확인하고 안전한 대안을 사용하세요.

### 세션 타임아웃
세션은 기본 60분 후 만료됩니다. 필요시 새 세션을 생성하세요.

## 더 많은 예시

더 많은 Code Interpreter 활용 예시는 [pyhub-llm 문서](https://github.com/pyhub-kr/pyhub-llm)를 참고하세요.