# Code Interpreter Tool Examples

이 디렉토리는 pyhub-llm의 Code Interpreter 도구 사용 예시를 포함합니다.

## Code Interpreter란?

Code Interpreter는 LLM이 Python 코드를 안전하게 실행할 수 있게 해주는 도구입니다. 데이터 분석, 시각화, 계산 등 다양한 작업을 자연어로 요청하면 LLM이 자동으로 코드를 생성하고 실행합니다.

### 주요 기능

- 🔒 **안전한 실행 환경**: 위험한 코드는 자동으로 차단
- 📊 **데이터 분석**: pandas, numpy 등 주요 라이브러리 지원
- 📈 **시각화**: matplotlib, seaborn으로 차트 생성
- 💾 **세션 관리**: 변수와 상태를 유지하며 연속 작업 가능
- 🚀 **다중 백엔드**: Local, Docker, Remote 실행 지원 (Docker는 개발 중)

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

response = llm.ask_with_tools(
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

## 사용법

### 기본 사용

```python
from pyhub.llm import OpenAILLM
from pyhub.llm.agents.tools import CodeInterpreter

# LLM과 도구 초기화
llm = OpenAILLM(model="gpt-4")
code_tool = CodeInterpreter(backend="local")

# 코드 실행 요청
response = llm.ask_with_tools(
    "Analyze this data and create a visualization",
    tools=[code_tool]
)
```

### 세션 사용

```python
# 세션 ID로 상태 유지
session_id = "my_analysis"

# 첫 번째 실행
response1 = llm.ask_with_tools(
    "Load data and create df",
    tools=[code_tool],
    tool_kwargs={"session_id": session_id}
)

# 같은 세션에서 계속 작업
response2 = llm.ask_with_tools(
    "Using df from before, create visualizations",
    tools=[code_tool],
    tool_kwargs={"session_id": session_id}
)
```

### 파일 작업

```python
# 파일 업로드하여 분석
response = llm.ask_with_tools(
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

현재는 Local 백엔드만 지원하며, Docker 백엔드는 개발 중입니다:

- **Local**: 제한된 로컬 Python 환경에서 실행
- **Docker** (개발 중): 완전히 격리된 컨테이너에서 실행
- **Remote** (계획 중): 원격 서버에서 실행

## 문제 해결

### "Module not found" 오류
필요한 라이브러리가 설치되어 있지 않습니다:
```bash
pip install pandas matplotlib seaborn numpy scipy scikit-learn
```

### 보안 검증 실패
코드에 허용되지 않은 작업이 포함되어 있습니다. 에러 메시지를 확인하고 안전한 대안을 사용하세요.

### 세션 타임아웃
세션은 기본 60분 후 만료됩니다. 필요시 새 세션을 생성하세요.

## 더 많은 예시

더 많은 Code Interpreter 활용 예시는 [pyhub-llm 문서](https://github.com/pyhub-kr/pyhub-llm)를 참고하세요.