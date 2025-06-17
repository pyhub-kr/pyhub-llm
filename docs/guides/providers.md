# 프로바이더 가이드

pyhub-llm이 지원하는 각 LLM 프로바이더의 특징과 사용법을 알아봅니다.

## 지원 프로바이더 개요

| 프로바이더 | 주요 모델 | 특징 | 가격대 |
|----------|---------|------|-------|
| OpenAI | GPT-4o, GPT-4o-mini | 가장 널리 사용, 안정적 | 중-고 |
| Anthropic | Claude 3.5 Sonnet/Haiku | 긴 컨텍스트, 안전성 | 중-고 |
| Google | Gemini 2.0 Flash | 멀티모달, 무료 티어 | 저-중 |
| Ollama | Llama, Mistral 등 | 로컬 실행, 무료 | 무료 |
| Upstage | Solar | 한국어 특화 | 저-중 |

## OpenAI

### 설정 및 초기화

```python
from pyhub.llm import OpenAILLM, LLM

# 자동 감지
llm = LLM.create("gpt-4o-mini")

# 명시적 생성
openai_llm = OpenAILLM(
    model="gpt-4o-mini",
    api_key="your-api-key",  # 또는 환경변수 OPENAI_API_KEY
    temperature=0.7,
    max_tokens=1000
)
```

### 지원 모델

```python
# GPT-4o 시리즈 (최신, 빠름)
llm_4o = LLM.create("gpt-4o")          # 고성능
llm_4o_mini = LLM.create("gpt-4o-mini") # 경제적

# GPT-4 Turbo
llm_4_turbo = LLM.create("gpt-4-turbo") # 128k 컨텍스트

# GPT-3.5 (레거시)
llm_35 = LLM.create("gpt-3.5-turbo")   # 저렴, 빠름
```

### OpenAI 특화 기능

```python
# 함수 호출 (Function Calling)
from pydantic import BaseModel

class WeatherParams(BaseModel):
    location: str
    unit: str = "celsius"

def get_weather(params: WeatherParams):
    # 실제 날씨 API 호출
    return f"{params.location}의 날씨는 맑음, 22{params.unit[0].upper()}"

# 도구 정의
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": WeatherParams.model_json_schema()
    }
}]

# 도구와 함께 질문
reply = openai_llm.ask_with_tools(
    "서울 날씨 어때?",
    tools=tools
)

# JSON 모드
reply = openai_llm.ask(
    "다음 정보를 JSON으로 정리하세요: 이름-김철수, 나이-30",
    response_format={"type": "json_object"}
)
```

### 이미지 생성 (DALL-E)

```python
# 이미지 생성 (별도 구현 필요)
from openai import OpenAI

client = OpenAI()
response = client.images.generate(
    model="dall-e-3",
    prompt="귀여운 고양이가 노트북으로 코딩하는 모습",
    size="1024x1024",
    quality="standard",
    n=1,
)
image_url = response.data[0].url
```

## Anthropic Claude

### 설정 및 초기화

```python
from pyhub.llm import AnthropicLLM

# 자동 감지
llm = LLM.create("claude-3-5-sonnet-latest")

# 명시적 생성
claude_llm = AnthropicLLM(
    model="claude-3-5-haiku-latest",
    api_key="your-api-key",  # 또는 환경변수 ANTHROPIC_API_KEY
    max_tokens=4096
)
```

### 지원 모델

```python
# Claude 3.5 시리즈 (최신)
sonnet_35 = LLM.create("claude-3-5-sonnet-latest")  # 최고 성능
haiku_35 = LLM.create("claude-3-5-haiku-latest")    # 빠르고 경제적

# Claude 3 시리즈
opus_3 = LLM.create("claude-3-opus-20240229")       # 최고급
sonnet_3 = LLM.create("claude-3-sonnet-20240229")   # 균형
haiku_3 = LLM.create("claude-3-haiku-20240307")     # 경제적
```

### Claude 특화 기능

```python
# 긴 컨텍스트 처리 (최대 200k 토큰)
with open("long_document.txt", "r") as f:
    long_text = f.read()

reply = claude_llm.ask(f"""
다음 문서를 분석해주세요:

{long_text}

주요 내용을 3가지로 요약해주세요.
""")

# 시스템 프롬프트 활용
ethical_claude = AnthropicLLM(
    model="claude-3-5-sonnet-latest",
    system_prompt="""
당신은 윤리적이고 도움이 되는 AI 어시스턴트입니다.
정확하고 균형잡힌 정보를 제공하며, 해로운 내용은 거부합니다.
"""
)

# Claude의 추론 능력 활용
reply = claude_llm.ask("""
다음 논리 퍼즐을 풀어주세요:

세 명의 친구 A, B, C가 있습니다.
- A는 항상 진실을 말합니다
- B는 항상 거짓을 말합니다  
- C는 진실과 거짓을 번갈아 말합니다

한 명에게 "당신은 A입니까?"라고 물었더니 "네"라고 답했습니다.
이 사람은 누구일까요? 이유를 설명해주세요.
""")
```

## Google Gemini

### 설정 및 초기화

```python
from pyhub.llm import GoogleLLM

# 자동 감지
llm = LLM.create("gemini-2.0-flash-exp")

# 명시적 생성  
gemini_llm = GoogleLLM(
    model="gemini-2.0-flash-exp",
    api_key="your-api-key",  # 또는 환경변수 GOOGLE_API_KEY
    temperature=0.9
)
```

### 지원 모델

```python
# Gemini 2.0 시리즈 (최신)
flash_2 = LLM.create("gemini-2.0-flash-exp")        # 실험적, 빠름

# Gemini 1.5 시리즈
pro_15 = LLM.create("gemini-1.5-pro")               # 고성능
flash_15 = LLM.create("gemini-1.5-flash")           # 빠르고 효율적

# Gemini 1.0 시리즈
pro_10 = LLM.create("gemini-1.0-pro")               # 안정적
```

### Gemini 특화 기능

```python
# 멀티모달 능력 (이미지, 비디오, 오디오)
reply = gemini_llm.ask(
    "이 이미지들을 비교 분석해주세요",
    files=["image1.jpg", "image2.jpg", "image3.jpg"]
)

# 안전 설정
from google.generativeai.types import HarmCategory, HarmBlockThreshold

gemini_safe = GoogleLLM(
    model="gemini-1.5-pro",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }
)

# 무료 티어 활용
# Google AI Studio에서 무료 API 키 발급 가능
free_gemini = GoogleLLM(
    model="gemini-1.5-flash",
    api_key="free-tier-api-key"
)
```

## Ollama (로컬 모델)

### 설정 및 초기화

```python
from pyhub.llm import OllamaLLM

# Ollama 서버가 로컬에서 실행 중이어야 함
# ollama serve

# 자동 감지
llm = LLM.create("llama3.2:latest")

# 명시적 생성
ollama_llm = OllamaLLM(
    model="llama3.2:latest",
    base_url="http://localhost:11434",  # 기본값
    temperature=0.8
)
```

### 지원 모델

```python
# 인기 모델들
llama32 = LLM.create("llama3.2:latest")         # Meta의 최신 모델
mistral = LLM.create("mistral:latest")          # 경량 고성능
codellama = LLM.create("codellama:latest")      # 코드 특화
phi3 = LLM.create("phi3:latest")                # Microsoft 소형 모델

# 한국어 모델
solar = LLM.create("solar:latest")              # 한국어 지원
```

### Ollama 특화 기능

```python
# 모델 관리
import subprocess

# 모델 다운로드
subprocess.run(["ollama", "pull", "llama3.2:latest"])

# 사용 가능한 모델 확인
result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
print(result.stdout)

# 커스텀 모델 생성
modelfile = """
FROM llama3.2:latest
PARAMETER temperature 0.5
SYSTEM 당신은 한국어를 유창하게 구사하는 AI 어시스턴트입니다.
"""

with open("Modelfile", "w") as f:
    f.write(modelfile)

subprocess.run(["ollama", "create", "korean-llama", "-f", "Modelfile"])

# 커스텀 모델 사용
korean_llm = OllamaLLM(model="korean-llama")
```

## Upstage Solar

### 설정 및 초기화

```python
from pyhub.llm import UpstageLLM

# 자동 감지
llm = LLM.create("solar-pro")

# 명시적 생성
solar_llm = UpstageLLM(
    model="solar-pro",
    api_key="your-api-key",  # 또는 환경변수 UPSTAGE_API_KEY
)
```

### 지원 모델

```python
# Solar 시리즈
solar_pro = LLM.create("solar-pro")      # 고성능
solar_mini = LLM.create("solar-mini")    # 경량화

# 특화 모델
solar_doc = LLM.create("solar-docvqa")   # 문서 분석
```

### Solar 특화 기능

```python
# 한국어 최적화
reply = solar_llm.ask("""
다음 한국어 문장을 분석해주세요:
"아버지가 방에 들어가신다"

1. 문법 구조
2. 존댓말 사용
3. 의미 해석
""")

# 문서 분석 (DocVQA)
doc_solar = UpstageLLM(model="solar-docvqa")
reply = doc_solar.ask(
    "이 문서에서 계약 금액을 찾아주세요",
    files=["contract.pdf"]
)
```

## 프로바이더 선택 가이드

### 사용 사례별 추천

```python
# 1. 일반 대화 및 질의응답
general_llm = LLM.create("gpt-4o-mini")  # 비용 효율적
# 또는
general_llm = LLM.create("claude-3-5-haiku-latest")  # 빠른 응답

# 2. 복잡한 추론 및 분석
complex_llm = LLM.create("gpt-4o")  # OpenAI 최고 성능
# 또는
complex_llm = LLM.create("claude-3-5-sonnet-latest")  # 뛰어난 추론

# 3. 코드 생성 및 기술 문서
code_llm = LLM.create("gpt-4-turbo")  # 코드 이해도 높음
# 또는 
code_llm = LLM.create("codellama:latest")  # 로컬, 무료

# 4. 한국어 작업
korean_llm = LLM.create("solar-pro")  # 한국어 특화
# 또는
korean_llm = LLM.create("claude-3-5-sonnet-latest")  # 한국어도 우수

# 5. 멀티모달 (이미지/비디오)
multimodal_llm = LLM.create("gemini-2.0-flash-exp")  # 최신 멀티모달
# 또는
multimodal_llm = LLM.create("gpt-4o")  # 이미지 이해

# 6. 프라이버시 중시 (로컬)
private_llm = LLM.create("llama3.2:latest")  # 완전 로컬
# 또는
private_llm = LLM.create("mistral:latest")  # 경량 로컬
```

### 비용 최적화 전략

```python
class CostOptimizedLLM:
    def __init__(self):
        # 작업별 모델 구분
        self.simple_llm = LLM.create("gpt-4o-mini", stateless=True)
        self.complex_llm = LLM.create("gpt-4o")
        self.local_llm = LLM.create("llama3.2:latest")
    
    def ask(self, prompt, complexity="auto"):
        if complexity == "auto":
            # 프롬프트 길이로 복잡도 추정
            complexity = "complex" if len(prompt) > 500 else "simple"
        
        if complexity == "simple":
            return self.simple_llm.ask(prompt)
        elif complexity == "complex":
            return self.complex_llm.ask(prompt)
        elif complexity == "local":
            return self.local_llm.ask(prompt)

# 사용
optimizer = CostOptimizedLLM()

# 간단한 질문 - 저렴한 모델 사용
reply1 = optimizer.ask("오늘 날짜는?", complexity="simple")

# 복잡한 분석 - 고급 모델 사용  
reply2 = optimizer.ask("이 코드의 시간복잡도를 분석하고...", complexity="complex")

# 민감한 데이터 - 로컬 모델 사용
reply3 = optimizer.ask("이 개인정보를 분석해서...", complexity="local")
```

## 프로바이더 마이그레이션

### 프로바이더 간 전환

```python
# 기존 OpenAI 코드
old_llm = LLM.create("gpt-4o-mini")
reply = old_llm.ask("질문")

# Claude로 전환 - 코드 변경 최소화
new_llm = LLM.create("claude-3-5-haiku-latest")  # 이 줄만 변경
reply = new_llm.ask("질문")  # 동일한 인터페이스

# 동적 프로바이더 전환
def create_llm_with_fallback(primary_model, fallback_model):
    try:
        return LLM.create(primary_model)
    except Exception as e:
        print(f"Primary model failed: {e}")
        return LLM.create(fallback_model)

# 사용
llm = create_llm_with_fallback(
    primary_model="gpt-4o",
    fallback_model="llama3.2:latest"  # 로컬 폴백
)
```

### A/B 테스트

```python
import random
from collections import defaultdict

class ABTestLLM:
    def __init__(self, models, weights=None):
        self.models = {name: LLM.create(name) for name in models}
        self.weights = weights or [1] * len(models)
        self.results = defaultdict(list)
    
    def ask(self, prompt, **kwargs):
        # 가중치 기반 모델 선택
        model_name = random.choices(
            list(self.models.keys()),
            weights=self.weights
        )[0]
        
        llm = self.models[model_name]
        reply = llm.ask(prompt, **kwargs)
        
        # 결과 기록
        self.results[model_name].append({
            'prompt': prompt,
            'response': reply.text,
            'tokens': reply.usage.total_tokens if reply.usage else 0,
            'time': reply.elapsed_time
        })
        
        return reply
    
    def get_statistics(self):
        stats = {}
        for model, results in self.results.items():
            if results:
                avg_tokens = sum(r['tokens'] for r in results) / len(results)
                avg_time = sum(r['time'] for r in results) / len(results)
                stats[model] = {
                    'count': len(results),
                    'avg_tokens': avg_tokens,
                    'avg_time': avg_time
                }
        return stats

# 사용
ab_test = ABTestLLM(
    models=["gpt-4o-mini", "claude-3-5-haiku-latest"],
    weights=[0.5, 0.5]  # 50:50 분할
)

# 테스트 실행
for _ in range(100):
    ab_test.ask("파이썬 팁을 하나 알려주세요")

# 결과 분석
print(ab_test.get_statistics())
```

## 다음 단계

- [구조화된 출력](structured-output.md) - JSON 스키마와 Pydantic 활용
- [고급 기능](advanced.md) - 스트리밍, 비동기, 임베딩 등
- [API 레퍼런스](../api-reference/index.md) - 전체 API 문서