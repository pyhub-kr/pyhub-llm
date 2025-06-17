# 🔑 API 키 완벽 가이드

API 키는 AI 서비스를 사용하기 위한 "입장권"입니다. 자세히 알아봅시다!

## 🤔 API 키가 왜 필요한가요?

### 실생활 비유
- **헬스장 회원증**: 헬스장을 이용하려면 회원증이 필요하듯이
- **도서관 카드**: 도서관에서 책을 빌리려면 카드가 필요하듯이
- **API 키**: AI 서비스를 사용하려면 API 키가 필요합니다

### API 키의 역할
1. **신원 확인**: "당신이 누구인지" 확인
2. **사용량 추적**: 얼마나 사용했는지 기록
3. **요금 청구**: 사용한 만큼 비용 계산

## 🏪 API 키 받는 곳

### 1. OpenAI (ChatGPT 만든 회사)
- **사이트**: [platform.openai.com](https://platform.openai.com)
- **가격**: 사용한 만큼 (1,000토큰당 약 2원)
- **무료 크레딧**: 처음 가입 시 $5 제공

### 2. Anthropic (Claude 만든 회사)
- **사이트**: [console.anthropic.com](https://console.anthropic.com)
- **가격**: 사용한 만큼 (OpenAI와 비슷)
- **무료 크레딧**: 처음 가입 시 제공

### 3. Ollama (무료!)
- **사이트**: [ollama.ai](https://ollama.ai)
- **가격**: 완전 무료 (내 컴퓨터에서 실행)
- **단점**: 성능이 조금 떨어질 수 있음

## 📝 OpenAI API 키 받기 (단계별 가이드)

### 1단계: 회원가입
```
1. platform.openai.com 접속
2. "Sign up" 클릭
3. 이메일로 가입 또는 구글 계정으로 가입
```

### 2단계: API 키 생성
```
1. 로그인 후 우측 상단 프로필 클릭
2. "API keys" 선택
3. "Create new secret key" 클릭
4. 키 이름 입력 (예: "my-first-key")
5. "Create secret key" 클릭
```

### 3단계: API 키 저장
```
⚠️ 중요: API 키는 한 번만 보여집니다!
1. 생성된 키를 복사 (sk-로 시작하는 긴 문자열)
2. 안전한 곳에 저장
3. "Done" 클릭
```

## 🔒 API 키 안전하게 보관하기

### ❌ 하면 안 되는 것들
```python
# 절대 이렇게 하지 마세요!
api_key = "sk-abc123..."  # 코드에 직접 쓰기 ❌
```

### ✅ 올바른 방법들

#### 방법 1: 환경 변수 사용 (추천!)
```python
import os

# 환경 변수에서 API 키 가져오기
api_key = os.environ.get("OPENAI_API_KEY")

# 사용하기
from pyhub.llm import LLM
assistant = LLM.create("gpt-4o-mini", api_key=api_key)
```

#### 방법 2: .env 파일 사용
1. `.env` 파일 만들기:
```bash
# .env 파일 내용
OPENAI_API_KEY=sk-abc123...
```

2. Python에서 사용:
```python
from pyhub.llm import LLM

# pyhub-llm이 자동으로 .env 파일을 읽습니다
assistant = LLM.create("gpt-4o-mini")
```

3. `.gitignore`에 추가 (중요!):
```
.env
```

## 💰 비용 이해하기

### 토큰이란?
- **토큰**: AI가 이해하는 텍스트의 단위
- **한글**: 1글자 ≈ 2-3토큰
- **영어**: 1단어 ≈ 1-2토큰

### 예상 비용
```
"안녕하세요" = 약 5토큰
"Hello" = 약 1토큰

1,000토큰 = 약 2원
→ "안녕하세요" 200번 = 약 2원
```

### 비용 절약 팁
1. **짧고 명확한 질문** 사용
2. **캐싱** 활용 (같은 질문 반복 안 함)
3. **저렴한 모델** 사용 (gpt-4o-mini)

## 🚦 첫 API 키 설정하기

### Windows (명령 프롬프트)
```cmd
set OPENAI_API_KEY=sk-abc123...
```

### Mac/Linux (터미널)
```bash
export OPENAI_API_KEY=sk-abc123...
```

### 영구 설정 (Mac/Linux)
```bash
echo 'export OPENAI_API_KEY=sk-abc123...' >> ~/.bashrc
source ~/.bashrc
```

## ✅ API 키 설정 확인

```python
import os

# API 키가 설정되었는지 확인
api_key = os.environ.get("OPENAI_API_KEY")

if api_key:
    print("✅ API 키가 설정되었습니다!")
    print(f"키 시작 부분: {api_key[:7]}...")
else:
    print("❌ API 키가 설정되지 않았습니다.")
```

## 🆘 문제 해결

### "API 키가 유효하지 않습니다"
- API 키를 다시 확인하세요
- 복사할 때 공백이 포함되지 않았는지 확인

### "크레딧이 부족합니다"
- OpenAI 대시보드에서 잔액 확인
- 결제 수단 등록 필요

### "API 키를 찾을 수 없습니다"
- 환경 변수 이름 확인 (OPENAI_API_KEY)
- 터미널을 재시작해보세요

## 🎯 다음 단계

API 키 준비가 끝났다면, 드디어 [첫 AI 대화](../01-hello-llm/first-chat.md)를 시작해봅시다!