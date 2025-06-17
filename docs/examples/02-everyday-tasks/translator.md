# 🌍 다국어 번역기 만들기

AI를 활용해 전문 번역가 수준의 번역 도구를 만들어봅시다!

## 🎯 만들 것

```
한국어: 안녕하세요, 만나서 반갑습니다.
영어: Hello, nice to meet you.
일본어: こんにちは、お会いできて嬉しいです。
중국어: 你好，很高兴见到你。
```

## 📝 기본 번역기

### Step 1: 간단한 번역기

```python
# translator.py
from pyhub.llm import LLM

class Translator:
    """AI 번역기"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.ai = LLM.create(model)
        
    def translate(self, text, target_language="English", source_language="auto"):
        """텍스트를 번역합니다"""
        if source_language == "auto":
            # 언어 자동 감지
            source_prompt = "원문의 언어를 감지해서"
        else:
            source_prompt = f"{source_language}를"
            
        prompt = f"""
        다음 텍스트를 {source_prompt} {target_language}로 번역해주세요.
        자연스럽고 정확하게 번역해주세요.
        
        원문:
        {text}
        
        번역문만 출력해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def detect_language(self, text):
        """언어를 감지합니다"""
        prompt = f"""
        다음 텍스트가 어떤 언어인지 알려주세요:
        "{text}"
        
        언어 이름만 답해주세요. (예: Korean, English, Japanese)
        """
        
        response = self.ai.ask(prompt)
        return response.text.strip()

# 사용 예시
translator = Translator()

# 기본 번역
text = "인공지능은 우리의 미래를 바꿀 것입니다."
print("🇰🇷 원문:", text)
print("🇺🇸 영어:", translator.translate(text, "English"))
print("🇯🇵 일본어:", translator.translate(text, "Japanese"))
print("🇨🇳 중국어:", translator.translate(text, "Chinese"))
print("🇫🇷 프랑스어:", translator.translate(text, "French"))

# 언어 감지
print("\n🔍 언어 감지:")
texts = [
    "Hello, world!",
    "안녕하세요",
    "こんにちは",
    "Bonjour"
]

for t in texts:
    lang = translator.detect_language(t)
    print(f"'{t}' → {lang}")
```

### Step 2: 고급 번역기

```python
class AdvancedTranslator(Translator):
    """고급 기능을 갖춘 번역기"""
    
    def translate_with_context(self, text, target_language, context="general"):
        """문맥을 고려한 번역"""
        context_guides = {
            "formal": "격식있는 공식적인 상황",
            "casual": "친근한 일상 대화",
            "business": "비즈니스 상황",
            "technical": "기술/전문 용어",
            "academic": "학술적인 내용"
        }
        
        context_desc = context_guides.get(context, context)
        
        prompt = f"""
        다음 텍스트를 {target_language}로 번역해주세요.
        상황: {context_desc}
        
        원문:
        {text}
        
        해당 상황에 맞는 적절한 표현으로 번역해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def translate_with_alternatives(self, text, target_language):
        """여러 번역 옵션 제공"""
        prompt = f"""
        다음 텍스트를 {target_language}로 번역해주세요.
        3가지 다른 버전으로 번역해주세요:
        
        원문: {text}
        
        형식:
        1. [정확한 직역]
        2. [자연스러운 의역]
        3. [상황에 맞는 관용구 사용]
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def translate_with_explanation(self, text, target_language):
        """번역과 설명을 함께 제공"""
        prompt = f"""
        다음 텍스트를 {target_language}로 번역하고 설명해주세요:
        
        원문: {text}
        
        다음 형식으로 답해주세요:
        번역: [번역문]
        
        설명:
        - 주요 단어/표현 설명
        - 문화적 차이나 뉘앙스
        - 주의할 점
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
adv_translator = AdvancedTranslator()

# 문맥별 번역
korean_text = "잘 부탁드립니다"
print("🎯 문맥별 번역 - '잘 부탁드립니다':\n")

contexts = ["formal", "casual", "business"]
for ctx in contexts:
    result = adv_translator.translate_with_context(
        korean_text, 
        "English", 
        context=ctx
    )
    print(f"{ctx.upper()}: {result}")

# 다양한 번역 옵션
print("\n🔄 번역 옵션들:")
options = adv_translator.translate_with_alternatives(
    "시간이 약이다",
    "English"
)
print(options)

# 번역 설명
print("\n📚 번역 설명:")
explanation = adv_translator.translate_with_explanation(
    "정이 많다",
    "English"
)
print(explanation)
```

## 🎨 실용적인 번역 도구

### 1. 문서 번역기

```python
class DocumentTranslator(AdvancedTranslator):
    """문서 번역 전문 도구"""
    
    def translate_document(self, document, target_language, preserve_format=True):
        """문서 전체를 번역합니다"""
        # 문단별로 분리
        paragraphs = document.strip().split('\n\n')
        translated_paragraphs = []
        
        for i, para in enumerate(paragraphs, 1):
            print(f"번역 중... ({i}/{len(paragraphs)})")
            
            if preserve_format:
                # 특수 형식 보존
                if para.startswith('#'):  # 제목
                    title = para.lstrip('#').strip()
                    trans_title = self.translate(title, target_language)
                    translated_paragraphs.append(f"{'#' * para.count('#')} {trans_title}")
                elif para.startswith('-') or para.startswith('*'):  # 목록
                    lines = para.split('\n')
                    trans_lines = []
                    for line in lines:
                        text = line.lstrip('-*').strip()
                        trans_text = self.translate(text, target_language)
                        trans_lines.append(f"{line[:line.find(text)]}{trans_text}")
                    translated_paragraphs.append('\n'.join(trans_lines))
                else:
                    # 일반 문단
                    translated_paragraphs.append(
                        self.translate(para, target_language)
                    )
            else:
                translated_paragraphs.append(
                    self.translate(para, target_language)
                )
        
        return '\n\n'.join(translated_paragraphs)
    
    def create_bilingual_document(self, document, target_language):
        """원문과 번역문을 나란히 표시"""
        paragraphs = document.strip().split('\n\n')
        bilingual_doc = []
        
        for para in paragraphs:
            # 원문
            bilingual_doc.append(f"[원문]\n{para}")
            
            # 번역
            translation = self.translate(para, target_language)
            bilingual_doc.append(f"[{target_language}]\n{translation}")
            
            bilingual_doc.append("-" * 50)
        
        return '\n\n'.join(bilingual_doc)

# 사용 예시
doc_translator = DocumentTranslator()

# 샘플 문서
document = """
# AI와 미래

인공지능은 21세기의 가장 중요한 기술 중 하나입니다.

## 주요 응용 분야

- 의료: 질병 진단과 신약 개발
- 교육: 맞춤형 학습 시스템
- 교통: 자율주행 자동차

AI는 우리의 일상을 변화시키고 있습니다.
"""

# 문서 번역
print("📄 문서 번역:")
translated = doc_translator.translate_document(document, "English")
print(translated)

# 대역 문서
print("\n📑 대역 문서:")
bilingual = doc_translator.create_bilingual_document(
    "AI는 미래다.\n\n우리는 준비해야 한다.",
    "English"
)
print(bilingual)
```

### 2. 대화 번역기

```python
class ConversationTranslator(AdvancedTranslator):
    """대화 번역 전문 도구"""
    
    def translate_conversation(self, messages, target_language):
        """대화를 번역합니다"""
        translated_messages = []
        
        for msg in messages:
            speaker = msg.get("speaker", "Unknown")
            text = msg.get("text", "")
            
            # 대화체 번역
            prompt = f"""
            대화에서 나온 다음 말을 {target_language}로 번역해주세요.
            구어체로 자연스럽게 번역해주세요.
            
            화자: {speaker}
            원문: {text}
            """
            
            response = self.ai.ask(prompt)
            
            translated_messages.append({
                "speaker": speaker,
                "original": text,
                "translated": response.text
            })
        
        return translated_messages
    
    def real_time_translate(self, text, target_language, previous_context=""):
        """실시간 대화 번역 (문맥 고려)"""
        prompt = f"""
        실시간 대화 번역입니다.
        
        이전 대화:
        {previous_context}
        
        현재 발언: {text}
        
        {target_language}로 자연스럽게 번역해주세요.
        짧고 구어체로 번역하세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
conv_translator = ConversationTranslator()

# 대화 번역
conversation = [
    {"speaker": "A", "text": "안녕! 오랜만이야"},
    {"speaker": "B", "text": "정말 오랜만이네. 잘 지냈어?"},
    {"speaker": "A", "text": "응, 요즘 새 프로젝트 때문에 바빴어"},
    {"speaker": "B", "text": "어떤 프로젝트야?"}
]

print("💬 대화 번역:")
translated_conv = conv_translator.translate_conversation(conversation, "English")

for msg in translated_conv:
    print(f"\n{msg['speaker']}: {msg['original']}")
    print(f"→ {msg['translated']}")

# 실시간 번역
print("\n⚡ 실시간 번역:")
context = "A: 안녕하세요 / B: Hello"
new_message = "오늘 날씨 좋네요"
real_time = conv_translator.real_time_translate(
    new_message, 
    "English", 
    context
)
print(f"원문: {new_message}")
print(f"번역: {real_time}")
```

### 3. 전문 분야 번역기

```python
class SpecializedTranslator(AdvancedTranslator):
    """전문 분야 번역기"""
    
    def __init__(self):
        super().__init__()
        self.domains = {
            "medical": "의학/의료",
            "legal": "법률",
            "technical": "기술/IT",
            "financial": "금융/경제",
            "scientific": "과학/연구"
        }
    
    def translate_specialized(self, text, target_language, domain):
        """전문 용어를 정확히 번역합니다"""
        domain_desc = self.domains.get(domain, domain)
        
        prompt = f"""
        다음은 {domain_desc} 분야의 전문 텍스트입니다.
        {target_language}로 정확하게 번역해주세요.
        
        주의사항:
        - 전문 용어는 해당 분야의 표준 번역 사용
        - 필요시 원어를 괄호 안에 병기
        - 정확성이 최우선
        
        원문:
        {text}
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def create_glossary(self, terms, source_lang, target_lang, domain):
        """용어집을 생성합니다"""
        domain_desc = self.domains.get(domain, domain)
        
        prompt = f"""
        {domain_desc} 분야의 용어를 {source_lang}에서 {target_lang}로 번역해주세요.
        
        용어 목록:
        {', '.join(terms)}
        
        다음 형식으로 작성해주세요:
        1. [원어] → [번역] (설명)
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def translate_with_terminology(self, text, target_language, glossary):
        """용어집을 활용한 번역"""
        prompt = f"""
        다음 텍스트를 {target_language}로 번역해주세요.
        
        제공된 용어집을 반드시 따라주세요:
        {glossary}
        
        원문:
        {text}
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
spec_translator = SpecializedTranslator()

# 전문 분야 번역
medical_text = "The patient presented with acute myocardial infarction and was immediately administered thrombolytic therapy."

print("🏥 의학 번역:")
medical_trans = spec_translator.translate_specialized(
    medical_text,
    "Korean",
    "medical"
)
print(f"원문: {medical_text}")
print(f"번역: {medical_trans}")

# 용어집 생성
print("\n📖 IT 용어집:")
it_terms = ["machine learning", "neural network", "deep learning", "algorithm"]
glossary = spec_translator.create_glossary(
    it_terms,
    "English",
    "Korean",
    "technical"
)
print(glossary)

# 용어집 활용 번역
print("\n🔧 용어집 활용 번역:")
tech_glossary = """
- API → 응용 프로그램 인터페이스
- framework → 프레임워크
- deployment → 배포
"""

tech_text = "We need to update the API framework before deployment."
term_trans = spec_translator.translate_with_terminology(
    tech_text,
    "Korean",
    tech_glossary
)
print(f"원문: {tech_text}")
print(f"번역: {term_trans}")
```

## 🌐 다국어 동시 번역기

```python
class MultiLanguageTranslator(AdvancedTranslator):
    """여러 언어로 동시에 번역하는 도구"""
    
    def __init__(self):
        super().__init__()
        self.common_languages = {
            "en": "English",
            "ja": "Japanese", 
            "zh": "Chinese",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
            "ar": "Arabic"
        }
    
    def translate_to_multiple(self, text, target_languages):
        """여러 언어로 동시에 번역합니다"""
        translations = {}
        
        prompt = f"""
        다음 텍스트를 {', '.join(target_languages)}로 번역해주세요:
        
        원문: {text}
        
        각 언어별로 번역을 제공해주세요:
        """
        
        for lang in target_languages:
            prompt += f"\n{lang}:"
        
        response = self.ai.ask(prompt)
        
        # 결과 파싱
        lines = response.text.strip().split('\n')
        for line in lines:
            for lang in target_languages:
                if line.startswith(f"{lang}:"):
                    translations[lang] = line.replace(f"{lang}:", "").strip()
        
        return translations
    
    def create_multilingual_card(self, phrase):
        """다국어 카드를 만듭니다"""
        languages = ["English", "Japanese", "Chinese", "Spanish", "French"]
        
        prompt = f"""
        '{phrase}'를 다음 언어로 번역하고 발음도 함께 제공해주세요:
        {', '.join(languages)}
        
        형식:
        언어: 번역 (발음)
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def compare_translations(self, text, languages):
        """번역을 비교 분석합니다"""
        translations = self.translate_to_multiple(text, languages)
        
        prompt = f"""
        다음 번역들을 비교 분석해주세요:
        
        원문: {text}
        
        번역들:
        """
        
        for lang, trans in translations.items():
            prompt += f"\n{lang}: {trans}"
        
        prompt += """
        
        각 번역의 특징과 뉘앙스 차이를 설명해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
multi_translator = MultiLanguageTranslator()

# 다국어 동시 번역
text = "평화는 모든 인류의 꿈입니다"
print("🌍 다국어 번역:")

languages = ["English", "Japanese", "Chinese", "Spanish", "French"]
translations = multi_translator.translate_to_multiple(text, languages)

for lang, trans in translations.items():
    print(f"{lang}: {trans}")

# 다국어 카드
print("\n🎴 다국어 카드 - '감사합니다':")
card = multi_translator.create_multilingual_card("감사합니다")
print(card)

# 번역 비교
print("\n🔍 번역 비교 분석:")
comparison = multi_translator.compare_translations(
    "시간은 금이다",
    ["English", "Japanese", "Chinese"]
)
print(comparison)
```

## 📱 번역 도우미 앱

```python
class TranslationApp:
    """통합 번역 애플리케이션"""
    
    def __init__(self):
        self.translator = AdvancedTranslator()
        self.history = []
    
    def interactive_mode(self):
        """대화형 번역 모드"""
        print("🌐 AI 번역기에 오신 것을 환영합니다!")
        print("사용법: [원문] → [목표 언어]")
        print("예시: 안녕하세요 → English")
        print("종료: quit\n")
        
        while True:
            user_input = input("번역할 텍스트: ").strip()
            
            if user_input.lower() in ['quit', '종료']:
                self.show_summary()
                break
            
            if '→' in user_input:
                text, lang = user_input.split('→')
                text = text.strip()
                lang = lang.strip()
            else:
                text = user_input
                lang = "English"  # 기본값
            
            # 번역 수행
            translation = self.translator.translate(text, lang)
            
            # 결과 표시
            print(f"\n📝 원문: {text}")
            print(f"🌏 {lang}: {translation}\n")
            
            # 기록 저장
            self.history.append({
                "original": text,
                "target_lang": lang,
                "translation": translation,
                "timestamp": datetime.now()
            })
    
    def show_summary(self):
        """사용 요약을 표시합니다"""
        if not self.history:
            return
        
        print("\n📊 번역 요약:")
        print(f"총 번역 수: {len(self.history)}")
        
        # 언어별 통계
        lang_stats = {}
        for item in self.history:
            lang = item["target_lang"]
            lang_stats[lang] = lang_stats.get(lang, 0) + 1
        
        print("\n언어별 번역:")
        for lang, count in lang_stats.items():
            print(f"- {lang}: {count}회")
        
        print("\n최근 번역 3개:")
        for item in self.history[-3:]:
            print(f"- {item['original'][:20]}... → {item['target_lang']}")

# 앱 실행
if __name__ == "__main__":
    app = TranslationApp()
    app.interactive_mode()
```

## ✅ 핵심 정리

1. **기본 번역**부터 **전문 번역**까지 단계별 구현
2. **문맥과 도메인**을 고려한 정확한 번역
3. **다국어 동시 번역**으로 효율성 향상
4. **대화형 인터페이스**로 사용성 개선

## 🚀 다음 단계

번역기를 완성했으니, 이제 [똑똑한 요약기](summarizer.md)를 만들어 긴 텍스트를 효율적으로 처리해봅시다!