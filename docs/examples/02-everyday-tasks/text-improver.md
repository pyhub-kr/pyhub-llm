# ✏️ 텍스트 개선 도구 만들기

맞춤법 검사, 문법 교정, 문체 개선까지! AI를 활용한 글쓰기 도우미를 만들어봅시다.

## 🎯 만들 것

```
원문: 오늘 회의에서 논의한 내용을 정리하면 다음과 같습니다 매출이 전년대비 20% 증가했구요...

개선: 오늘 회의에서 논의한 내용을 정리하면 다음과 같습니다. 매출이 전년 대비 20% 증가했고...
```

## 📝 기본 텍스트 개선기

### Step 1: 맞춤법과 문법 검사기

```python
# text_improver.py
from pyhub.llm import LLM

class TextImprover:
    """텍스트를 개선하는 AI 도구"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.ai = LLM.create(model)
    
    def check_spelling_grammar(self, text):
        """맞춤법과 문법을 검사합니다"""
        prompt = f"""
        다음 텍스트의 맞춤법과 문법을 검사해주세요.
        틀린 부분만 수정하고, 원문의 의미와 스타일은 유지해주세요.
        
        원문:
        {text}
        
        수정된 텍스트만 출력해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def show_corrections(self, text):
        """수정 사항을 자세히 보여줍니다"""
        prompt = f"""
        다음 텍스트의 맞춤법과 문법 오류를 찾아주세요.
        각 오류에 대해 설명도 함께 제공해주세요.
        
        텍스트:
        {text}
        
        다음 형식으로 답해주세요:
        1. [오류 부분] → [수정안]: 설명
        2. [오류 부분] → [수정안]: 설명
        ...
        
        오류가 없다면 "오류가 없습니다"라고 답해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
improver = TextImprover()

# 테스트 텍스트
text = """
안녕하세요 여러분 오늘은 파이썬으로 텍스트를 개선하는 방법에 대해 알아보겠습니다
많은 분들이 글쓰기를 어려워 하시는대 AI를 활용하면 쉽게 개선할수 있습니다
"""

# 맞춤법 검사
print("📝 원문:")
print(text)

print("\n✅ 수정된 텍스트:")
corrected = improver.check_spelling_grammar(text)
print(corrected)

print("\n🔍 수정 사항 설명:")
corrections = improver.show_corrections(text)
print(corrections)
```

### Step 2: 문체 개선기

```python
class StyleImprover(TextImprover):
    """문체를 개선하는 AI 도구"""
    
    def improve_style(self, text, style="formal"):
        """문체를 개선합니다"""
        style_guides = {
            "formal": "격식있고 전문적인 문체",
            "casual": "친근하고 편안한 문체",
            "academic": "학술적이고 객관적인 문체",
            "business": "비즈니스에 적합한 간결한 문체"
        }
        
        style_desc = style_guides.get(style, style)
        
        prompt = f"""
        다음 텍스트를 {style_desc}로 개선해주세요.
        내용은 유지하되 표현 방식만 바꿔주세요.
        
        원문:
        {text}
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def simplify(self, text):
        """복잡한 문장을 쉽게 만듭니다"""
        prompt = f"""
        다음 텍스트를 더 쉽고 명확하게 다시 써주세요.
        - 긴 문장은 짧게 나누기
        - 어려운 단어는 쉬운 단어로
        - 복잡한 구조는 단순하게
        
        원문:
        {text}
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def make_concise(self, text):
        """간결하게 만듭니다"""
        prompt = f"""
        다음 텍스트를 더 간결하게 만들어주세요.
        핵심 내용은 모두 유지하면서 불필요한 표현만 제거해주세요.
        
        원문:
        {text}
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
style_improver = StyleImprover()

# 원본 텍스트
original = """
제가 생각하기에는 이번 프로젝트가 성공적으로 완료되기 위해서는 
팀원들 간의 원활한 소통이 매우 중요하다고 봅니다. 
각자의 의견을 자유롭게 표현할 수 있는 분위기를 만들어야 할 것 같습니다.
"""

print("🎨 문체 변환 예시:\n")

# 다양한 문체로 변환
styles = ["formal", "casual", "business"]
for style in styles:
    print(f"### {style.upper()} 스타일:")
    result = style_improver.improve_style(original, style)
    print(result)
    print()

# 문장 단순화
print("### 단순화:")
simplified = style_improver.simplify(original)
print(simplified)

# 간결화
print("\n### 간결화:")
concise = style_improver.make_concise(original)
print(concise)
```

## 🎯 실용적인 글쓰기 도구

### 1. 이메일 작성 도우미

```python
class EmailWriter(TextImprover):
    """이메일 작성을 도와주는 AI"""
    
    def write_email(self, purpose, key_points, tone="professional"):
        """이메일 초안을 작성합니다"""
        prompt = f"""
        다음 정보를 바탕으로 이메일을 작성해주세요:
        
        목적: {purpose}
        주요 내용: {key_points}
        톤: {tone}
        
        한국 비즈니스 이메일 형식에 맞춰 작성해주세요.
        (인사말 - 본문 - 마무리 인사)
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def improve_email(self, draft):
        """이메일 초안을 개선합니다"""
        prompt = f"""
        다음 이메일 초안을 개선해주세요:
        - 더 명확하고 전문적으로
        - 예의바르고 정중하게
        - 핵심 메시지가 잘 전달되도록
        
        초안:
        {draft}
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def create_response(self, original_email, response_type="accept"):
        """이메일 답장을 작성합니다"""
        response_guides = {
            "accept": "긍정적으로 수락하는",
            "decline": "정중하게 거절하는",
            "request_info": "추가 정보를 요청하는",
            "follow_up": "후속 조치를 안내하는"
        }
        
        guide = response_guides.get(response_type, response_type)
        
        prompt = f"""
        다음 이메일에 대해 {guide} 답장을 작성해주세요:
        
        원본 이메일:
        {original_email}
        
        정중하고 전문적인 톤으로 작성해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
email_writer = EmailWriter()

# 이메일 작성
print("📧 이메일 초안 작성:")
email = email_writer.write_email(
    purpose="프로젝트 진행 상황 보고",
    key_points="1단계 완료, 2단계 진행 중 (70%), 예상 완료일 12월 15일",
    tone="professional"
)
print(email)

# 이메일 개선
print("\n📝 이메일 개선:")
draft = """
안녕하세요 김부장님
프로젝트 관련해서 말씀드립니다
1단계는 끝났고 2단계 하고있습니다
12월 15일쯤 끝날것 같습니다
"""
improved = email_writer.improve_email(draft)
print(improved)

# 답장 작성
print("\n💌 답장 작성:")
original = """
안녕하세요,
다음 주 화요일 오후 2시에 미팅이 가능하신지 확인 부탁드립니다.
프로젝트 진행 상황에 대해 논의하고자 합니다.
"""
reply = email_writer.create_response(original, "accept")
print(reply)
```

### 2. 보고서 작성 도우미

```python
class ReportWriter(TextImprover):
    """보고서 작성을 도와주는 AI"""
    
    def create_outline(self, topic, report_type="general"):
        """보고서 개요를 작성합니다"""
        prompt = f"""
        '{topic}'에 대한 {report_type} 보고서의 개요를 작성해주세요.
        
        다음 형식으로 작성해주세요:
        1. 제목
        2. 목차 (대제목과 소제목)
        3. 각 섹션별 주요 내용 요약
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def write_summary(self, content, max_length=200):
        """요약문을 작성합니다"""
        prompt = f"""
        다음 내용을 {max_length}자 이내로 요약해주세요.
        핵심 정보만 포함하고 명확하게 작성해주세요.
        
        내용:
        {content}
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def improve_paragraph(self, paragraph, focus="clarity"):
        """문단을 개선합니다"""
        focus_guides = {
            "clarity": "명확성을 높이도록",
            "flow": "문장 흐름이 자연스럽도록",
            "evidence": "근거와 예시를 추가하도록",
            "impact": "설득력을 높이도록"
        }
        
        guide = focus_guides.get(focus, focus)
        
        prompt = f"""
        다음 문단을 {guide} 개선해주세요:
        
        {paragraph}
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
report_writer = ReportWriter()

# 보고서 개요 작성
print("📋 보고서 개요:")
outline = report_writer.create_outline(
    topic="원격 근무의 생산성 영향",
    report_type="분석"
)
print(outline)

# 요약문 작성
print("\n📄 요약문 작성:")
long_content = """
최근 3개월간의 원격 근무 실시 결과, 직원들의 만족도는 85%로 매우 높게 나타났습니다.
생산성 지표를 분석한 결과, 개인 업무의 경우 평균 15% 향상되었으나,
팀 협업이 필요한 프로젝트의 경우 약 10% 감소한 것으로 나타났습니다.
이는 대면 소통의 부재로 인한 것으로 분석되며, 
온라인 협업 도구의 적극적인 활용과 주기적인 팀 미팅을 통해 개선이 가능할 것으로 보입니다.
"""
summary = report_writer.write_summary(long_content, max_length=100)
print(summary)

# 문단 개선
print("\n✨ 문단 개선:")
original_paragraph = """
원격 근무는 좋은 점도 있고 나쁜 점도 있습니다. 
집에서 일하니까 편하긴 한데 가끔 외롭기도 합니다.
"""
improved = report_writer.improve_paragraph(original_paragraph, "impact")
print(improved)
```

## 🔧 고급 기능

### 1. 다국어 글쓰기 도우미

```python
class MultilingualWriter(TextImprover):
    """다국어 글쓰기를 지원하는 AI"""
    
    def translate_and_improve(self, text, target_lang="English"):
        """번역하면서 개선합니다"""
        prompt = f"""
        다음 텍스트를 {target_lang}로 번역하면서
        더 자연스럽고 해당 언어의 관용구를 사용해 개선해주세요:
        
        {text}
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def check_translation(self, original, translated, source_lang="Korean", target_lang="English"):
        """번역의 정확성을 검토합니다"""
        prompt = f"""
        다음 {source_lang} 원문과 {target_lang} 번역을 비교해주세요:
        
        원문: {original}
        번역: {translated}
        
        평가 항목:
        1. 의미 전달의 정확성 (1-10점)
        2. 자연스러움 (1-10점)
        3. 개선 제안
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
multilingual = MultilingualWriter()

korean_text = "이번 프로젝트를 통해 많은 것을 배웠습니다. 특히 팀워크의 중요성을 깨달았습니다."

# 번역 및 개선
print("🌍 번역 및 개선:")
english = multilingual.translate_and_improve(korean_text, "English")
print(f"English: {english}")

japanese = multilingual.translate_and_improve(korean_text, "Japanese")
print(f"日本語: {japanese}")

# 번역 검토
print("\n🔍 번역 품질 검토:")
review = multilingual.check_translation(korean_text, english)
print(review)
```

### 2. SEO 최적화 도우미

```python
class SEOWriter(TextImprover):
    """SEO에 최적화된 글쓰기 도우미"""
    
    def optimize_for_seo(self, content, keywords):
        """SEO를 위해 콘텐츠를 최적화합니다"""
        prompt = f"""
        다음 콘텐츠를 SEO에 최적화해주세요:
        
        콘텐츠: {content}
        타겟 키워드: {keywords}
        
        요구사항:
        - 키워드를 자연스럽게 포함
        - 가독성 유지
        - 제목과 소제목 구조화
        - 메타 설명 제안
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def create_meta_description(self, content, max_length=155):
        """메타 설명을 생성합니다"""
        prompt = f"""
        다음 콘텐츠의 메타 설명을 {max_length}자 이내로 작성해주세요.
        검색 결과에서 클릭을 유도할 수 있도록 매력적으로 작성해주세요:
        
        {content}
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
seo_writer = SEOWriter()

content = """
파이썬은 배우기 쉽고 강력한 프로그래밍 언어입니다.
초보자도 쉽게 시작할 수 있으며, 웹 개발, 데이터 분석, AI 등 다양한 분야에서 활용됩니다.
"""

# SEO 최적화
print("🔍 SEO 최적화:")
optimized = seo_writer.optimize_for_seo(
    content,
    keywords=["파이썬 프로그래밍", "파이썬 입문", "파이썬 배우기"]
)
print(optimized)

# 메타 설명 생성
print("\n📝 메타 설명:")
meta = seo_writer.create_meta_description(content)
print(meta)
```

## 📊 종합 텍스트 분석기

```python
class TextAnalyzer(TextImprover):
    """텍스트를 종합적으로 분석하는 도구"""
    
    def analyze_text(self, text):
        """텍스트를 다각도로 분석합니다"""
        prompt = f"""
        다음 텍스트를 분석해주세요:
        
        {text}
        
        분석 항목:
        1. 문체와 톤
        2. 가독성 수준 (초급/중급/고급)
        3. 주요 키워드 3개
        4. 개선이 필요한 부분
        5. 전체적인 평가 (1-10점)
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def compare_texts(self, text1, text2):
        """두 텍스트를 비교합니다"""
        prompt = f"""
        다음 두 텍스트를 비교 분석해주세요:
        
        텍스트 1:
        {text1}
        
        텍스트 2:
        {text2}
        
        비교 항목:
        - 문체와 톤의 차이
        - 정보 전달의 명확성
        - 설득력
        - 각각의 장단점
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
analyzer = TextAnalyzer()

# 텍스트 분석
sample_text = """
인공지능 기술의 발전으로 우리의 일상생활이 크게 변화하고 있습니다.
스마트폰의 음성 비서부터 자율주행 자동차까지, 
AI는 이미 우리 생활 곳곳에 스며들어 있습니다.
이러한 변화는 편리함을 가져다주지만, 
동시에 일자리 감소와 같은 우려도 제기되고 있습니다.
"""

print("📊 텍스트 분석 결과:")
analysis = analyzer.analyze_text(sample_text)
print(analysis)

# 텍스트 비교
text_a = "AI는 인류에게 큰 도움이 될 것입니다."
text_b = "AI 기술이 가져올 혜택은 무궁무진하며, 인류 발전에 핵심적인 역할을 할 것으로 예상됩니다."

print("\n🔍 텍스트 비교:")
comparison = analyzer.compare_texts(text_a, text_b)
print(comparison)
```

## ✅ 핵심 정리

1. **맞춤법/문법 검사**로 기본적인 오류 수정
2. **문체 개선**으로 목적에 맞는 글쓰기
3. **특수 목적 글쓰기** (이메일, 보고서, SEO)
4. **텍스트 분석**으로 글의 품질 평가

## 🚀 다음 단계

텍스트 개선 도구를 만들었으니, 이제 [다국어 번역기](translator.md)를 만들어 언어의 장벽을 넘어봅시다!