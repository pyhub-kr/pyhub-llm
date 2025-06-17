# 📄 똑똑한 요약기 만들기

긴 텍스트를 핵심만 담아 간결하게 요약하는 AI 도구를 만들어봅시다!

## 🎯 만들 것

```
원문: (3000자 긴 기사)
요약: 
• 핵심 포인트 1
• 핵심 포인트 2  
• 핵심 포인트 3
```

## 📝 기본 요약기

### Step 1: 간단한 텍스트 요약기

```python
# summarizer.py
from pyhub.llm import LLM

class TextSummarizer:
    """텍스트 요약 AI 도구"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.ai = LLM.create(model)
    
    def summarize(self, text, max_length=200):
        """텍스트를 요약합니다"""
        prompt = f"""
        다음 텍스트를 {max_length}자 이내로 요약해주세요.
        핵심 내용만 포함하고 명확하게 작성해주세요.
        
        원문:
        {text}
        
        요약:
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def bullet_points(self, text, num_points=5):
        """핵심 포인트를 추출합니다"""
        prompt = f"""
        다음 텍스트에서 가장 중요한 {num_points}개의 핵심 포인트를 추출해주세요.
        
        원문:
        {text}
        
        각 포인트는 한 줄로 간단명료하게 작성해주세요:
        • 포인트 1
        • 포인트 2
        ...
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def one_line_summary(self, text):
        """한 줄 요약을 생성합니다"""
        prompt = f"""
        다음 텍스트를 한 문장으로 요약해주세요.
        가장 핵심적인 내용만 담아주세요.
        
        원문:
        {text}
        
        한 줄 요약:
        """
        
        response = self.ai.ask(prompt)
        return response.text.strip()

# 사용 예시
summarizer = TextSummarizer()

# 샘플 텍스트
long_text = """
인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현하려는 
컴퓨터 과학의 한 분야입니다. 1950년대에 처음 등장한 이후, AI는 놀라운 속도로 
발전해왔습니다. 특히 최근 딥러닝 기술의 발전으로 이미지 인식, 자연어 처리, 
게임 플레이 등 다양한 분야에서 인간 수준을 뛰어넘는 성능을 보여주고 있습니다.

AI의 발전은 우리 일상생활에도 큰 변화를 가져왔습니다. 스마트폰의 음성 비서, 
추천 시스템, 자율주행 자동차 등이 대표적인 예입니다. 의료 분야에서는 AI가 
질병 진단을 돕고, 금융 분야에서는 사기 탐지와 투자 분석에 활용됩니다.

그러나 AI의 급속한 발전은 새로운 과제도 가져왔습니다. 일자리 대체, 
프라이버시 침해, AI의 편향성 문제 등이 사회적 이슈로 대두되고 있습니다. 
따라서 AI 기술을 발전시키는 동시에 윤리적이고 책임감 있는 AI 개발이 
중요해지고 있습니다.
"""

# 다양한 요약 방식
print("📝 일반 요약 (200자):")
print(summarizer.summarize(long_text, max_length=200))

print("\n🎯 핵심 포인트:")
print(summarizer.bullet_points(long_text, num_points=3))

print("\n📌 한 줄 요약:")
print(summarizer.one_line_summary(long_text))
```

### Step 2: 고급 요약 기능

```python
class AdvancedSummarizer(TextSummarizer):
    """고급 요약 기능을 갖춘 도구"""
    
    def summarize_by_sections(self, text, section_delimiter="\n\n"):
        """섹션별로 요약합니다"""
        sections = text.split(section_delimiter)
        summaries = []
        
        for i, section in enumerate(sections, 1):
            if section.strip():
                summary = self.summarize(section, max_length=100)
                summaries.append(f"섹션 {i}: {summary}")
        
        return "\n".join(summaries)
    
    def progressive_summary(self, text, levels=[500, 200, 50]):
        """단계별로 점진적으로 요약합니다"""
        results = {}
        current_text = text
        
        for level in levels:
            summary = self.summarize(current_text, max_length=level)
            results[f"{level}자 요약"] = summary
            current_text = summary  # 다음 단계의 입력으로 사용
        
        return results
    
    def extract_key_information(self, text, info_types):
        """특정 정보를 추출합니다"""
        info_prompts = {
            "people": "등장하는 인물이나 조직",
            "dates": "날짜나 시간 정보",
            "numbers": "숫자나 통계 데이터",
            "locations": "장소나 지역 정보",
            "events": "중요한 사건이나 이벤트",
            "conclusions": "결론이나 핵심 메시지"
        }
        
        results = {}
        for info_type in info_types:
            if info_type in info_prompts:
                prompt = f"""
                다음 텍스트에서 {info_prompts[info_type]}만 추출해주세요:
                
                {text}
                
                없으면 "해당 없음"이라고 답해주세요.
                """
                
                response = self.ai.ask(prompt)
                results[info_type] = response.text
        
        return results

# 사용 예시
adv_summarizer = AdvancedSummarizer()

# 섹션별 요약
print("📑 섹션별 요약:")
section_summary = adv_summarizer.summarize_by_sections(long_text)
print(section_summary)

# 단계별 요약
print("\n📊 단계별 요약:")
progressive = adv_summarizer.progressive_summary(long_text)
for level, summary in progressive.items():
    print(f"\n[{level}]")
    print(summary)

# 정보 추출
print("\n🔍 핵심 정보 추출:")
extracted = adv_summarizer.extract_key_information(
    long_text,
    ["numbers", "events", "conclusions"]
)
for info_type, content in extracted.items():
    print(f"\n{info_type.upper()}:")
    print(content)
```

## 🎨 특화된 요약 도구

### 1. 뉴스 기사 요약기

```python
class NewsArticleSummarizer(AdvancedSummarizer):
    """뉴스 기사 전문 요약 도구"""
    
    def summarize_news(self, article):
        """뉴스 기사를 요약합니다"""
        prompt = f"""
        다음 뉴스 기사를 요약해주세요.
        
        기사:
        {article}
        
        다음 형식으로 요약해주세요:
        
        📰 헤드라인: (핵심 내용을 한 줄로)
        
        📋 주요 내용:
        • (핵심 사실 1)
        • (핵심 사실 2)
        • (핵심 사실 3)
        
        🔍 배경:
        (간단한 배경 설명)
        
        💡 시사점:
        (이 뉴스의 의미나 영향)
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def create_news_brief(self, articles):
        """여러 기사를 하나의 브리핑으로 만듭니다"""
        prompt = "다음 뉴스들을 종합하여 오늘의 주요 뉴스 브리핑을 작성해주세요:\n\n"
        
        for i, article in enumerate(articles, 1):
            prompt += f"기사 {i}:\n{article[:500]}...\n\n"
        
        prompt += """
        브리핑 형식:
        🗞️ 오늘의 주요 뉴스
        
        1. [제목] - 요약
        2. [제목] - 요약
        ...
        
        📊 오늘의 트렌드:
        (전체적인 동향 분석)
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
news_summarizer = NewsArticleSummarizer()

# 뉴스 기사 예시
news_article = """
[속보] 정부, AI 산업 육성 위해 10조원 투자 계획 발표

정부가 인공지능(AI) 산업 육성을 위해 향후 5년간 10조원을 투자한다고 발표했다. 
이번 투자는 AI 인프라 구축, 인재 양성, 기업 지원 등 세 가지 축을 중심으로 진행될 예정이다.

과학기술정보통신부는 오늘 'AI 강국 도약 전략'을 발표하며, 2030년까지 글로벌 AI 
선도국가로 도약하겠다는 목표를 제시했다. 주요 내용으로는 AI 전문인력 10만명 양성, 
AI 스타트업 1000개 육성, 공공 분야 AI 도입 확대 등이 포함됐다.

업계에서는 이번 정부의 대규모 투자가 국내 AI 생태계 활성화에 크게 기여할 것으로 
기대하고 있다. 특히 중소기업과 스타트업에 대한 지원이 확대되어 AI 기술 개발과 
사업화가 가속화될 전망이다.
"""

print("📰 뉴스 요약:")
news_summary = news_summarizer.summarize_news(news_article)
print(news_summary)

# 여러 기사 브리핑
print("\n📢 종합 뉴스 브리핑:")
articles = [
    "AI 투자 10조원 발표...",
    "삼성전자, 새로운 AI 칩 개발...",
    "네이버, AI 검색 서비스 출시..."
]
brief = news_summarizer.create_news_brief(articles)
print(brief)
```

### 2. 회의록 요약기

```python
class MeetingSummarizer(AdvancedSummarizer):
    """회의록 요약 전문 도구"""
    
    def summarize_meeting(self, transcript):
        """회의록을 요약합니다"""
        prompt = f"""
        다음 회의록을 요약해주세요:
        
        {transcript}
        
        다음 형식으로 정리해주세요:
        
        📅 회의 개요
        - 주제:
        - 참석자:
        - 일시:
        
        🎯 주요 안건
        1. 
        2. 
        
        💬 주요 논의사항
        • 
        • 
        
        ✅ 결정사항
        • 
        • 
        
        📋 액션 아이템
        • [담당자] 할 일 (기한)
        • [담당자] 할 일 (기한)
        
        📌 다음 회의
        - 일시:
        - 안건:
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def extract_action_items(self, transcript):
        """액션 아이템만 추출합니다"""
        prompt = f"""
        다음 회의록에서 액션 아이템(할 일)만 추출해주세요:
        
        {transcript}
        
        형식:
        1. [담당자] 할 일 내용 (기한)
        2. [담당자] 할 일 내용 (기한)
        
        담당자가 명확하지 않으면 [미정]으로 표시해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def create_meeting_minutes(self, notes):
        """메모를 정식 회의록으로 변환합니다"""
        prompt = f"""
        다음 회의 메모를 정식 회의록 형식으로 작성해주세요:
        
        메모:
        {notes}
        
        전문적이고 공식적인 문체로 작성하되,
        핵심 내용은 모두 포함해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
meeting_summarizer = MeetingSummarizer()

# 회의록 예시
meeting_transcript = """
김부장: 이번 프로젝트 진행 상황을 공유하겠습니다. 현재 1단계는 완료했고, 2단계 진행 중입니다.
이과장: 2단계에서 예상보다 시간이 걸리고 있는데, 추가 인력이 필요할 것 같습니다.
박팀장: 개발팀에서 2명 지원 가능합니다. 다음 주부터 투입하겠습니다.
김부장: 좋습니다. 그럼 이과장님은 다음 주까지 상세 계획서를 작성해 주세요.
이과장: 알겠습니다. 금요일까지 제출하겠습니다.
박팀장: 저는 개발 환경 세팅을 미리 준비해두겠습니다.
김부장: 다음 회의는 다음 주 수요일 오후 2시에 하겠습니다.
"""

print("📋 회의록 요약:")
meeting_summary = meeting_summarizer.summarize_meeting(meeting_transcript)
print(meeting_summary)

print("\n✅ 액션 아이템:")
actions = meeting_summarizer.extract_action_items(meeting_transcript)
print(actions)
```

### 3. 학술 논문 요약기

```python
class AcademicSummarizer(AdvancedSummarizer):
    """학술 논문 요약 도구"""
    
    def summarize_paper(self, paper_text):
        """논문을 요약합니다"""
        prompt = f"""
        다음 학술 논문을 요약해주세요:
        
        {paper_text}
        
        다음 구조로 요약해주세요:
        
        📚 제목:
        
        🎯 연구 목적:
        (이 연구가 해결하려는 문제)
        
        🔬 방법론:
        (어떻게 연구했는지)
        
        📊 주요 결과:
        • 발견 1
        • 발견 2
        
        💡 결론:
        (연구의 의미와 기여)
        
        🔍 한계점:
        (연구의 제한사항)
        
        📖 핵심 키워드:
        #키워드1 #키워드2 #키워드3
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def create_literature_review(self, papers):
        """여러 논문을 종합한 문헌 검토를 작성합니다"""
        prompt = "다음 논문들을 종합하여 문헌 검토를 작성해주세요:\n\n"
        
        for i, paper in enumerate(papers, 1):
            prompt += f"논문 {i}:\n{paper['title']}\n{paper['summary']}\n\n"
        
        prompt += """
        문헌 검토 형식:
        1. 연구 동향
        2. 주요 발견사항
        3. 연구 간 공통점과 차이점
        4. 향후 연구 방향
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
academic_summarizer = AcademicSummarizer()

# 논문 요약 (예시)
paper_abstract = """
This study investigates the impact of artificial intelligence on workplace productivity. 
Using a mixed-methods approach, we surveyed 500 companies and conducted in-depth 
interviews with 50 managers. Our findings show that AI implementation led to an 
average 25% increase in productivity, particularly in data analysis and customer 
service tasks. However, we also identified challenges including employee resistance 
and the need for extensive training. The study concludes that successful AI adoption 
requires careful change management and continuous learning programs.
"""

print("🎓 논문 요약:")
paper_summary = academic_summarizer.summarize_paper(paper_abstract)
print(paper_summary)
```

## 📊 요약 품질 평가기

```python
class SummaryEvaluator(TextSummarizer):
    """요약 품질을 평가하는 도구"""
    
    def evaluate_summary(self, original, summary):
        """요약의 품질을 평가합니다"""
        prompt = f"""
        원문과 요약을 비교하여 요약의 품질을 평가해주세요.
        
        원문:
        {original}
        
        요약:
        {summary}
        
        다음 기준으로 평가해주세요 (각 항목 1-10점):
        1. 정확성: 원문의 내용을 정확히 반영했는가?
        2. 완전성: 중요한 정보가 빠지지 않았는가?
        3. 간결성: 불필요한 내용 없이 간결한가?
        4. 가독성: 읽기 쉽고 이해하기 쉬운가?
        5. 일관성: 논리적 흐름이 일관되는가?
        
        각 점수와 총평을 제공해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def compare_summaries(self, original, summaries):
        """여러 요약을 비교합니다"""
        prompt = f"원문:\n{original}\n\n"
        
        for i, summary in enumerate(summaries, 1):
            prompt += f"요약 {i}:\n{summary}\n\n"
        
        prompt += """
        위 요약들을 비교하여 어떤 요약이 가장 좋은지 평가해주세요.
        각 요약의 장단점을 분석하고 순위를 매겨주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text

# 사용 예시
evaluator = SummaryEvaluator()

# 원문
original = "AI는 인간의 지능을 모방한 기술로, 최근 급속히 발전하고 있습니다..."

# 다양한 요약 생성
summary1 = summarizer.summarize(original, 50)
summary2 = summarizer.one_line_summary(original)

# 요약 평가
print("📊 요약 품질 평가:")
evaluation = evaluator.evaluate_summary(original, summary1)
print(evaluation)

# 요약 비교
print("\n🔍 요약 비교:")
comparison = evaluator.compare_summaries(original, [summary1, summary2])
print(comparison)
```

## 🚀 통합 요약 시스템

```python
class IntegratedSummarizer:
    """다양한 요약 기능을 통합한 시스템"""
    
    def __init__(self):
        self.general = TextSummarizer()
        self.news = NewsArticleSummarizer()
        self.meeting = MeetingSummarizer()
        self.academic = AcademicSummarizer()
    
    def auto_summarize(self, text, text_type="auto"):
        """텍스트 유형을 자동 감지하여 요약합니다"""
        if text_type == "auto":
            # 텍스트 유형 자동 감지
            text_type = self._detect_text_type(text)
        
        # 유형별 요약 수행
        if text_type == "news":
            return self.news.summarize_news(text)
        elif text_type == "meeting":
            return self.meeting.summarize_meeting(text)
        elif text_type == "academic":
            return self.academic.summarize_paper(text)
        else:
            return self.general.summarize(text)
    
    def _detect_text_type(self, text):
        """텍스트 유형을 감지합니다"""
        prompt = f"""
        다음 텍스트의 유형을 판단해주세요:
        
        {text[:500]}...
        
        다음 중 하나로만 답해주세요:
        - news (뉴스 기사)
        - meeting (회의록)
        - academic (학술 논문)
        - general (일반 텍스트)
        """
        
        response = self.general.ai.ask(prompt)
        return response.text.strip().lower()
    
    def batch_summarize(self, texts, output_format="markdown"):
        """여러 텍스트를 일괄 요약합니다"""
        results = []
        
        for i, text in enumerate(texts, 1):
            print(f"요약 중... ({i}/{len(texts)})")
            
            # 자동 요약
            summary = self.auto_summarize(text)
            
            # 결과 저장
            results.append({
                "index": i,
                "original_length": len(text),
                "summary": summary,
                "reduction_rate": f"{(1 - len(summary)/len(text))*100:.1f}%"
            })
        
        # 결과 포맷팅
        if output_format == "markdown":
            return self._format_markdown(results)
        else:
            return results
    
    def _format_markdown(self, results):
        """결과를 마크다운으로 포맷팅합니다"""
        output = "# 요약 결과\n\n"
        
        for r in results:
            output += f"## 문서 {r['index']}\n"
            output += f"- 원문 길이: {r['original_length']}자\n"
            output += f"- 압축률: {r['reduction_rate']}\n\n"
            output += f"### 요약\n{r['summary']}\n\n"
            output += "---\n\n"
        
        return output

# 사용 예시
integrated = IntegratedSummarizer()

# 자동 요약
print("🤖 자동 요약:")
auto_summary = integrated.auto_summarize(news_article)
print(auto_summary)

# 일괄 요약
print("\n📦 일괄 요약:")
texts = [news_article, meeting_transcript, paper_abstract]
batch_results = integrated.batch_summarize(texts)
print(batch_results)
```

## ✅ 핵심 정리

1. **다양한 요약 방식** - 일반, 핵심 포인트, 한 줄 요약
2. **특화된 요약** - 뉴스, 회의록, 논문별 전문 요약
3. **요약 품질 평가** - 정확성, 완전성, 간결성 평가
4. **통합 시스템** - 자동 감지 및 일괄 처리

## 🚀 다음 단계

일상 작업 자동화 도구들을 모두 만들었습니다! 이제 [대화 이어가기](../03-conversations/)로 넘어가 더 복잡한 대화형 AI를 만들어봅시다!