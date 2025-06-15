#!/usr/bin/env python3
"""
예제: 문서 요약기
난이도: 고급
설명: 다양한 형식의 문서를 요약하는 고급 시스템
요구사항: 
  - pyhub-llm (pip install pyhub-llm)
  - PyPDF2 (pip install PyPDF2)
  - python-docx (pip install python-docx)
  - beautifulsoup4 (pip install beautifulsoup4)
  - OPENAI_API_KEY 환경 변수
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json

from pyhub.llm import LLM
from pyhub.llm.cache import FileCache
from pyhub.llm.templates.engine import TemplateEngine


class DocumentType(Enum):
    """문서 타입"""
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class DocumentSection:
    """문서 섹션"""
    title: str
    content: str
    level: int
    word_count: int
    position: int


@dataclass
class SummaryResult:
    """요약 결과"""
    executive_summary: str
    key_points: List[str]
    section_summaries: Dict[str, str]
    statistics: Dict[str, Any]
    recommendations: Optional[List[str]] = None


class DocumentSummarizer:
    """문서 요약기"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = LLM.create(model)
        self.cache = FileCache("summaries")
        self.template_engine = TemplateEngine()
        self._setup_templates()
    
    def _setup_templates(self):
        """템플릿 설정"""
        # 요약 템플릿
        self.template_engine.add_template(
            "summarize",
            """다음 텍스트를 요약하세요:

제목: {{ title }}
길이: {{ word_count }}단어

내용:
{{ content }}

요구사항:
- {{ style }} 스타일로 요약
- 핵심 포인트 {{ num_points }}개 추출
- 최대 {{ max_words }}단어로 요약"""
        )
        
        # 섹션 분석 템플릿
        self.template_engine.add_template(
            "analyze_section",
            """다음 섹션을 분석하세요:

{{ section_content }}

다음 형식으로 응답하세요:
1. 주제: (한 문장)
2. 핵심 내용: (2-3문장)
3. 중요도: (상/중/하)
4. 키워드: (쉼표로 구분)"""
        )
    
    def load_document(self, file_path: str) -> Tuple[str, DocumentType]:
        """문서 로드"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.txt':
            return self._load_text(file_path), DocumentType.TEXT
        elif suffix == '.pdf':
            return self._load_pdf(file_path), DocumentType.PDF
        elif suffix == '.docx':
            return self._load_docx(file_path), DocumentType.DOCX
        elif suffix in ['.html', '.htm']:
            return self._load_html(file_path), DocumentType.HTML
        elif suffix == '.md':
            return self._load_markdown(file_path), DocumentType.MARKDOWN
        else:
            # 기본적으로 텍스트로 처리
            return self._load_text(file_path), DocumentType.TEXT
    
    def _load_text(self, file_path: str) -> str:
        """텍스트 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_pdf(self, file_path: str) -> str:
        """PDF 파일 로드"""
        try:
            import PyPDF2
            
            text = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text.append(page.extract_text())
            
            return '\n'.join(text)
        except ImportError:
            raise ImportError("PyPDF2가 설치되지 않았습니다. pip install PyPDF2")
    
    def _load_docx(self, file_path: str) -> str:
        """DOCX 파일 로드"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            text = []
            
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            
            return '\n'.join(text)
        except ImportError:
            raise ImportError("python-docx가 설치되지 않았습니다. pip install python-docx")
    
    def _load_html(self, file_path: str) -> str:
        """HTML 파일 로드"""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                
                # 스크립트와 스타일 제거
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # 텍스트 추출
                text = soup.get_text()
                
                # 공백 정리
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                return text
        except ImportError:
            raise ImportError("beautifulsoup4가 설치되지 않았습니다. pip install beautifulsoup4")
    
    def _load_markdown(self, file_path: str) -> str:
        """마크다운 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_sections(self, content: str, doc_type: DocumentType) -> List[DocumentSection]:
        """문서에서 섹션 추출"""
        sections = []
        
        if doc_type == DocumentType.MARKDOWN:
            # 마크다운 헤더 기반 섹션 분리
            pattern = r'^(#{1,6})\s+(.+)$'
            current_section = None
            current_content = []
            position = 0
            
            for line in content.split('\n'):
                match = re.match(pattern, line)
                if match:
                    # 이전 섹션 저장
                    if current_section:
                        content_text = '\n'.join(current_content).strip()
                        if content_text:
                            current_section.content = content_text
                            current_section.word_count = len(content_text.split())
                            sections.append(current_section)
                    
                    # 새 섹션 시작
                    level = len(match.group(1))
                    title = match.group(2)
                    current_section = DocumentSection(
                        title=title,
                        content="",
                        level=level,
                        word_count=0,
                        position=position
                    )
                    current_content = []
                    position += 1
                else:
                    current_content.append(line)
            
            # 마지막 섹션 저장
            if current_section and current_content:
                content_text = '\n'.join(current_content).strip()
                if content_text:
                    current_section.content = content_text
                    current_section.word_count = len(content_text.split())
                    sections.append(current_section)
        
        else:
            # 기본: 단락 기반 분리
            paragraphs = re.split(r'\n\n+', content)
            for i, para in enumerate(paragraphs):
                if para.strip():
                    sections.append(DocumentSection(
                        title=f"섹션 {i+1}",
                        content=para.strip(),
                        level=1,
                        word_count=len(para.split()),
                        position=i
                    ))
        
        return sections
    
    def summarize_section(self, section: DocumentSection, style: str = "concise") -> str:
        """섹션 요약"""
        # 캐시 확인
        cache_key = f"section_{section.title}_{style}_{section.word_count}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # 짧은 섹션은 그대로 반환
        if section.word_count < 50:
            return section.content
        
        # 요약 생성
        prompt = self.template_engine.render_template(
            "summarize",
            title=section.title,
            content=section.content,
            word_count=section.word_count,
            style=style,
            num_points=3,
            max_words=100
        )
        
        reply = self.llm.ask(prompt)
        summary = reply.text
        
        # 캐시 저장
        self.cache.set(cache_key, summary)
        
        return summary
    
    def generate_executive_summary(
        self,
        sections: List[DocumentSection],
        max_words: int = 300
    ) -> str:
        """전체 요약 생성"""
        # 섹션 요약들을 합침
        section_summaries = []
        for section in sections[:10]:  # 최대 10개 섹션
            summary = self.summarize_section(section)
            section_summaries.append(f"{section.title}: {summary}")
        
        combined = "\n".join(section_summaries)
        
        # 전체 요약 생성
        prompt = f"""다음 섹션 요약들을 바탕으로 전체 문서의 핵심 요약을 작성하세요.
최대 {max_words}단어로 작성하고, 가장 중요한 내용을 포함하세요.

섹션 요약:
{combined}

전체 요약:"""
        
        reply = self.llm.ask(prompt)
        return reply.text
    
    def extract_key_points(self, content: str, num_points: int = 5) -> List[str]:
        """핵심 포인트 추출"""
        prompt = f"""다음 텍스트에서 가장 중요한 {num_points}개의 핵심 포인트를 추출하세요.
각 포인트는 한 문장으로 작성하세요.

텍스트:
{content[:3000]}  # 토큰 제한

형식:
1. 첫 번째 핵심 포인트
2. 두 번째 핵심 포인트
..."""
        
        reply = self.llm.ask(prompt)
        
        # 응답 파싱
        points = []
        for line in reply.text.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                point = re.sub(r'^\d+\.\s*', '', line)
                points.append(point)
        
        return points[:num_points]
    
    def analyze_document_structure(self, sections: List[DocumentSection]) -> Dict[str, Any]:
        """문서 구조 분석"""
        total_words = sum(s.word_count for s in sections)
        
        # 섹션별 통계
        section_stats = {}
        for section in sections:
            if section.level not in section_stats:
                section_stats[section.level] = {
                    "count": 0,
                    "total_words": 0,
                    "avg_words": 0
                }
            
            stats = section_stats[section.level]
            stats["count"] += 1
            stats["total_words"] += section.word_count
        
        # 평균 계산
        for level_stats in section_stats.values():
            if level_stats["count"] > 0:
                level_stats["avg_words"] = level_stats["total_words"] / level_stats["count"]
        
        return {
            "total_sections": len(sections),
            "total_words": total_words,
            "avg_words_per_section": total_words / len(sections) if sections else 0,
            "section_hierarchy": section_stats,
            "depth": max(s.level for s in sections) if sections else 0
        }
    
    def generate_recommendations(self, summary_result: SummaryResult) -> List[str]:
        """문서 개선 추천사항 생성"""
        recommendations = []
        stats = summary_result.statistics
        
        # 문서 길이 관련
        if stats["total_words"] > 5000:
            recommendations.append("문서가 매우 깁니다. 핵심 내용만 담은 요약본을 별도로 제공하는 것을 고려하세요.")
        elif stats["total_words"] < 500:
            recommendations.append("문서가 짧습니다. 더 자세한 설명이나 예시를 추가하는 것을 고려하세요.")
        
        # 구조 관련
        if stats["depth"] > 4:
            recommendations.append("문서 구조가 복잡합니다. 섹션 계층을 단순화하는 것을 고려하세요.")
        
        if stats["avg_words_per_section"] > 500:
            recommendations.append("섹션이 너무 깁니다. 하위 섹션으로 분할하는 것을 고려하세요.")
        
        # AI 분석 기반 추천
        prompt = f"""다음 문서 요약을 바탕으로 문서 개선을 위한 추천사항 2-3개를 제시하세요:

요약: {summary_result.executive_summary}

주요 포인트:
{chr(10).join(f"- {point}" for point in summary_result.key_points[:3])}

추천사항 (각각 한 문장):**"""
        
        reply = self.llm.ask(prompt)
        
        # 응답 파싱
        for line in reply.text.split('\n'):
            line = line.strip()
            if line and not line.startswith('추천사항'):
                recommendations.append(line.lstrip('- ').lstrip('• '))
        
        return recommendations[:5]  # 최대 5개
    
    def summarize(
        self,
        file_path: str,
        style: str = "detailed",
        include_recommendations: bool = True
    ) -> SummaryResult:
        """문서 요약 메인 함수"""
        print(f"📄 문서 로드 중: {file_path}")
        
        # 문서 로드
        content, doc_type = self.load_document(file_path)
        print(f"  문서 타입: {doc_type.value}")
        
        # 섹션 추출
        sections = self.extract_sections(content, doc_type)
        print(f"  섹션 수: {len(sections)}")
        
        # 구조 분석
        statistics = self.analyze_document_structure(sections)
        print(f"  총 단어 수: {statistics['total_words']:,}")
        
        # 섹션별 요약
        print("  섹션 요약 중...")
        section_summaries = {}
        for section in sections[:20]:  # 최대 20개 섹션
            summary = self.summarize_section(section, style)
            section_summaries[section.title] = summary
        
        # 전체 요약
        print("  전체 요약 생성 중...")
        executive_summary = self.generate_executive_summary(sections)
        
        # 핵심 포인트
        print("  핵심 포인트 추출 중...")
        key_points = self.extract_key_points(content)
        
        # 결과 생성
        result = SummaryResult(
            executive_summary=executive_summary,
            key_points=key_points,
            section_summaries=section_summaries,
            statistics=statistics
        )
        
        # 추천사항
        if include_recommendations:
            print("  추천사항 생성 중...")
            result.recommendations = self.generate_recommendations(result)
        
        print("✅ 요약 완료!")
        return result


def example_text_summary():
    """텍스트 파일 요약 예제"""
    print("\n📄 텍스트 파일 요약")
    print("-" * 50)
    
    # 예제 텍스트 파일 생성
    sample_text = """# 인공지능의 미래

## 서론
인공지능(AI)은 21세기의 가장 중요한 기술 혁신 중 하나로 자리잡았습니다. 
기계학습과 딥러닝의 발전으로 AI는 이제 우리 일상생활의 많은 부분에 영향을 미치고 있습니다.

## 현재 상황
### 기술 발전
최근 몇 년간 AI 기술은 놀라운 속도로 발전했습니다. 
특히 자연어 처리와 컴퓨터 비전 분야에서 인간 수준의 성능을 달성했습니다.

### 산업 응용
AI는 의료, 금융, 제조, 교육 등 다양한 산업 분야에서 활용되고 있습니다.
각 분야에서 효율성 증대와 새로운 가치 창출을 이끌어내고 있습니다.

## 미래 전망
### 기술적 진보
앞으로 AI는 더욱 정교하고 범용적인 형태로 발전할 것으로 예상됩니다.
AGI(Artificial General Intelligence)의 실현 가능성도 논의되고 있습니다.

### 사회적 영향
AI의 발전은 일자리 구조, 교육 시스템, 사회 제도 전반에 큰 변화를 가져올 것입니다.
이에 대한 준비와 대응이 필요합니다.

## 결론
AI는 인류에게 큰 기회이자 도전입니다. 
기술의 발전과 함께 윤리적, 사회적 고려사항을 균형있게 다루어야 합니다."""
    
    # 임시 파일 생성
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_text)
        temp_path = f.name
    
    try:
        # 요약기 생성
        summarizer = DocumentSummarizer()
        
        # 요약 실행
        result = summarizer.summarize(temp_path, style="concise")
        
        # 결과 출력
        print(f"\n📋 전체 요약:")
        print(result.executive_summary)
        
        print(f"\n🎯 핵심 포인트:")
        for i, point in enumerate(result.key_points, 1):
            print(f"  {i}. {point}")
        
        print(f"\n📊 문서 통계:")
        print(f"  - 총 단어 수: {result.statistics['total_words']:,}")
        print(f"  - 섹션 수: {result.statistics['total_sections']}")
        print(f"  - 문서 깊이: {result.statistics['depth']}")
        
        if result.recommendations:
            print(f"\n💡 개선 추천사항:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")
    
    finally:
        # 임시 파일 삭제
        os.unlink(temp_path)


def example_batch_processing():
    """배치 처리 예제"""
    print("\n📚 배치 문서 처리")
    print("-" * 50)
    
    # 여러 문서 생성
    documents = {
        "report1.txt": "This is a technical report about machine learning...",
        "report2.txt": "Financial analysis for Q4 2024...",
        "report3.txt": "Marketing strategy proposal..."
    }
    
    summarizer = DocumentSummarizer()
    results = {}
    
    # 임시 파일들 생성 및 처리
    import tempfile
    temp_files = []
    
    try:
        for filename, content in documents.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content * 50)  # 내용 확장
                temp_files.append(f.name)
                
                print(f"\n처리 중: {filename}")
                result = summarizer.summarize(f.name, style="brief", include_recommendations=False)
                results[filename] = result
        
        # 전체 결과 요약
        print("\n📊 배치 처리 결과:")
        for filename, result in results.items():
            print(f"\n{filename}:")
            print(f"  요약: {result.executive_summary[:100]}...")
            print(f"  단어 수: {result.statistics['total_words']:,}")
    
    finally:
        # 임시 파일들 삭제
        for temp_file in temp_files:
            os.unlink(temp_file)


def example_export_results():
    """결과 내보내기 예제"""
    print("\n💾 요약 결과 내보내기")
    print("-" * 50)
    
    # 샘플 문서
    sample_text = "인공지능은 미래 기술의 핵심입니다." * 100
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_path = f.name
    
    try:
        summarizer = DocumentSummarizer()
        result = summarizer.summarize(temp_path)
        
        # JSON으로 내보내기
        export_data = {
            "file": os.path.basename(temp_path),
            "summary": result.executive_summary,
            "key_points": result.key_points,
            "statistics": result.statistics,
            "recommendations": result.recommendations,
            "timestamp": str(datetime.now())
        }
        
        json_output = json.dumps(export_data, ensure_ascii=False, indent=2)
        print("JSON 형식:")
        print(json_output[:300] + "...")
        
        # 마크다운으로 내보내기
        markdown_output = f"""# 문서 요약 보고서

## 파일 정보
- 파일명: {os.path.basename(temp_path)}
- 처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 전체 요약
{result.executive_summary}

## 핵심 포인트
{chr(10).join(f'- {point}' for point in result.key_points)}

## 통계
- 총 단어 수: {result.statistics['total_words']:,}
- 섹션 수: {result.statistics['total_sections']}

## 추천사항
{chr(10).join(f'{i+1}. {rec}' for i, rec in enumerate(result.recommendations or []))}
"""
        
        print("\n마크다운 형식:")
        print(markdown_output[:300] + "...")
    
    finally:
        os.unlink(temp_path)


def main():
    """문서 요약기 예제 메인 함수"""
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        return
    
    print("📄 문서 요약기 예제")
    print("=" * 50)
    
    try:
        # 1. 텍스트 파일 요약
        example_text_summary()
        
        # 2. 배치 처리
        example_batch_processing()
        
        # 3. 결과 내보내기
        example_export_results()
        
        print("\n✅ 모든 문서 요약 예제 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()