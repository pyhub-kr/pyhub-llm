#!/usr/bin/env python3
"""
예제: 고급 번역기
난이도: 고급
설명: 컨텍스트 인식 번역 및 다국어 지원 시스템
요구사항:
  - pyhub-llm (pip install "pyhub-llm[all]")
  - OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pyhub.llm import LLM

# from pyhub.llm.cache import FileCache  # 캐시 기능은 현재 예제에서 사용하지 않음
# from pyhub.llm.templates.engine import TemplateEngine  # 템플릿 엔진 대신 직접 문자열 포맷팅 사용


class TranslationType(Enum):
    """번역 타입"""

    GENERAL = "general"  # 일반 번역
    TECHNICAL = "technical"  # 기술 문서
    BUSINESS = "business"  # 비즈니스 문서
    CREATIVE = "creative"  # 창의적 번역
    LEGAL = "legal"  # 법률 문서
    MEDICAL = "medical"  # 의료 문서


@dataclass
class TranslationContext:
    """번역 컨텍스트"""

    domain: TranslationType
    formality: str  # formal, informal, neutral
    target_audience: str
    glossary: Dict[str, str] = field(default_factory=dict)
    previous_translations: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class TranslationResult:
    """번역 결과"""

    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    alternative_translations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    word_count: Dict[str, int] = field(default_factory=dict)


class AdvancedTranslator:
    """고급 번역기"""

    # 지원 언어
    SUPPORTED_LANGUAGES = {
        "ko": "Korean",
        "en": "English",
        "ja": "Japanese",
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi",
    }

    def __init__(self, model: str = "gpt-4o"):
        self.llm = LLM.create(model)
        # self.cache = FileCache("translations")  # 캐시 기능은 현재 예제에서 사용하지 않음
        # self.template_engine = TemplateEngine()
        self._setup_templates()

    def _setup_templates(self):
        """번역 템플릿 설정"""
        self.templates = {
            "translate": """Translate the following text from {source_lang} to {target_lang}.

Context:
- Domain: {domain}
- Formality: {formality}
- Target audience: {audience}

{glossary_section}

{previous_section}

Text to translate:
{text}

Provide:
1. Main translation
2. Alternative translations (if applicable)
3. Translation notes (cultural/contextual considerations)""",
            "detect_language": """Detect the language of the following text and respond with the ISO 639-1 language code (e.g., 'en', 'ko', 'ja'):

Text: {text}

Language code:""",
        }

    def detect_language(self, text: str) -> str:
        """언어 감지"""
        # 캐시 확인 (캐시 기능은 현재 비활성화)
        # cache_key = f"detect_{hash(text[:100])}"
        # cached = self.cache.get(cache_key)
        # if cached:
        #     return cached

        prompt = self.templates["detect_language"].format(text=text[:500])  # 처음 500자만 사용

        reply = self.llm.ask(prompt)
        language_code = reply.text.strip().lower()

        # 유효성 검사
        if language_code not in self.SUPPORTED_LANGUAGES:
            # 휴리스틱 감지
            if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text):
                language_code = "ko"
            elif any(ord(char) >= 0x4E00 and ord(char) <= 0x9FFF for char in text):
                language_code = "zh"
            elif any(ord(char) >= 0x3040 and ord(char) <= 0x309F for char in text):
                language_code = "ja"
            else:
                language_code = "en"

        # 캐시 저장 (캐시 기능은 현재 비활성화)
        # self.cache.set(cache_key, language_code)

        return language_code

    def create_glossary(self, terms: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """다국어 용어집 생성"""
        # 예: {"API": {"ko": "API", "ja": "API"}, "function": {"ko": "함수", "ja": "関数"}}
        return terms

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        context: Optional[TranslationContext] = None,
    ) -> TranslationResult:
        """텍스트 번역"""
        # 언어 감지
        if not source_language:
            source_language = self.detect_language(text)
            print(f"감지된 언어: {self.SUPPORTED_LANGUAGES.get(source_language, source_language)}")

        # 같은 언어면 그대로 반환
        if source_language == target_language:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                confidence_score=1.0,
                notes=["원문과 대상 언어가 동일합니다."],
            )

        # 기본 컨텍스트
        if not context:
            context = TranslationContext(domain=TranslationType.GENERAL, formality="neutral", target_audience="general")

        # 캐시 키 (캐시 기능은 현재 비활성화)
        # cache_key = f"translate_{source_language}_{target_language}_{context.domain.value}_{hash(text)}"
        # cached = self.cache.get(cache_key)
        # if cached:
        #     return cached

        # 번역 프롬프트 생성
        # Glossary 섹션 생성
        glossary_section = ""
        if context.glossary:
            glossary_lines = ["Glossary (use these translations):"]
            for term, translation in context.glossary.items():
                glossary_lines.append(f"- {term} → {translation}")
            glossary_section = "\n".join(glossary_lines)

        # Previous translations 섹션 생성
        previous_section = ""
        if context.previous_translations:
            previous_lines = ["Previous translations for context:"]
            for orig, trans in context.previous_translations[-3:]:
                previous_lines.append(f'- "{orig}" → "{trans}"')
            previous_section = "\n".join(previous_lines)

        prompt = self.templates["translate"].format(
            text=text,
            source_lang=self.SUPPORTED_LANGUAGES.get(source_language, source_language),
            target_lang=self.SUPPORTED_LANGUAGES.get(target_language, target_language),
            domain=context.domain.value,
            formality=context.formality,
            audience=context.target_audience,
            glossary_section=glossary_section,
            previous_section=previous_section,
        )

        # 번역 수행
        reply = self.llm.ask(prompt)

        # 결과 파싱
        result = self._parse_translation_result(reply.text, text, source_language, target_language)

        # 캐시 저장 (캐시 기능은 현재 비활성화)
        # self.cache.set(cache_key, result)

        return result

    def _parse_translation_result(
        self, response: str, original_text: str, source_lang: str, target_lang: str
    ) -> TranslationResult:
        """번역 결과 파싱"""
        lines = response.strip().split("\n")

        # 메인 번역 추출
        main_translation = ""
        alternatives = []
        notes = []

        section = "main"
        for line in lines:
            line = line.strip()

            if "alternative" in line.lower() or "대체" in line.lower():
                section = "alternatives"
                continue
            elif "note" in line.lower() or "참고" in line.lower():
                section = "notes"
                continue

            if line and not line.startswith(("1.", "2.", "3.", "-", "•")):
                if section == "main" and not main_translation:
                    main_translation = line
            elif line.startswith(("-", "•", "1.", "2.", "3.")):
                content = re.sub(r"^[-•\d.]\s*", "", line)
                if section == "alternatives":
                    alternatives.append(content)
                elif section == "notes":
                    notes.append(content)

        # 단어 수 계산
        word_count = {"source": len(original_text.split()), "target": len(main_translation.split())}

        # 신뢰도 점수 계산 (휴리스틱)
        confidence = 0.9
        if alternatives:
            confidence = 0.8  # 대체 번역이 있으면 약간 낮춤
        if abs(word_count["source"] - word_count["target"]) > word_count["source"] * 0.5:
            confidence -= 0.1  # 단어 수 차이가 크면 낮춤

        return TranslationResult(
            original_text=original_text,
            translated_text=main_translation or response.strip(),
            source_language=source_lang,
            target_language=target_lang,
            confidence_score=confidence,
            alternative_translations=alternatives,
            notes=notes,
            word_count=word_count,
        )

    def translate_batch(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None,
        context: Optional[TranslationContext] = None,
    ) -> List[TranslationResult]:
        """배치 번역"""
        results = []

        # 컨텍스트 초기화
        if context and context.previous_translations is None:
            context.previous_translations = []

        for i, text in enumerate(texts):
            print(f"번역 중 {i+1}/{len(texts)}...")

            result = self.translate(text, target_language, source_language, context)
            results.append(result)

            # 컨텍스트 업데이트
            if context:
                context.previous_translations.append((text[:50], result.translated_text[:50]))

        return results

    def translate_document(self, document: str, target_language: str, preserve_formatting: bool = True) -> str:
        """문서 번역"""
        # 단락 분리
        paragraphs = document.split("\n\n")
        translated_paragraphs = []

        # 문서 타입 추론
        doc_type = self._infer_document_type(document)
        context = TranslationContext(
            domain=doc_type,
            formality="formal" if doc_type in [TranslationType.LEGAL, TranslationType.BUSINESS] else "neutral",
            target_audience="professional",
        )

        # 각 단락 번역
        for para in paragraphs:
            if para.strip():
                # 포맷 보존
                if preserve_formatting:
                    # 들여쓰기 보존
                    indent = len(para) - len(para.lstrip())
                    text = para.strip()

                    # 마크다운 헤더 보존
                    header_match = re.match(r"^(#{1,6})\s+(.+)$", text)
                    if header_match:
                        level = header_match.group(1)
                        content = header_match.group(2)
                        result = self.translate(content, target_language, context=context)
                        translated = f"{level} {result.translated_text}"
                    else:
                        result = self.translate(text, target_language, context=context)
                        translated = result.translated_text

                    # 들여쓰기 복원
                    translated = " " * indent + translated
                else:
                    result = self.translate(para, target_language, context=context)
                    translated = result.translated_text

                translated_paragraphs.append(translated)
            else:
                translated_paragraphs.append("")

        return "\n\n".join(translated_paragraphs)

    def _infer_document_type(self, document: str) -> TranslationType:
        """문서 타입 추론"""
        doc_lower = document.lower()

        # 키워드 기반 추론
        if any(word in doc_lower for word in ["contract", "agreement", "legal", "계약", "법적"]):
            return TranslationType.LEGAL
        elif any(word in doc_lower for word in ["diagnosis", "treatment", "patient", "진단", "치료", "환자"]):
            return TranslationType.MEDICAL
        elif any(word in doc_lower for word in ["revenue", "profit", "business", "수익", "비즈니스"]):
            return TranslationType.BUSINESS
        elif any(word in doc_lower for word in ["function", "class", "api", "code", "함수", "클래스"]):
            return TranslationType.TECHNICAL
        else:
            return TranslationType.GENERAL

    def create_translation_memory(self, translations: List[TranslationResult]) -> Dict[str, Any]:
        """번역 메모리 생성"""
        memory = {"version": "1.0", "created_at": datetime.now().isoformat(), "translations": []}

        for trans in translations:
            memory["translations"].append(
                {
                    "source": trans.original_text,
                    "target": trans.translated_text,
                    "source_lang": trans.source_language,
                    "target_lang": trans.target_language,
                    "confidence": trans.confidence_score,
                    "domain": "general",  # 실제로는 컨텍스트에서 가져와야 함
                }
            )

        return memory

    def quality_check(self, translation_result: TranslationResult) -> Dict[str, Any]:
        """번역 품질 검사"""
        issues = []
        score = 100

        # 길이 비교
        len_ratio = translation_result.word_count["target"] / translation_result.word_count["source"]
        if len_ratio < 0.5 or len_ratio > 2.0:
            issues.append("번역문의 길이가 원문과 크게 다릅니다.")
            score -= 10

        # 번역되지 않은 부분 확인
        if translation_result.source_language == "en" and translation_result.target_language == "ko":
            # 영어 단어가 그대로 남아있는지 확인
            english_pattern = re.compile(r"[a-zA-Z]{4,}")
            english_words = english_pattern.findall(translation_result.translated_text)
            if len(english_words) > translation_result.word_count["target"] * 0.3:
                issues.append("번역되지 않은 영어 단어가 많습니다.")
                score -= 15

        # 특수 문자 보존 확인
        source_special = set(re.findall(r"[^\w\s]", translation_result.original_text))
        target_special = set(re.findall(r"[^\w\s]", translation_result.translated_text))
        if len(source_special - target_special) > 3:
            issues.append("일부 특수 문자가 누락되었을 수 있습니다.")
            score -= 5

        return {"score": max(0, score), "issues": issues, "passed": score >= 70}


def example_basic_translation():
    """기본 번역 예제"""
    print("\n🌐 기본 번역")
    print("-" * 50)

    translator = AdvancedTranslator()

    # 다양한 언어 번역
    texts = [
        ("Hello, world!", "ko"),
        ("안녕하세요, 반갑습니다.", "en"),
        ("人工知能の未来", "ko"),
        ("Bonjour le monde", "en"),
    ]

    for text, target_lang in texts:
        result = translator.translate(text, target_lang)
        print(f"\n원문: {result.original_text}")
        print(f"번역: {result.translated_text}")
        print(f"언어: {result.source_language} → {result.target_language}")
        print(f"신뢰도: {result.confidence_score:.1%}")

        if result.alternative_translations:
            print(f"대체 번역: {', '.join(result.alternative_translations)}")


def example_contextual_translation():
    """컨텍스트 기반 번역 예제"""
    print("\n📝 컨텍스트 번역")
    print("-" * 50)

    translator = AdvancedTranslator()

    # 기술 문서 컨텍스트
    tech_context = TranslationContext(
        domain=TranslationType.TECHNICAL,
        formality="neutral",
        target_audience="developers",
        glossary={"function": "함수", "class": "클래스", "inheritance": "상속"},
    )

    technical_text = """
The function processes input data and returns a list.
It uses inheritance to extend the base class functionality.
"""

    result = translator.translate(technical_text, "ko", context=tech_context)
    print("기술 문서 번역:")
    print(f"원문: {result.original_text}")
    print(f"번역: {result.translated_text}")

    # 비즈니스 문서 컨텍스트
    business_context = TranslationContext(
        domain=TranslationType.BUSINESS,
        formality="formal",
        target_audience="executives",
        glossary={"revenue": "수익", "profit margin": "이익률", "stakeholder": "이해관계자"},
    )

    business_text = "Our revenue increased by 20% this quarter."

    result = translator.translate(business_text, "ko", context=business_context)
    print("\n비즈니스 문서 번역:")
    print(f"원문: {result.original_text}")
    print(f"번역: {result.translated_text}")


def example_document_translation():
    """문서 번역 예제"""
    print("\n📄 문서 번역")
    print("-" * 50)

    translator = AdvancedTranslator()

    document = """
# AI Translation Guide

## Introduction
This document explains how to use the advanced translation system.

## Features
- Multi-language support
- Context-aware translation
- Batch processing

## Usage
1. Initialize the translator
2. Set the context
3. Translate your text

## Conclusion
The system provides high-quality translations for various domains.
"""

    translated_doc = translator.translate_document(document, "ko")
    print("번역된 문서:")
    print(translated_doc)


def example_batch_translation():
    """배치 번역 예제"""
    print("\n📚 배치 번역")
    print("-" * 50)

    translator = AdvancedTranslator()

    # 연관된 문장들
    sentences = [
        "The new AI model shows promising results.",
        "It achieves 95% accuracy on the test dataset.",
        "Further improvements are being developed.",
        "The model will be released next month.",
    ]

    # 컨텍스트 유지하며 번역
    context = TranslationContext(domain=TranslationType.TECHNICAL, formality="neutral", target_audience="researchers")

    results = translator.translate_batch(sentences, "ko", context=context)

    print("배치 번역 결과:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.original_text}")
        print(f"   → {result.translated_text}")


def example_quality_check():
    """번역 품질 검사 예제"""
    print("\n✅ 번역 품질 검사")
    print("-" * 50)

    translator = AdvancedTranslator()

    # 번역 수행
    text = "Artificial intelligence is transforming the way we live and work."
    result = translator.translate(text, "ko")

    print(f"원문: {result.original_text}")
    print(f"번역: {result.translated_text}")

    # 품질 검사
    quality = translator.quality_check(result)
    print(f"\n품질 점수: {quality['score']}/100")
    print(f"통과 여부: {'✅ 통과' if quality['passed'] else '❌ 실패'}")

    if quality["issues"]:
        print("\n발견된 이슈:")
        for issue in quality["issues"]:
            print(f"  - {issue}")


def example_translation_memory():
    """번역 메모리 예제"""
    print("\n💾 번역 메모리")
    print("-" * 50)

    translator = AdvancedTranslator()

    # 여러 번역 수행
    texts = ["Welcome to our service", "Please login to continue", "Thank you for using our product"]

    results = []
    for text in texts:
        result = translator.translate(text, "ko")
        results.append(result)
        print(f"'{text}' → '{result.translated_text}'")

    # 번역 메모리 생성
    memory = translator.create_translation_memory(results)

    print("\n번역 메모리 생성됨:")
    print(f"  - 버전: {memory['version']}")
    print(f"  - 번역 수: {len(memory['translations'])}")
    print(f"  - 생성 시간: {memory['created_at']}")


def main():
    """고급 번역기 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("🌐 고급 번역기 예제")
    print("=" * 50)

    try:
        # 1. 기본 번역
        example_basic_translation()

        # 2. 컨텍스트 번역
        example_contextual_translation()

        # 3. 문서 번역
        example_document_translation()

        # 4. 배치 번역
        example_batch_translation()

        # 5. 품질 검사
        example_quality_check()

        # 6. 번역 메모리
        example_translation_memory()

        print("\n✅ 모든 번역기 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
