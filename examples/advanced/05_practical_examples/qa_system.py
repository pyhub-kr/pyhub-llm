#!/usr/bin/env python3
"""
예제: 질문 답변 시스템 (Q&A System)
난이도: 고급
설명: 문서 기반 질문 답변 및 지식 관리 시스템
요구사항:
  - pyhub-llm (pip install "pyhub-llm[all]")
  - numpy (pip install numpy)
  - OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyhub.llm import LLM
from pyhub.llm.cache import FileCache


@dataclass
class Document:
    """문서"""

    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List["DocumentChunk"] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DocumentChunk:
    """문서 청크"""

    id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    """답변"""

    question: str
    answer: str
    confidence: float
    sources: List[DocumentChunk]
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeBase:
    """지식 베이스"""

    def __init__(self, storage_path: str = "knowledge_base"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.documents: Dict[str, Document] = {}
        self.chunks: List[DocumentChunk] = []
        self.embeddings_cache = FileCache(str(self.storage_path / "embeddings"))
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """지식 베이스 로드"""
        kb_file = self.storage_path / "knowledge_base.json"
        if kb_file.exists():
            with open(kb_file, "r", encoding="utf-8") as f:
                data = json.load(f)

                # 문서 복원
                for doc_data in data.get("documents", []):
                    doc = Document(
                        id=doc_data["id"],
                        title=doc_data["title"],
                        content=doc_data["content"],
                        metadata=doc_data.get("metadata", {}),
                        created_at=datetime.fromisoformat(doc_data["created_at"]),
                    )
                    self.documents[doc.id] = doc

                # 청크 복원
                for chunk_data in data.get("chunks", []):
                    chunk = DocumentChunk(
                        id=chunk_data["id"],
                        document_id=chunk_data["document_id"],
                        content=chunk_data["content"],
                        embedding=chunk_data.get("embedding"),
                        position=chunk_data.get("position", 0),
                        metadata=chunk_data.get("metadata", {}),
                    )
                    self.chunks.append(chunk)

    def save(self):
        """지식 베이스 저장"""
        data = {
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat(),
                }
                for doc in self.documents.values()
            ],
            "chunks": [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "position": chunk.position,
                    "metadata": chunk.metadata,
                }
                for chunk in self.chunks
            ],
        }

        kb_file = self.storage_path / "knowledge_base.json"
        with open(kb_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_document(self, title: str, content: str, metadata: Optional[Dict] = None) -> Document:
        """문서 추가"""
        # 문서 ID 생성
        doc_id = hashlib.md5(f"{title}{content}".encode()).hexdigest()[:8]

        # 문서 생성
        doc = Document(id=doc_id, title=title, content=content, metadata=metadata or {})

        # 청크 생성
        chunks = self._create_chunks(doc)
        doc.chunks = chunks
        self.chunks.extend(chunks)

        # 저장
        self.documents[doc_id] = doc
        self.save()

        return doc

    def _create_chunks(self, document: Document, chunk_size: int = 500) -> List[DocumentChunk]:
        """문서를 청크로 분할"""
        chunks = []

        # 단락 기반 분할
        paragraphs = document.content.split("\n\n")
        current_chunk = []
        current_size = 0
        position = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > chunk_size and current_chunk:
                # 현재 청크 저장
                chunk_content = "\n\n".join(current_chunk)
                chunk_id = f"{document.id}_{position}"

                chunk = DocumentChunk(id=chunk_id, document_id=document.id, content=chunk_content, position=position)
                chunks.append(chunk)

                # 새 청크 시작
                current_chunk = [para]
                current_size = para_size
                position += 1
            else:
                current_chunk.append(para)
                current_size += para_size

        # 마지막 청크
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunk_id = f"{document.id}_{position}"

            chunk = DocumentChunk(id=chunk_id, document_id=document.id, content=chunk_content, position=position)
            chunks.append(chunk)

        return chunks

    def search_chunks(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """청크 검색"""
        # 쿼리 임베딩
        query_embedding = self._get_embedding(query)

        # 청크별 유사도 계산
        results = []
        for chunk in self.chunks:
            if chunk.embedding is None:
                chunk.embedding = self._get_embedding(chunk.content)

            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            results.append((chunk, similarity))

        # 상위 k개 반환
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        # 캐시 확인
        cache_key = f"embed_{hashlib.md5(text.encode()).hexdigest()}"
        cached = self.embeddings_cache.get(cache_key)
        if cached:
            return cached

        # 임베딩 생성
        llm = LLM.create("text-embedding-3-small")
        embeddings = llm.embed([text])
        embedding = embeddings[0]

        # 캐시 저장
        self.embeddings_cache.set(cache_key, embedding)

        return embedding

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        try:
            import numpy as np

            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except ImportError:
            # NumPy 없이 계산
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)


class QASystem:
    """질문 답변 시스템"""

    def __init__(self, model: str = "gpt-4o", knowledge_base: Optional[KnowledgeBase] = None):
        self.llm = LLM.create(model)
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.cache = FileCache("qa_cache")
        self.conversation_history: List[Tuple[str, str]] = []

    def add_knowledge(self, title: str, content: str, metadata: Optional[Dict] = None):
        """지식 추가"""
        doc = self.knowledge_base.add_document(title, content, metadata)
        print(f"✅ 문서 추가됨: {doc.title} (ID: {doc.id})")
        return doc

    def answer_question(
        self,
        question: str,
        use_knowledge_base: bool = True,
        include_sources: bool = True,
        max_context_length: int = 2000,
    ) -> Answer:
        """질문에 답변"""
        print(f"🤔 질문: {question}")

        # 캐시 확인
        cache_key = f"qa_{hashlib.md5(question.encode()).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached and not include_sources:  # 소스가 필요한 경우 캐시 사용 안함
            return Answer(**cached)

        # 관련 문서 검색
        relevant_chunks = []
        context = ""

        if use_knowledge_base and self.knowledge_base.chunks:
            print("📚 지식 베이스 검색 중...")
            search_results = self.knowledge_base.search_chunks(question, top_k=5)

            for chunk, similarity in search_results:
                if similarity > 0.7:  # 유사도 임계값
                    relevant_chunks.append(chunk)
                    context += f"\n---\n{chunk.content}\n"

                    if len(context) > max_context_length:
                        break

            print(f"  관련 문서 {len(relevant_chunks)}개 발견")

        # 프롬프트 생성
        prompt = self._create_prompt(question, context)

        # LLM에 질문
        reply = self.llm.ask(prompt)
        answer_text = reply.text

        # 신뢰도 계산
        confidence = self._calculate_confidence(question, answer_text, relevant_chunks)

        # 답변 생성
        answer = Answer(
            question=question,
            answer=answer_text,
            confidence=confidence,
            sources=relevant_chunks if include_sources else [],
            metadata={"model": self.llm.model, "timestamp": datetime.now().isoformat(), "context_used": bool(context)},
        )

        # 대화 기록 저장
        self.conversation_history.append((question, answer_text))

        # 캐시 저장 (소스 제외)
        if not include_sources:
            cache_data = {
                "question": answer.question,
                "answer": answer.answer,
                "confidence": answer.confidence,
                "sources": [],
                "metadata": answer.metadata,
            }
            self.cache.set(cache_key, cache_data)

        return answer

    def _create_prompt(self, question: str, context: str) -> str:
        """프롬프트 생성"""
        if context:
            prompt = f"""다음 컨텍스트를 참고하여 질문에 답변하세요.
컨텍스트에 답이 없다면 일반적인 지식으로 답변하되, 그 사실을 명시하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        else:
            # 대화 기록 포함
            history = ""
            if self.conversation_history:
                recent = self.conversation_history[-3:]  # 최근 3개
                history = "\n".join([f"Q: {q}\nA: {a}" for q, a in recent])

                prompt = f"""이전 대화:
{history}

현재 질문: {question}

답변:"""
            else:
                prompt = f"질문: {question}\n\n답변:"

        return prompt

    def _calculate_confidence(self, question: str, answer: str, sources: List[DocumentChunk]) -> float:
        """답변 신뢰도 계산"""
        confidence = 0.5  # 기본 신뢰도

        # 소스가 있으면 신뢰도 증가
        if sources:
            confidence += 0.3 * min(len(sources) / 3, 1.0)

        # 답변 길이에 따른 조정
        if len(answer) > 100:
            confidence += 0.1

        # 불확실한 표현 확인
        uncertain_phrases = ["아마도", "추측", "확실하지 않", "might", "maybe", "possibly"]
        if any(phrase in answer.lower() for phrase in uncertain_phrases):
            confidence -= 0.2

        return max(0.1, min(1.0, confidence))

    def ask_followup(self, question: str) -> Answer:
        """후속 질문"""
        # 이전 대화 컨텍스트 활용
        return self.answer_question(question, use_knowledge_base=True)

    def explain_answer(self, answer: Answer) -> str:
        """답변 설명"""
        prompt = f"""다음 답변이 어떻게 도출되었는지 설명하세요:

질문: {answer.question}
답변: {answer.answer}

사용된 소스 수: {len(answer.sources)}
신뢰도: {answer.confidence:.1%}

설명:"""

        reply = self.llm.ask(prompt)
        return reply.text

    def generate_related_questions(self, question: str, num_questions: int = 3) -> List[str]:
        """관련 질문 생성"""
        prompt = f"""다음 질문과 관련된 후속 질문 {num_questions}개를 생성하세요:

원래 질문: {question}

관련 질문:
1."""

        reply = self.llm.ask(prompt)

        # 질문 파싱
        questions = []
        for line in reply.text.split("\n"):
            line = line.strip()
            if re.match(r"^\d+\.", line):
                question = re.sub(r"^\d+\.\s*", "", line)
                questions.append(question)

        return questions[:num_questions]

    def export_qa_history(self, format: str = "json") -> str:
        """Q&A 기록 내보내기"""
        history = []

        for q, a in self.conversation_history:
            history.append({"question": q, "answer": a, "timestamp": datetime.now().isoformat()})

        if format == "json":
            return json.dumps(history, ensure_ascii=False, indent=2)
        elif format == "markdown":
            lines = ["# Q&A History\n"]
            for i, (q, a) in enumerate(self.conversation_history, 1):
                lines.append(f"## {i}. {q}")
                lines.append(f"\n{a}\n")
            return "\n".join(lines)
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")


def example_basic_qa():
    """기본 Q&A 예제"""
    print("\n💬 기본 Q&A")
    print("-" * 50)

    qa_system = QASystem()

    # 지식 없이 질문
    questions = [
        "Python에서 리스트와 튜플의 차이는 무엇인가요?",
        "기계학습과 딥러닝의 차이점을 설명해주세요.",
        "REST API란 무엇인가요?",
    ]

    for question in questions:
        answer = qa_system.answer_question(question, use_knowledge_base=False)
        print(f"\n❓ Q: {question}")
        print(f"💡 A: {answer.answer[:200]}...")
        print(f"📊 신뢰도: {answer.confidence:.1%}")


def example_knowledge_based_qa():
    """지식 기반 Q&A 예제"""
    print("\n📚 지식 기반 Q&A")
    print("-" * 50)

    qa_system = QASystem()

    # 지식 추가
    qa_system.add_knowledge(
        "Python 가이드",
        """Python은 고수준 프로그래밍 언어입니다.
        
주요 특징:
- 간결하고 읽기 쉬운 문법
- 동적 타이핑
- 자동 메모리 관리
- 풍부한 표준 라이브러리

Python의 철학:
- Beautiful is better than ugly
- Explicit is better than implicit
- Simple is better than complex""",
    )

    qa_system.add_knowledge(
        "웹 개발 기초",
        """웹 개발은 크게 프론트엔드와 백엔드로 나뉩니다.

프론트엔드:
- HTML: 구조
- CSS: 스타일
- JavaScript: 동작

백엔드:
- 서버 측 로직
- 데이터베이스 관리
- API 개발""",
    )

    # 지식 기반 질문
    questions = [
        "Python의 주요 특징은 무엇인가요?",
        "웹 개발에서 프론트엔드란 무엇인가요?",
        "Python의 철학에 대해 알려주세요.",
    ]

    for question in questions:
        answer = qa_system.answer_question(question)
        print(f"\n❓ Q: {question}")
        print(f"💡 A: {answer.answer}")
        print(f"📊 신뢰도: {answer.confidence:.1%}")
        print(f"📚 소스: {len(answer.sources)}개 문서 참조")


def example_conversational_qa():
    """대화형 Q&A 예제"""
    print("\n🗣️ 대화형 Q&A")
    print("-" * 50)

    qa_system = QASystem()

    # 초기 질문
    answer1 = qa_system.answer_question("인공지능이란 무엇인가요?")
    print("Q1: 인공지능이란 무엇인가요?")
    print(f"A1: {answer1.answer[:150]}...")

    # 후속 질문
    answer2 = qa_system.ask_followup("그럼 기계학습은 뭐죠?")
    print("\nQ2: 그럼 기계학습은 뭐죠?")
    print(f"A2: {answer2.answer[:150]}...")

    # 관련 질문 생성
    related = qa_system.generate_related_questions("인공지능이란 무엇인가요?")
    print("\n🔍 관련 질문 제안:")
    for i, q in enumerate(related, 1):
        print(f"  {i}. {q}")


def example_answer_explanation():
    """답변 설명 예제"""
    print("\n🔍 답변 설명")
    print("-" * 50)

    qa_system = QASystem()

    # 지식 추가
    qa_system.add_knowledge(
        "AI 윤리",
        """AI 윤리는 인공지능의 개발과 사용에 있어 중요한 원칙들입니다.

주요 원칙:
1. 투명성: AI의 결정 과정이 이해 가능해야 함
2. 공정성: 편견 없는 AI
3. 책임성: AI의 결정에 대한 책임 소재 명확화
4. 프라이버시: 개인정보 보호""",
    )

    # 질문 및 답변
    answer = qa_system.answer_question("AI 윤리의 주요 원칙은 무엇인가요?")
    print(f"❓ 질문: {answer.question}")
    print(f"💡 답변: {answer.answer}")

    # 답변 설명
    explanation = qa_system.explain_answer(answer)
    print("\n📝 답변 설명:")
    print(explanation)


def example_export_history():
    """기록 내보내기 예제"""
    print("\n💾 Q&A 기록 내보내기")
    print("-" * 50)

    qa_system = QASystem()

    # 여러 질문 수행
    questions = ["클라우드 컴퓨팅이란?", "도커(Docker)의 장점은?", "마이크로서비스 아키텍처란?"]

    for q in questions:
        qa_system.answer_question(q, use_knowledge_base=False)

    # JSON 내보내기
    json_export = qa_system.export_qa_history("json")
    print("JSON 형식:")
    print(json_export[:300] + "...")

    # 마크다운 내보내기
    md_export = qa_system.export_qa_history("markdown")
    print("\n마크다운 형식:")
    print(md_export[:300] + "...")


def example_advanced_search():
    """고급 검색 예제"""
    print("\n🔎 고급 지식 검색")
    print("-" * 50)

    # 지식 베이스 생성
    kb = KnowledgeBase()

    # 다양한 문서 추가
    documents = [
        ("Python 기초", "Python은 배우기 쉬운 프로그래밍 언어입니다..."),
        ("Python 고급", "Python의 고급 기능에는 데코레이터, 제너레이터..."),
        ("JavaScript 기초", "JavaScript는 웹 브라우저에서 실행되는..."),
        ("데이터베이스", "관계형 데이터베이스는 테이블 형태로..."),
    ]

    for title, content in documents:
        kb.add_document(title, content * 10)  # 내용 확장

    # 검색
    query = "Python 프로그래밍"
    results = kb.search_chunks(query, top_k=3)

    print(f"🔍 검색어: {query}")
    print("\n검색 결과:")
    for chunk, similarity in results:
        doc = kb.documents.get(chunk.document_id)
        print(f"\n📄 문서: {doc.title if doc else 'Unknown'}")
        print(f"   유사도: {similarity:.2%}")
        print(f"   내용: {chunk.content[:100]}...")


def main():
    """Q&A 시스템 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        return

    print("❓ 질문 답변 시스템 예제")
    print("=" * 50)

    try:
        # 1. 기본 Q&A
        example_basic_qa()

        # 2. 지식 기반 Q&A
        example_knowledge_based_qa()

        # 3. 대화형 Q&A
        example_conversational_qa()

        # 4. 답변 설명
        example_answer_explanation()

        # 5. 기록 내보내기
        example_export_history()

        # 6. 고급 검색
        example_advanced_search()

        print("\n✅ 모든 Q&A 시스템 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
