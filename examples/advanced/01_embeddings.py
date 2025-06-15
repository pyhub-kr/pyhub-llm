#!/usr/bin/env python3
"""
예제: 임베딩
난이도: 고급
설명: 텍스트 임베딩 생성, 유사도 계산, 문서 검색 시스템 구현
요구사항:
  - pyhub-llm (pip install "pyhub-llm[all]")
  - scikit-learn (pip install scikit-learn)
  - numpy (pip install numpy)
  - OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import os
import sys
from typing import Any, Dict, List, Tuple

from pyhub.llm.types import Embed, EmbedList

try:
    import numpy as np
except ImportError:
    print("⚠️  scikit-learn이 필요합니다: pip install scikit-learn")
    sys.exit(1)

from pyhub.llm import LLM


def example_basic_embedding():
    """기본 임베딩 생성 예제"""
    print("\n🔢 기본 임베딩 생성")
    print("-" * 50)

    # 임베딩 모델 사용
    llm = LLM.create("text-embedding-3-small")

    # 단일 텍스트 임베딩
    text = "인공지능은 인간의 지능을 모방한 기술입니다."
    print(f"텍스트: {text}")

    embed: Embed = llm.embed(text)
    print(f"임베딩 차원: {len(embed.array)}")
    print(f"임베딩 벡터 (처음 5개): {embed.array[:5]}")

    # 여러 텍스트 임베딩
    texts = [
        "파이썬은 프로그래밍 언어입니다.",
        "Python is a programming language.",
        "자바스크립트는 웹 개발에 사용됩니다.",
        "기계학습은 데이터에서 패턴을 찾습니다.",
    ]

    print("\n여러 텍스트 임베딩 생성:")
    embed_list: EmbedList = llm.embed(texts)
    print(f"생성된 임베딩 수: {len(embed_list.arrays)}")

    for i, text in enumerate(texts):
        print(f"  {i+1}. {text[:30]}... → 벡터 크기: {len(embed_list.arrays[i])}")


def example_similarity_calculation():
    """유사도 계산 예제"""
    print("\n📊 유사도 계산")
    print("-" * 50)

    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("⚠️  scikit-learn이 필요합니다: pip install scikit-learn")
        return

    llm = LLM.create("text-embedding-3-small")

    # 유사도를 계산할 텍스트 쌍
    text_pairs = [
        ("고양이는 귀여운 동물입니다.", "강아지는 충실한 반려동물입니다."),
        ("파이썬으로 웹 개발하기", "Python web development"),
        ("오늘 날씨가 좋네요", "내일 비가 온대요"),
        ("기계학습과 딥러닝", "머신러닝과 심층학습"),
        ("사과는 빨간색입니다", "프로그래밍은 재미있습니다"),
    ]

    print("텍스트 쌍별 유사도:\n")

    for text1, text2 in text_pairs:
        # 임베딩 생성
        embed_list: EmbedList = llm.embed([text1, text2])
        vec1 = np.array(embed_list.arrays[0]).reshape(1, -1)
        vec2 = np.array(embed_list.arrays[1]).reshape(1, -1)

        # 코사인 유사도 계산
        similarity = cosine_similarity(vec1, vec2)[0][0]

        print(f"텍스트 1: {text1}")
        print(f"텍스트 2: {text2}")
        print(f"유사도: {similarity:.3f} {'🟢' if similarity > 0.8 else '🟡' if similarity > 0.6 else '🔴'}")
        print("-" * 30)


def example_document_search():
    """문서 검색 시스템 예제"""
    print("\n🔍 문서 검색 시스템")
    print("-" * 50)

    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("⚠️  scikit-learn이 필요합니다: pip install scikit-learn")
        return

    class DocumentSearch:
        def __init__(self, model="text-embedding-3-small"):
            self.llm = LLM.create(model)
            self.documents = []
            self.embeddings = []
            self.metadata = []

        def add_documents(self, docs: List[Dict[str, Any]]):
            """문서 추가 및 임베딩 생성"""
            texts = [doc["content"] for doc in docs]
            new_embeddings = self.llm.embed(texts).arrays

            self.documents.extend(docs)
            self.embeddings.extend(new_embeddings)

            print(f"✅ {len(docs)}개 문서 추가됨 (총 {len(self.documents)}개)")

        def search(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict]]:
            """쿼리와 가장 유사한 문서 검색"""
            # 쿼리 임베딩
            query_embedding = self.llm.embed(query).array

            # 모든 문서와의 유사도 계산
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                similarities.append((sim, i))

            # 상위 k개 선택
            similarities.sort(reverse=True)
            results = []
            for sim, idx in similarities[:top_k]:
                results.append((sim, self.documents[idx]))

            return results

        def search_with_filter(self, query: str, filter_fn, top_k: int = 3):
            """필터 조건과 함께 검색"""
            # 쿼리 임베딩
            query_embedding = self.llm.embed(query).array

            # 필터링된 문서와의 유사도 계산
            similarities = []
            for i, (doc, doc_embedding) in enumerate(zip(self.documents, self.embeddings)):
                if filter_fn(doc):
                    sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                    similarities.append((sim, i))

            # 상위 k개 선택
            similarities.sort(reverse=True)
            results = []
            for sim, idx in similarities[:top_k]:
                results.append((sim, self.documents[idx]))

            return results

    # 문서 검색 시스템 생성
    search_engine = DocumentSearch()

    # 샘플 문서 추가
    documents = [
        {
            "id": 1,
            "title": "파이썬 기초",
            "content": "파이썬은 배우기 쉬운 프로그래밍 언어입니다. 문법이 간단하고 읽기 쉽습니다.",
            "category": "프로그래밍",
            "level": "초급",
        },
        {
            "id": 2,
            "title": "머신러닝 입문",
            "content": "머신러닝은 데이터를 학습하여 패턴을 찾는 기술입니다. 인공지능의 한 분야입니다.",
            "category": "AI",
            "level": "중급",
        },
        {
            "id": 3,
            "title": "웹 개발 가이드",
            "content": "웹 개발은 HTML, CSS, JavaScript를 사용합니다. 프론트엔드와 백엔드로 나뉩니다.",
            "category": "웹",
            "level": "중급",
        },
        {
            "id": 4,
            "title": "데이터 분석",
            "content": "데이터 분석은 정보를 추출하는 과정입니다. 파이썬의 pandas와 numpy를 활용합니다.",
            "category": "데이터",
            "level": "중급",
        },
        {
            "id": 5,
            "title": "딥러닝 심화",
            "content": "딥러닝은 신경망을 깊게 쌓은 머신러닝입니다. GPU를 활용하여 대규모 데이터를 학습합니다.",
            "category": "AI",
            "level": "고급",
        },
    ]

    search_engine.add_documents(documents)

    # 검색 테스트
    queries = ["파이썬으로 데이터 분석하는 방법", "인공지능과 머신러닝", "초보자를 위한 프로그래밍", "웹사이트 만들기"]

    print("\n📋 검색 결과:")
    for query in queries:
        print(f"\n🔍 검색어: '{query}'")
        results = search_engine.search(query, top_k=2)

        for rank, (score, doc) in enumerate(results, 1):
            print(f"  {rank}. [{score:.3f}] {doc['title']} ({doc['category']}/{doc['level']})")
            print(f"     {doc['content'][:50]}...")

    # 필터링 검색
    print("\n\n📋 필터링 검색 (AI 카테고리만):")
    query = "기계학습 기초"
    results = search_engine.search_with_filter(query, filter_fn=lambda doc: doc["category"] == "AI", top_k=2)

    print(f"🔍 검색어: '{query}' (AI 카테고리)")
    for rank, (score, doc) in enumerate(results, 1):
        print(f"  {rank}. [{score:.3f}] {doc['title']}")


def example_clustering():
    """문서 클러스터링 예제"""
    print("\n🗂️ 문서 클러스터링")
    print("-" * 50)

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("⚠️  scikit-learn이 필요합니다: pip install scikit-learn")
        return

    llm = LLM.create("text-embedding-3-small")

    # 클러스터링할 문서들
    documents = [
        # 프로그래밍 관련
        "파이썬은 인기 있는 프로그래밍 언어입니다.",
        "자바스크립트로 웹 애플리케이션을 개발합니다.",
        "C++은 시스템 프로그래밍에 사용됩니다.",
        # 음식 관련
        "김치는 한국의 전통 발효 음식입니다.",
        "피자는 이탈리아에서 유래한 음식입니다.",
        "초밥은 일본의 대표적인 요리입니다.",
        # 스포츠 관련
        "축구는 전 세계적으로 인기 있는 스포츠입니다.",
        "농구는 미국에서 시작된 스포츠입니다.",
        "테니스는 라켓을 사용하는 스포츠입니다.",
    ]

    # 임베딩 생성
    print("임베딩 생성 중...")
    embeddings = llm.embed(documents).arrays
    embeddings_array = np.array(embeddings)

    # K-means 클러스터링
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_array)

    # 결과 출력
    print(f"\n{n_clusters}개 클러스터로 분류 결과:\n")
    for i in range(n_clusters):
        print(f"📁 클러스터 {i+1}:")
        cluster_docs = [doc for j, doc in enumerate(documents) if clusters[j] == i]
        for doc in cluster_docs:
            print(f"  - {doc}")
        print()

    # 클러스터 중심과의 거리
    print("클러스터 중심과의 거리:")
    distances = kmeans.transform(embeddings_array)
    for i, doc in enumerate(documents):
        cluster = clusters[i]
        distance = distances[i][cluster]
        print(f"  '{doc[:30]}...' → 클러스터 {cluster+1} (거리: {distance:.3f})")


def example_semantic_search_rag():
    """의미 기반 검색 + RAG 예제"""
    print("\n🤖 의미 기반 검색 + RAG (Retrieval Augmented Generation)")
    print("-" * 50)

    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("⚠️  scikit-learn이 필요합니다: pip install scikit-learn")
        return

    class RAGSystem:
        def __init__(self, embedding_model="text-embedding-3-small", generation_model="gpt-4o-mini"):
            self.embed_llm = LLM.create(embedding_model)
            self.gen_llm = LLM.create(generation_model)
            self.knowledge_base = []
            self.embeddings = []

        def add_knowledge(self, documents: List[Dict[str, str]]):
            """지식 베이스에 문서 추가"""
            texts = [doc["content"] for doc in documents]
            new_embeddings = self.embed_llm.embed(texts).arrays

            self.knowledge_base.extend(documents)
            self.embeddings.extend(new_embeddings)

            print(f"✅ {len(documents)}개 문서가 지식 베이스에 추가됨")

        def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
            """관련 문서 검색"""
            query_embedding = self.embed_llm.embed(query).array

            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                similarities.append((sim, i))

            similarities.sort(reverse=True)

            relevant_docs = []
            for sim, idx in similarities[:top_k]:
                doc = self.knowledge_base[idx].copy()
                doc["relevance_score"] = sim
                relevant_docs.append(doc)

            return relevant_docs

        def generate_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
            """검색된 문서를 기반으로 답변 생성"""
            # 1. 관련 문서 검색
            relevant_docs = self.retrieve(query, top_k)

            # 2. 컨텍스트 구성
            context = "\n\n".join(
                [f"[문서 {i+1}] {doc['title']}\n{doc['content']}" for i, doc in enumerate(relevant_docs)]
            )

            # 3. 프롬프트 구성
            prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

참고 문서:
{context}

질문: {query}

답변 시 참고한 문서 번호를 명시해주세요.
"""

            # 4. 답변 생성
            reply = self.gen_llm.ask(prompt)

            return {"query": query, "answer": reply.text, "sources": relevant_docs, "context_used": context}

    # RAG 시스템 생성
    rag = RAGSystem()

    # 지식 베이스 구축
    knowledge_documents = [
        {
            "title": "파이썬 소개",
            "content": "파이썬은 1991년 귀도 반 로섬이 개발한 고급 프로그래밍 언어입니다. 간결하고 읽기 쉬운 문법이 특징입니다.",
        },
        {
            "title": "파이썬의 장점",
            "content": "파이썬은 배우기 쉽고, 다양한 라이브러리를 제공하며, 데이터 분석과 인공지능 분야에서 널리 사용됩니다.",
        },
        {
            "title": "머신러닝 개요",
            "content": "머신러닝은 명시적으로 프로그래밍하지 않고도 컴퓨터가 학습할 수 있도록 하는 인공지능의 한 분야입니다.",
        },
        {
            "title": "딥러닝과 신경망",
            "content": "딥러닝은 인공신경망을 깊게 쌓아 복잡한 패턴을 학습하는 머신러닝 기법입니다. 이미지 인식, 자연어 처리 등에 활용됩니다.",
        },
        {
            "title": "웹 개발 기초",
            "content": "웹 개발은 HTML로 구조를, CSS로 스타일을, JavaScript로 동작을 구현합니다. 백엔드는 서버 측 로직을 담당합니다.",
        },
    ]

    rag.add_knowledge(knowledge_documents)

    # 질문 및 답변 생성
    questions = [
        "파이썬은 언제 누가 만들었나요?",
        "머신러닝과 딥러닝의 차이점은 무엇인가요?",
        "파이썬이 인공지능 분야에서 인기 있는 이유는?",
        "웹 개발을 시작하려면 무엇을 배워야 하나요?",
    ]

    print("\n💬 RAG 기반 질의응답:\n")
    for question in questions:
        result = rag.generate_answer(question, top_k=2)

        print(f"❓ 질문: {question}")
        print(f"💡 답변: {result['answer']}")
        print("📚 참고 문서:")
        for doc in result["sources"]:
            print(f"   - [{doc['relevance_score']:.3f}] {doc['title']}")
        print("-" * 50)


def main():
    """임베딩 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("🔢 임베딩 예제")
    print("=" * 50)

    try:
        # 1. 기본 임베딩 생성
        example_basic_embedding()

        # 2. 유사도 계산
        example_similarity_calculation()

        # 3. 문서 검색 시스템
        example_document_search()

        # 4. 문서 클러스터링
        example_clustering()

        # 5. RAG 시스템
        example_semantic_search_rag()

        print("\n✅ 모든 임베딩 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
