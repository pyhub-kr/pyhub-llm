#!/usr/bin/env python3
"""
예제: Streamlit과 pyhub-llm 통합
난이도: 고급
설명: Streamlit을 사용한 AI 챗봇 웹 애플리케이션
요구사항: 
  - pyhub-llm (pip install pyhub-llm)
  - streamlit (pip install streamlit)
  - OPENAI_API_KEY 환경 변수

실행 방법:
  streamlit run app.py
"""

import os
import streamlit as st
from datetime import datetime
import json
import time
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
from pyhub.llm import LLM
from pyhub.llm.types import Message


# 페이지 설정
st.set_page_config(
    page_title="AI 챗봇 - pyhub-llm",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/pyhub-kr/pyhub-llm',
        'Report a bug': 'https://github.com/pyhub-kr/pyhub-llm/issues',
        'About': "pyhub-llm을 사용한 AI 챗봇 예제"
    }
)

# CSS 스타일
st.markdown("""
<style>
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'conversation_stats' not in st.session_state:
    st.session_state.conversation_stats = []


def create_new_conversation():
    """새 대화 생성"""
    import uuid
    conv_id = str(uuid.uuid4())
    timestamp = datetime.now()
    
    st.session_state.conversations[conv_id] = {
        'id': conv_id,
        'title': f"대화 {timestamp.strftime('%Y-%m-%d %H:%M')}",
        'messages': [],
        'created_at': timestamp,
        'updated_at': timestamp
    }
    st.session_state.current_conversation_id = conv_id
    st.session_state.messages = []
    return conv_id


def save_message(role: str, content: str, tokens: int = 0):
    """메시지 저장"""
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'tokens': tokens
    }
    
    st.session_state.messages.append(message)
    
    # 현재 대화에도 저장
    if st.session_state.current_conversation_id:
        conv = st.session_state.conversations[st.session_state.current_conversation_id]
        conv['messages'].append(message)
        conv['updated_at'] = datetime.now()
    
    # 토큰 통계 업데이트
    st.session_state.total_tokens += tokens


def main():
    # 타이틀
    st.title("🤖 AI 챗봇 with pyhub-llm")
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OPENAI_API_KEY 환경 변수를 설정해주세요.")
            st.code("export OPENAI_API_KEY='your-api-key'")
            st.stop()
        else:
            st.success("✅ API 키 설정됨")
        
        # 모델 선택
        model = st.selectbox(
            "모델 선택",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        
        # 파라미터 설정
        with st.expander("고급 설정", expanded=False):
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            max_tokens = st.slider("Max Tokens", 100, 4000, 1000, 100)
            stream_mode = st.checkbox("스트리밍 모드", value=True)
            
            # 시스템 프롬프트
            system_prompt = st.text_area(
                "시스템 프롬프트",
                value="당신은 도움이 되고 친절한 AI 어시스턴트입니다.",
                height=100
            )
        
        # LLM 인스턴스 생성/업데이트
        if st.button("설정 적용", type="primary"):
            try:
                st.session_state.llm = LLM.create(
                    model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt
                )
                st.success("✅ 설정이 적용되었습니다!")
            except Exception as e:
                st.error(f"❌ 오류: {str(e)}")
        
        st.markdown("---")
        
        # 대화 관리
        st.header("💬 대화 관리")
        
        # 새 대화 버튼
        if st.button("🆕 새 대화", type="secondary", use_container_width=True):
            create_new_conversation()
            st.rerun()
        
        # 대화 목록
        if st.session_state.conversations:
            st.subheader("대화 목록")
            for conv_id, conv in sorted(
                st.session_state.conversations.items(), 
                key=lambda x: x[1]['updated_at'], 
                reverse=True
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(
                        f"📝 {conv['title'][:20]}...",
                        key=f"conv_{conv_id}",
                        use_container_width=True
                    ):
                        st.session_state.current_conversation_id = conv_id
                        st.session_state.messages = conv['messages'].copy()
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{conv_id}"):
                        del st.session_state.conversations[conv_id]
                        if st.session_state.current_conversation_id == conv_id:
                            st.session_state.current_conversation_id = None
                            st.session_state.messages = []
                        st.rerun()
        
        # 대화 내보내기
        if st.session_state.messages:
            st.markdown("---")
            st.download_button(
                "💾 대화 내보내기 (JSON)",
                data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # 통계
        st.markdown("---")
        st.header("📊 통계")
        st.metric("총 토큰 사용", f"{st.session_state.total_tokens:,}")
        st.metric("총 대화 수", len(st.session_state.conversations))
        st.metric("현재 메시지 수", len(st.session_state.messages))
    
    # 메인 영역
    # LLM 초기화 확인
    if not st.session_state.llm:
        st.session_state.llm = LLM.create("gpt-4o-mini")
    
    # 현재 대화가 없으면 새로 생성
    if not st.session_state.current_conversation_id:
        create_new_conversation()
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["💬 채팅", "📊 분석", "🖼️ 이미지", "🔧 도구"])
    
    with tab1:
        # 채팅 인터페이스
        st.header("채팅")
        
        # 대화 내역 표시
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message.get("tokens", 0) > 0:
                        st.caption(f"토큰: {message['tokens']}")
        
        # 입력 영역
        if prompt := st.chat_input("메시지를 입력하세요..."):
            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.write(prompt)
            
            # 메시지 저장
            save_message("user", prompt)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                tokens_used = 0
                
                try:
                    if stream_mode:
                        # 스트리밍 모드
                        for chunk in st.session_state.llm.ask(prompt, stream=True):
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                    else:
                        # 일반 모드
                        with st.spinner("생각 중..."):
                            reply = st.session_state.llm.ask(prompt)
                            full_response = reply.text
                            tokens_used = reply.usage.total if reply.usage else 0
                            message_placeholder.markdown(full_response)
                    
                    # 응답 저장
                    save_message("assistant", full_response, tokens_used)
                    
                except Exception as e:
                    st.error(f"❌ 오류 발생: {str(e)}")
    
    with tab2:
        # 텍스트 분석
        st.header("텍스트 분석")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            analysis_text = st.text_area(
                "분석할 텍스트를 입력하세요",
                height=200,
                placeholder="텍스트를 입력하면 감정 분석, 요약, 키워드 추출 등을 수행합니다."
            )
        
        with col2:
            analysis_options = st.multiselect(
                "분석 옵션",
                ["감정 분석", "요약", "키워드 추출", "개체명 인식"],
                default=["감정 분석", "요약"]
            )
            
            if st.button("🔍 분석 시작", type="primary", use_container_width=True):
                if analysis_text:
                    with st.spinner("분석 중..."):
                        results = {}
                        
                        if "감정 분석" in analysis_options:
                            reply = st.session_state.llm.ask(
                                f"다음 텍스트의 감정을 분석하세요: {analysis_text}",
                                choices=["긍정", "부정", "중립"]
                            )
                            results["감정"] = reply.choice
                        
                        if "요약" in analysis_options:
                            reply = st.session_state.llm.ask(
                                f"다음 텍스트를 한 문장으로 요약하세요: {analysis_text}"
                            )
                            results["요약"] = reply.text
                        
                        if "키워드 추출" in analysis_options:
                            reply = st.session_state.llm.ask(
                                f"다음 텍스트의 핵심 키워드 5개를 추출하세요: {analysis_text}"
                            )
                            results["키워드"] = reply.text
                        
                        if "개체명 인식" in analysis_options:
                            reply = st.session_state.llm.ask(
                                f"다음 텍스트에서 인물, 장소, 조직 등의 개체명을 추출하세요: {analysis_text}"
                            )
                            results["개체명"] = reply.text
                        
                        # 결과 표시
                        st.success("✅ 분석 완료!")
                        for key, value in results.items():
                            st.subheader(key)
                            st.write(value)
                else:
                    st.warning("분석할 텍스트를 입력해주세요.")
    
    with tab3:
        # 이미지 분석
        st.header("이미지 분석")
        
        uploaded_file = st.file_uploader(
            "이미지를 업로드하세요",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            help="지원 형식: PNG, JPG, JPEG, GIF, WebP"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(uploaded_file, caption="업로드된 이미지", use_column_width=True)
            
            with col2:
                image_question = st.text_area(
                    "질문을 입력하세요",
                    value="이 이미지를 자세히 설명해주세요.",
                    height=100
                )
                
                if st.button("🖼️ 이미지 분석", type="primary", use_container_width=True):
                    with st.spinner("이미지 분석 중..."):
                        try:
                            # 임시 파일로 저장
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                                tmp.write(uploaded_file.getbuffer())
                                tmp_path = tmp.name
                            
                            # 이미지 분석
                            reply = st.session_state.llm.ask(image_question, files=[tmp_path])
                            
                            st.success("✅ 분석 완료!")
                            st.write(reply.text)
                            
                            # 임시 파일 삭제
                            os.unlink(tmp_path)
                            
                        except Exception as e:
                            st.error(f"❌ 오류 발생: {str(e)}")
    
    with tab4:
        # 도구 및 유틸리티
        st.header("도구 및 유틸리티")
        
        tool_option = st.selectbox(
            "도구 선택",
            ["번역기", "코드 생성", "SQL 쿼리 생성", "정규식 생성"]
        )
        
        if tool_option == "번역기":
            col1, col2 = st.columns(2)
            
            with col1:
                source_text = st.text_area("원문", height=200)
                source_lang = st.selectbox("원문 언어", ["자동 감지", "한국어", "영어", "일본어", "중국어"])
            
            with col2:
                target_lang = st.selectbox("번역 언어", ["영어", "한국어", "일본어", "중국어", "스페인어", "프랑스어"])
                
                if st.button("🌐 번역", use_container_width=True):
                    if source_text:
                        with st.spinner("번역 중..."):
                            prompt = f"다음 텍스트를 {target_lang}로 번역하세요:\n{source_text}"
                            reply = st.session_state.llm.ask(prompt)
                            st.text_area("번역 결과", value=reply.text, height=200)
        
        elif tool_option == "코드 생성":
            col1, col2 = st.columns([2, 1])
            
            with col1:
                code_description = st.text_area(
                    "코드 설명",
                    placeholder="생성하고 싶은 코드를 설명하세요...",
                    height=150
                )
            
            with col2:
                language = st.selectbox("프로그래밍 언어", ["Python", "JavaScript", "Java", "C++", "SQL"])
                
                if st.button("💻 코드 생성", use_container_width=True):
                    if code_description:
                        with st.spinner("코드 생성 중..."):
                            prompt = f"{language}로 다음을 구현하는 코드를 작성하세요: {code_description}"
                            reply = st.session_state.llm.ask(prompt)
                            st.code(reply.text, language=language.lower())
        
        elif tool_option == "SQL 쿼리 생성":
            schema = st.text_area(
                "테이블 스키마",
                value="users (id, name, email, created_at)\norders (id, user_id, product_id, amount, order_date)\nproducts (id, name, price, category)",
                height=100
            )
            
            query_description = st.text_input(
                "쿼리 설명",
                placeholder="예: 2024년에 가장 많이 주문한 사용자 top 10"
            )
            
            if st.button("🗄️ SQL 생성"):
                if query_description:
                    with st.spinner("SQL 생성 중..."):
                        prompt = f"""
테이블 스키마:
{schema}

다음을 수행하는 SQL 쿼리를 작성하세요: {query_description}
"""
                        reply = st.session_state.llm.ask(prompt)
                        st.code(reply.text, language="sql")
        
        elif tool_option == "정규식 생성":
            regex_description = st.text_input(
                "정규식 설명",
                placeholder="예: 한국 전화번호 형식 (010-1234-5678)"
            )
            
            test_strings = st.text_area(
                "테스트 문자열 (줄 단위)",
                placeholder="테스트할 문자열을 입력하세요...",
                height=100
            )
            
            if st.button("🔤 정규식 생성"):
                if regex_description:
                    with st.spinner("정규식 생성 중..."):
                        prompt = f"다음을 매칭하는 정규식을 만들어주세요: {regex_description}"
                        reply = st.session_state.llm.ask(prompt)
                        
                        st.code(reply.text)
                        
                        # 테스트 문자열이 있으면 매칭 테스트
                        if test_strings:
                            import re
                            try:
                                # 정규식 추출 (간단한 방법)
                                pattern = reply.text.strip().strip('`').strip()
                                regex = re.compile(pattern)
                                
                                st.subheader("매칭 결과")
                                for line in test_strings.split('\n'):
                                    if line.strip():
                                        match = regex.search(line)
                                        if match:
                                            st.success(f"✅ 매칭: {line}")
                                        else:
                                            st.error(f"❌ 미매칭: {line}")
                            except Exception as e:
                                st.warning(f"정규식 테스트 실패: {str(e)}")
    
    # 푸터
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🤖 Powered by pyhub-llm")
    with col2:
        st.caption(f"📊 총 토큰 사용: {st.session_state.total_tokens:,}")
    with col3:
        st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()