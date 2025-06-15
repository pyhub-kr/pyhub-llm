#!/usr/bin/env python3
"""
예제: 템플릿 활용
난이도: 중급
설명: Jinja2 템플릿을 사용한 프롬프트 관리
요구사항: OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import os
from datetime import datetime
from pathlib import Path

from pyhub.llm import LLM
from pyhub.llm.templates.engine import TemplateEngine


def create_sample_templates():
    """샘플 템플릿 파일 생성"""
    templates_dir = Path("./templates")
    templates_dir.mkdir(exist_ok=True)

    # 1. 번역 템플릿
    translation_template = """다음 텍스트를 {{ target_language }}로 번역해주세요.
{% if style %}번역 스타일: {{ style }}{% endif %}
{% if context %}맥락: {{ context }}{% endif %}

원문:
{{ text }}

번역:"""

    (templates_dir / "translation.j2").write_text(translation_template, encoding="utf-8")

    # 2. 코드 리뷰 템플릿
    code_review_template = """다음 {{ language }} 코드를 리뷰해주세요.

리뷰 포인트:
{% for point in review_points %}
- {{ point }}
{% endfor %}

코드:
```{{ language }}
{{ code }}
```

리뷰 결과를 다음 형식으로 작성해주세요:
1. 전체 평가
2. 발견된 문제점
3. 개선 제안
4. 좋은 점"""

    (templates_dir / "code_review.j2").write_text(code_review_template, encoding="utf-8")

    # 3. 이메일 작성 템플릿
    email_template = """{{ recipient }}님께 보낼 {{ email_type }} 이메일을 작성해주세요.

발신자: {{ sender }}
목적: {{ purpose }}
{% if key_points %}
주요 내용:
{% for point in key_points %}
- {{ point }}
{% endfor %}
{% endif %}
톤: {{ tone | default('정중하고 전문적인') }}

이메일 형식:
- 제목
- 본문
- 맺음말"""

    (templates_dir / "email.j2").write_text(email_template, encoding="utf-8")

    # 4. 데이터 분석 템플릿
    analysis_template = """다음 데이터를 분석해주세요.

데이터 유형: {{ data_type }}
분석 목적: {{ purpose }}

데이터:
{% if data is mapping %}
{% for key, value in data.items() %}
{{ key }}: {{ value }}
{% endfor %}
{% else %}
{{ data }}
{% endif %}

다음 관점에서 분석해주세요:
{% for aspect in analysis_aspects %}
{{ loop.index }}. {{ aspect }}
{% endfor %}"""

    (templates_dir / "data_analysis.j2").write_text(analysis_template, encoding="utf-8")

    print("✅ 템플릿 파일 생성 완료: ./templates/")


def example_basic_templates():
    """기본 템플릿 사용 예제"""
    print("\n📝 기본 템플릿 사용 예제")
    print("-" * 50)

    # 템플릿 엔진 생성
    te = TemplateEngine("./templates")
    llm = LLM.create("gpt-4o-mini")

    # 1. 번역 템플릿 사용
    print("1️⃣ 번역 템플릿")

    variables = {
        "target_language": "영어",
        "style": "비즈니스 공식 문서",
        "text": "안녕하세요. 이번 프로젝트 진행 상황을 보고드립니다.",
    }

    prompt = te.render_template("translation.j2", variables)
    print(f"생성된 프롬프트:\n{prompt}\n")

    reply = llm.ask(prompt)
    print(f"번역 결과:\n{reply.text}\n")


def example_code_review_template():
    """코드 리뷰 템플릿 예제"""
    print("\n💻 코드 리뷰 템플릿 예제")
    print("-" * 50)

    te = TemplateEngine("./templates")
    llm = LLM.create("gpt-4o-mini")

    # 리뷰할 코드
    code = """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

# 사용 예
result = fibonacci(10)
print(result)
"""

    variables = {
        "language": "Python",
        "code": code.strip(),
        "review_points": ["코드 효율성", "가독성", "에러 처리", "네이밍 컨벤션", "잠재적 버그"],
    }

    prompt = te.render_template("code_review.j2", **variables)
    reply = llm.ask(prompt)

    print("🔍 코드 리뷰 결과:")
    print(reply.text)


def example_dynamic_templates():
    """동적 템플릿 생성 예제"""
    print("\n🔄 동적 템플릿 생성 예제")
    print("-" * 50)

    llm = LLM.create("gpt-4o-mini")

    # 런타임에 템플릿 생성
    from jinja2 import Template

    # 사용자 정의 템플릿
    custom_template = Template(
        """
    당신은 {{ role }}입니다.
    
    {% if constraints %}
    제약 사항:
    {% for constraint in constraints %}
    - {{ constraint }}
    {% endfor %}
    {% endif %}
    
    사용자 질문: {{ question }}
    
    {% if examples %}
    참고 예시:
    {% for example in examples %}
    예시 {{ loop.index }}: {{ example }}
    {% endfor %}
    {% endif %}
    
    답변:
    """
    )

    # 다양한 역할로 같은 질문에 답변
    question = "인공지능의 미래는 어떻게 될까요?"

    roles = [
        {
            "role": "낙관적인 미래학자",
            "constraints": ["긍정적인 면 강조", "구체적인 예시 포함"],
            "examples": ["의료 분야 혁신", "교육 개인화"],
        },
        {
            "role": "신중한 윤리학자",
            "constraints": ["윤리적 고려사항 포함", "균형잡힌 시각"],
            "examples": ["프라이버시 문제", "일자리 대체"],
        },
        {
            "role": "실용적인 엔지니어",
            "constraints": ["기술적 관점", "현실적인 예측"],
            "examples": ["현재 기술 한계", "향후 10년 전망"],
        },
    ]

    for role_config in roles:
        role_config["question"] = question
        prompt = custom_template.render(**role_config)

        print(f"\n🎭 {role_config['role']}의 답변:")
        reply = llm.ask(prompt)
        print(reply.text[:300] + "...")


def example_template_inheritance():
    """템플릿 상속 예제"""
    print("\n🏗️  템플릿 상속 예제")
    print("-" * 50)

    # 상속 구조를 가진 템플릿 생성
    templates_dir = Path("./templates")

    # 베이스 템플릿
    base_template = """
{% block header %}
당신은 전문적인 AI 어시스턴트입니다.
{% endblock %}

{% block context %}{% endblock %}

{% block task %}
다음 요청을 처리해주세요:
{% endblock %}

{% block content %}{% endblock %}

{% block footer %}
답변은 명확하고 구조화된 형식으로 작성해주세요.
{% endblock %}
"""

    (templates_dir / "base.j2").write_text(base_template, encoding="utf-8")

    # 상속받는 템플릿
    report_template = """
{% extends "base.j2" %}

{% block context %}
보고서 작성 가이드라인:
- 객관적이고 사실 기반
- 데이터 중심적 접근
- 명확한 결론 도출
{% endblock %}

{% block content %}
주제: {{ topic }}
기간: {{ period }}
데이터: {{ data | tojson }}

위 정보를 바탕으로 {{ report_type }} 보고서를 작성해주세요.
{% endblock %}
"""

    (templates_dir / "report.j2").write_text(report_template, encoding="utf-8")

    # 템플릿 사용
    te = TemplateEngine("./templates")
    llm = LLM.create("gpt-4o-mini")

    variables = {
        "topic": "2024년 1분기 판매 실적",
        "period": "2024.01.01 - 2024.03.31",
        "report_type": "분석",
        "data": {"총 매출": "15억원", "전년 대비": "+23%", "베스트셀러": ["제품A", "제품B"], "신규 고객": 1250},
    }

    prompt = te.render_template("report.j2", **variables)
    print("생성된 프롬프트:")
    print("-" * 30)
    print(prompt)
    print("-" * 30)

    reply = llm.ask(prompt)
    print("\n📊 보고서:")
    print(reply.text)


def example_template_filters():
    """템플릿 필터 사용 예제"""
    print("\n🔧 템플릿 필터 사용 예제")
    print("-" * 50)

    from jinja2 import Environment

    # 커스텀 필터 정의
    def format_price(value):
        """가격 포맷팅 필터"""
        return f"{value:,}원"

    def highlight(text, words):
        """텍스트 하이라이트 필터"""
        for word in words:
            text = text.replace(word, f"**{word}**")
        return text

    # 환경 설정
    env = Environment()
    env.filters["format_price"] = format_price
    env.filters["highlight"] = highlight

    # 상품 설명 템플릿
    product_template = env.from_string(
        """
상품명: {{ name | upper }}
가격: {{ price | format_price }}
할인가: {{ (price * discount_rate) | int | format_price }}
설명: {{ description | highlight(keywords) | truncate(100) }}

주요 특징:
{% for feature in features | sort %}
- {{ feature | capitalize }}
{% endfor %}

재고: {% if stock > 10 %}충분{% elif stock > 0 %}부족{% else %}품절{% endif %}
등록일: {{ created_at | default('정보 없음') }}
"""
    )

    llm = LLM.create("gpt-4o-mini")

    # 상품 데이터
    product = {
        "name": "스마트 워치 프로",
        "price": 350000,
        "discount_rate": 0.8,
        "description": "최신 기술이 적용된 스마트 워치로 건강 관리와 일상 생활을 더욱 편리하게 만들어줍니다.",
        "keywords": ["스마트", "건강"],
        "features": ["심박수 모니터링", "GPS 내장", "방수 기능", "7일 배터리"],
        "stock": 15,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
    }

    # 템플릿 렌더링
    rendered = product_template.render(**product)
    print("렌더링된 상품 정보:")
    print(rendered)

    # LLM에 마케팅 문구 요청
    prompt = f"""
다음 상품 정보를 바탕으로 매력적인 마케팅 문구를 3개 작성해주세요:

{rendered}
"""

    reply = llm.ask(prompt)
    print("\n💡 마케팅 문구:")
    print(reply.text)


def example_template_best_practices():
    """템플릿 모범 사례 예제"""
    print("\n✨ 템플릿 모범 사례 예제")
    print("-" * 50)

    # 재사용 가능한 템플릿 컴포넌트
    templates_dir = Path("./templates/components")
    templates_dir.mkdir(parents=True, exist_ok=True)

    # 매크로 정의
    macros_template = """
{% macro format_list(items, title="") -%}
{% if title %}{{ title }}:{% endif %}
{% for item in items %}
{{ loop.index }}. {{ item }}
{% endfor %}
{%- endmacro %}

{% macro format_table(data, headers) -%}
| {% for header in headers %}{{ header }} | {% endfor %}
|{% for _ in headers %} --- |{% endfor %}
{% for row in data %}
| {% for header in headers %}{{ row.get(header, '') }} | {% endfor %}
{% endfor %}
{%- endmacro %}
"""

    (templates_dir / "macros.j2").write_text(macros_template, encoding="utf-8")

    # 매크로를 사용하는 템플릿
    analysis_with_macros = """
{% import 'components/macros.j2' as macros %}

# 데이터 분석 리포트

## 요약
{{ summary }}

## 주요 발견사항
{{ macros.format_list(findings, "주요 발견사항") }}

## 데이터 테이블
{{ macros.format_table(data_table, ['항목', '값', '변화율']) }}

## 권장 사항
{{ macros.format_list(recommendations) }}
"""

    (templates_dir.parent / "analysis_report.j2").write_text(analysis_with_macros, encoding="utf-8")

    # 사용
    te = TemplateEngine("./templates")
    llm = LLM.create("gpt-4o-mini")

    report_data = {
        "summary": "2024년 1분기 실적이 전년 대비 크게 향상되었습니다.",
        "findings": ["매출 23% 증가", "신규 고객 45% 증가", "고객 만족도 92% 달성"],
        "data_table": [
            {"항목": "매출", "값": "15억원", "변화율": "+23%"},
            {"항목": "영업이익", "값": "3억원", "변화율": "+30%"},
            {"항목": "순이익", "값": "2.5억원", "변화율": "+28%"},
        ],
        "recommendations": [
            "성장 모멘텀 유지를 위한 마케팅 강화",
            "고객 서비스 품질 지속적 개선",
            "신제품 라인 확대 검토",
        ],
    }

    prompt = te.render_template("analysis_report.j2", **report_data)
    print("생성된 분석 리포트 템플릿:")
    print(prompt)

    # 추가 인사이트 요청
    insight_prompt = f"{prompt}\n\n위 리포트를 바탕으로 추가적인 전략적 인사이트를 제공해주세요."
    reply = llm.ask(insight_prompt)

    print("\n🎯 전략적 인사이트:")
    print(reply.text)


def main():
    """템플릿 활용 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        return

    print("📋 템플릿 활용 예제")
    print("=" * 50)

    try:
        # 템플릿 생성
        create_sample_templates()

        # 1. 기본 템플릿 사용
        example_basic_templates()

        # 2. 코드 리뷰 템플릿
        example_code_review_template()

        # 3. 동적 템플릿
        example_dynamic_templates()

        # 4. 템플릿 상속
        example_template_inheritance()

        # 5. 템플릿 필터
        example_template_filters()

        # 6. 모범 사례
        example_template_best_practices()

        print("\n✅ 모든 템플릿 예제 완료!")

        # 정리
        response = input("\n생성된 템플릿 파일을 삭제하시겠습니까? (y/n): ")
        if response.lower() == "y":
            import shutil

            if Path("./templates").exists():
                shutil.rmtree("./templates")
                print("템플릿 파일이 삭제되었습니다.")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
