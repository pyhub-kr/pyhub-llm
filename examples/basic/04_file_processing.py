#!/usr/bin/env python3
"""
예제: 파일 처리 (텍스트 및 이미지)
난이도: 초급
설명: 파일을 읽어서 처리하는 방법
요구사항: OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import os
import sys
from pathlib import Path

from pyhub.llm import LLM


def process_text_file(llm, file_path):
    """텍스트 파일 처리 예제"""
    print(f"\n📄 텍스트 파일 처리: {file_path}")

    try:
        # 파일 읽기
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"파일 크기: {len(content)}자")
        print("내용 미리보기:", content[:100] + "..." if len(content) > 100 else content)

        # 파일 내용 요약
        prompt = f"다음 텍스트를 한 문단으로 요약해주세요:\n\n{content}"
        reply = llm.ask(prompt)

        print(f"\n📝 요약 결과:\n{reply.text}")

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


def process_image_file(llm, image_path):
    """이미지 파일 처리 예제"""
    print(f"\n🖼️  이미지 파일 처리: {image_path}")

    try:
        # 파일 존재 확인
        if not os.path.exists(image_path):
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
            return

        # 이미지 분석
        prompt = "이 이미지를 자세히 설명해주세요."
        reply = llm.ask(prompt, files=[image_path])

        print(f"\n🔍 이미지 분석 결과:\n{reply.text}")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


def create_sample_files():
    """예제 파일 생성"""
    # 샘플 텍스트 파일 생성
    sample_text = """파이썬(Python)은 1991년 귀도 반 로섬(Guido van Rossum)이 개발한 고급 프로그래밍 언어입니다.
파이썬은 플랫폼에 독립적이며 인터프리터식, 객체지향적, 동적 타이핑 대화형 언어입니다.
파이썬이라는 이름은 코미디 그룹 몬티 파이썬에서 따온 것입니다.

파이썬의 주요 특징:
1. 간결하고 읽기 쉬운 문법
2. 풍부한 표준 라이브러리
3. 다양한 프로그래밍 패러다임 지원
4. 활발한 커뮤니티와 생태계

파이썬은 웹 개발, 데이터 분석, 인공지능, 자동화 등 다양한 분야에서 널리 사용되고 있습니다."""

    with open("sample_text.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    print("✅ 샘플 텍스트 파일 생성: sample_text.txt")

    # 이미지 파일 생성 안내
    print("💡 이미지 분석을 테스트하려면 'sample_image.jpg' 또는 'sample_image.png' 파일을 준비해주세요.")


def main():
    """파일 처리 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("📁 파일 처리 예제")
    print("=" * 50)

    # LLM 생성 (이미지 처리를 위해 vision 모델 사용)
    llm = LLM.create("gpt-4o-mini")

    # 샘플 파일 생성
    create_sample_files()

    # 1. 텍스트 파일 처리
    if os.path.exists("sample_text.txt"):
        process_text_file(llm, "sample_text.txt")

    # 2. 이미지 파일 처리 (있는 경우)
    image_files = ["sample_image.jpg", "sample_image.png", "test.jpg", "test.png"]
    image_found = False

    for img_file in image_files:
        if os.path.exists(img_file):
            process_image_file(llm, img_file)
            image_found = True
            break

    if not image_found:
        print("\n💡 이미지 파일이 없어 이미지 분석을 건너뜁니다.")
        print("   이미지 분석을 테스트하려면 위 파일명 중 하나로 이미지를 준비해주세요.")

    # 3. 여러 파일 동시 처리
    print("\n🎯 여러 파일 동시 처리 예제")

    # 현재 디렉토리의 모든 .txt 파일 찾기
    txt_files = list(Path(".").glob("*.txt"))[:3]  # 최대 3개까지만

    if len(txt_files) > 1:
        print(f"발견된 텍스트 파일: {[f.name for f in txt_files]}")

        # 모든 파일 내용을 하나로 합치기
        all_contents = []
        for file_path in txt_files:
            with open(file_path, "r", encoding="utf-8") as f:
                all_contents.append(f"[{file_path.name}]\n{f.read()}")

        combined = "\n\n".join(all_contents)
        prompt = f"다음 {len(txt_files)}개 파일의 공통 주제나 연관성을 찾아주세요:\n\n{combined[:1000]}..."

        reply = llm.ask(prompt)
        print(f"\n📊 분석 결과:\n{reply.text}")

    # 정리
    print("\n✅ 파일 처리 예제 완료!")

    # 생성된 샘플 파일 삭제 옵션
    if os.path.exists("sample_text.txt"):
        response = input("\n생성된 샘플 파일을 삭제하시겠습니까? (y/n): ")
        if response.lower() == "y":
            os.remove("sample_text.txt")
            print("샘플 파일이 삭제되었습니다.")


if __name__ == "__main__":
    main()
