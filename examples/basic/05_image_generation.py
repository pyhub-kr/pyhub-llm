"""
이미지 생성 예제
================

OpenAI의 DALL-E 모델을 사용하여 텍스트로부터 이미지를 생성합니다.
"""

import asyncio
from pathlib import Path
from pyhub.llm import OpenAILLM


def basic_image_generation():
    """기본 이미지 생성"""
    print("=== 기본 이미지 생성 ===")

    # DALL-E 3 모델 생성
    llm = OpenAILLM(model="dall-e-3")

    # 이미지 생성
    reply = llm.generate_image("A serene Japanese garden with cherry blossoms")

    print(f"이미지 URL: {reply.url}")
    print(f"크기: {reply.size}")
    print(f"모델: {reply.model}")

    if reply.revised_prompt:
        print(f"개선된 프롬프트: {reply.revised_prompt}")

    # 이미지 저장
    path = reply.save("japanese_garden.png")
    print(f"저장됨: {path}")


def advanced_options():
    """고급 옵션 사용"""
    print("\n=== 고급 옵션 ===")

    llm = OpenAILLM(model="dall-e-3")

    # 세로 형식, 고품질 이미지
    reply = llm.generate_image(
        "A majestic waterfall in a tropical forest",
        size="1024x1792",  # 세로 형식
        quality="hd",  # 고품질
        style="natural",  # 자연스러운 스타일
    )

    print(f"고품질 이미지 생성됨: {reply.size}")

    # 디렉토리에 저장 (자동 파일명)
    Path("outputs").mkdir(exist_ok=True)
    path = reply.save("outputs/")
    print(f"저장됨: {path}")


def save_variations():
    """다양한 저장 방법"""
    print("\n=== 다양한 저장 방법 ===")

    llm = OpenAILLM(model="dall-e-3")
    reply = llm.generate_image("A colorful abstract art piece")

    # 1. 현재 디렉토리에 자동 파일명으로 저장
    path1 = reply.save()
    print(f"자동 파일명: {path1}")

    # 2. 특정 파일명으로 저장
    path2 = reply.save("abstract_art.png")
    print(f"지정 파일명: {path2}")

    # 3. 파일이 이미 존재할 때 자동 번호 부여
    path3 = reply.save("abstract_art.png")  # abstract_art_1.png
    print(f"중복 방지: {path3}")

    # 4. BytesIO에 저장 (NEW!)
    from io import BytesIO

    buffer = BytesIO()
    reply.save(buffer)
    print(f"BytesIO에 저장됨, 크기: {len(buffer.getvalue())} bytes")

    # BytesIO에서 읽어서 다시 저장 가능
    buffer.seek(0)
    with open("abstract_from_buffer.png", "wb") as f:
        f.write(buffer.read())
    print("BytesIO에서 파일로 저장됨: abstract_from_buffer.png")


def image_manipulation():
    """이미지 조작 (Pillow 필요)"""
    print("\n=== 이미지 조작 ===")

    llm = OpenAILLM(model="dall-e-3")
    reply = llm.generate_image("A cute robot character")

    try:
        # PIL 이미지로 변환
        img = reply.to_pil()

        # 썸네일 생성
        thumbnail = img.copy()
        thumbnail.thumbnail((256, 256))
        thumbnail.save("robot_thumbnail.png")
        print("썸네일 생성됨: robot_thumbnail.png")

        # 크기 조정
        resized = img.resize((512, 512))
        resized.save("robot_512x512.png")
        print("크기 조정됨: robot_512x512.png")

        # 회전
        rotated = img.rotate(45, expand=True)
        rotated.save("robot_rotated.png")
        print("회전됨: robot_rotated.png")

    except ImportError:
        print("Pillow가 필요합니다: pip install 'pyhub-llm[image]'")


def check_model_support():
    """모델별 이미지 생성 지원 확인"""
    print("\n=== 모델 지원 확인 ===")

    models = [
        ("gpt-4o", OpenAILLM),
        ("gpt-4o-mini", OpenAILLM),
        ("dall-e-3", OpenAILLM),
        ("dall-e-2", OpenAILLM),
    ]

    for model_name, llm_class in models:
        llm = llm_class(model=model_name)
        supports = llm.supports("image_generation")

        if supports:
            sizes = llm.get_supported_image_sizes()
            print(f"{model_name}: ✅ 지원 (크기: {sizes})")
        else:
            print(f"{model_name}: ❌ 미지원")


async def async_generation():
    """비동기 이미지 생성"""
    print("\n=== 비동기 이미지 생성 ===")

    llm = OpenAILLM(model="dall-e-3")

    # 여러 이미지 동시 생성
    prompts = ["A futuristic city skyline", "An underwater coral reef", "A mountain landscape at sunrise"]

    # 병렬 생성
    tasks = [llm.generate_image_async(prompt) for prompt in prompts]
    images = await asyncio.gather(*tasks)

    print(f"{len(images)}개 이미지 생성 완료")

    # 병렬 저장
    Path("async_outputs").mkdir(exist_ok=True)
    save_tasks = [img.save_async(f"async_outputs/image_{i}.png") for i, img in enumerate(images)]
    paths = await asyncio.gather(*save_tasks)

    for path in paths:
        print(f"저장됨: {path}")


if __name__ == "__main__":
    # 기본 예제들 실행
    basic_image_generation()
    advanced_options()
    save_variations()
    image_manipulation()
    check_model_support()

    # 비동기 예제 실행
    print("\n비동기 예제를 실행하려면 주석을 해제하세요:")
    print("# asyncio.run(async_generation())")

    # 비동기 예제 (주석 해제하여 실행)
    # asyncio.run(async_generation())
