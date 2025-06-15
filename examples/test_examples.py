#!/usr/bin/env python3
"""
예제 테스트 스크립트
모든 예제가 정상적으로 실행되는지 확인합니다.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_requirements():
    """필수 패키지 확인"""
    print("📦 패키지 확인 중...")
    
    required_packages = {
        'pyhub.llm': 'pyhub-llm',
        'pydantic': 'pydantic',
        'jinja2': 'jinja2'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package}")
    
    if missing:
        print(f"\n⚠️  다음 패키지를 설치해주세요:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    # 환경 변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        print("export OPENAI_API_KEY='your-api-key'")
        return False
    
    print("\n✅ 모든 요구사항이 충족되었습니다.\n")
    return True


def test_example(file_path: Path, timeout: int = 30):
    """개별 예제 테스트"""
    print(f"🧪 테스트: {file_path.name}")
    
    try:
        # 대화형 예제는 건너뛰기
        if file_path.name in ['03_conversation.py']:
            print("  ⏭️  대화형 예제는 건너뜁니다.")
            return True
        
        # 예제 실행
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, 'PYTHONPATH': str(Path(__file__).parent.parent)}
        )
        
        if result.returncode == 0:
            print(f"  ✅ 성공")
            # 출력의 첫 몇 줄만 표시
            output_lines = result.stdout.strip().split('\n')[:3]
            for line in output_lines:
                print(f"     {line[:60]}...")
            return True
        else:
            print(f"  ❌ 실패 (종료 코드: {result.returncode})")
            print(f"     오류: {result.stderr.strip()[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  시간 초과 ({timeout}초)")
        return False
    except Exception as e:
        print(f"  ❌ 예외 발생: {e}")
        return False


def test_directory(directory: Path):
    """디렉토리의 모든 예제 테스트"""
    print(f"\n📁 {directory.name} 예제 테스트")
    print("=" * 50)
    
    py_files = sorted(directory.glob("*.py"))
    if not py_files:
        print("  파이썬 파일이 없습니다.")
        return 0, 0  # 튜플 반환
    
    success_count = 0
    total_count = 0
    
    for py_file in py_files:
        # README나 __init__ 파일은 건너뛰기
        if py_file.name in ['__init__.py', 'test_examples.py']:
            continue
            
        total_count += 1
        if test_example(py_file):
            success_count += 1
        print()
    
    print(f"📊 결과: {success_count}/{total_count} 성공")
    return success_count, total_count


def main():
    """메인 테스트 함수"""
    print("🚀 pyhub-llm 예제 테스트")
    print("=" * 50)
    
    # 요구사항 확인
    if not check_requirements():
        sys.exit(1)
    
    # 예제 디렉토리 찾기
    examples_dir = Path(__file__).parent
    
    # 각 난이도별 테스트
    total_success = 0
    total_count = 0
    
    for subdir in ['basic', 'intermediate', 'advanced']:
        dir_path = examples_dir / subdir
        if dir_path.exists() and dir_path.is_dir():
            success, count = test_directory(dir_path)
            total_success += success
            total_count += count
    
    # 최종 결과
    print("\n" + "=" * 50)
    print(f"🎯 전체 결과: {total_success}/{total_count} 성공")
    
    if total_success == total_count:
        print("✅ 모든 예제가 정상적으로 실행되었습니다!")
        sys.exit(0)
    else:
        print("⚠️  일부 예제에서 오류가 발생했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()