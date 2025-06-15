"""
eval() 사용 제거 및 안전한 대안 테스트
"""

import os
import sys

import pytest

# 테스트를 위해 examples 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_safe_math_evaluation():
    """안전한 수학 표현식 평가 테스트"""
    try:
        import simpleeval
    except ImportError:
        pytest.skip("simpleeval 라이브러리가 설치되지 않았습니다")

    import math

    evaluator = simpleeval.SimpleEval()
    # 수학 함수들 추가
    evaluator.functions.update(
        {"abs": abs, "round": round, "min": min, "max": max, "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt}
    )

    # 안전한 수학 표현식들
    test_cases = [
        ("2 + 3", 5),
        ("10 * 5", 50),
        ("15 / 3", 5.0),
        ("2 ** 3", 8),
        ("(1 + 2) * 3", 9),
        ("abs(-5)", 5),
        ("max(1, 2, 3)", 3),
        ("min(4, 5, 6)", 4),
    ]

    for expression, expected in test_cases:
        result = evaluator.eval(expression)
        assert result == expected, f"Expression '{expression}' should equal {expected}, got {result}"


def test_dangerous_expressions_blocked():
    """위험한 표현식이 차단되는지 테스트"""
    try:
        import simpleeval
    except ImportError:
        pytest.skip("simpleeval 라이브러리가 설치되지 않았습니다")

    evaluator = simpleeval.SimpleEval()

    # 위험한 표현식들
    dangerous_expressions = [
        "__import__('os').system('ls')",
        "exec('print(\"hack\")')",
        "eval('1+1')",
        "open('/etc/passwd').read()",
        "globals()",
        "locals()",
    ]

    for expression in dangerous_expressions:
        with pytest.raises((simpleeval.NameNotDefined, simpleeval.FunctionNotDefined, ValueError)):
            evaluator.eval(expression)


def test_calculator_function_safety():
    """계산기 함수의 안전성 테스트"""

    def safe_calculate(expression: str) -> str:
        """안전한 계산 함수"""
        try:
            import simpleeval
        except ImportError:
            # simpleeval이 없으면 기본 제한된 eval 사용
            import math

            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "log": math.log,
                "pi": math.pi,
                "e": math.e,
            }
            try:
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"

        evaluator = simpleeval.SimpleEval()
        try:
            result = evaluator.eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    # 안전한 계산 테스트
    assert safe_calculate("2 + 3") == "5"
    assert safe_calculate("10 * 5") == "50"

    # 위험한 계산은 에러 반환
    result = safe_calculate("__import__('os').system('ls')")
    assert result.startswith("Error:")


def test_allowed_characters_validation():
    """허용된 문자만 사용하는지 검증 테스트"""

    def is_safe_expression(expression: str) -> bool:
        """수학 표현식에 안전한 문자만 포함되어 있는지 확인"""
        import re

        # 숫자, 연산자, 괄호, 공백, 점(소수점), 기본 함수명만 허용
        allowed_pattern = re.compile(r"^[0-9+\-*/().,\s_a-zA-Z]+$")
        if not allowed_pattern.match(expression):
            return False

        # 위험한 키워드 확인
        dangerous_keywords = ["import", "exec", "eval", "open", "file", "globals", "locals", "__", "os", "sys"]
        expression_lower = expression.lower()
        return not any(keyword in expression_lower for keyword in dangerous_keywords)

    # 안전한 표현식
    assert is_safe_expression("2 + 3 * 4")
    assert is_safe_expression("sqrt(16)")
    assert is_safe_expression("sin(3.14)")

    # 위험한 표현식
    assert not is_safe_expression("__import__('os').system('ls')")
    assert not is_safe_expression("exec('print(1)')")
    assert not is_safe_expression("open('/etc/passwd')")


if __name__ == "__main__":
    # 직접 실행시 간단한 테스트 수행
    print("🔒 eval() 안전성 테스트 시작...")

    try:
        test_safe_math_evaluation()
        print("✅ 안전한 수학 표현식 평가 테스트 통과")
    except Exception as e:
        print(f"❌ 안전한 수학 표현식 평가 테스트 실패: {e}")

    try:
        test_dangerous_expressions_blocked()
        print("✅ 위험한 표현식 차단 테스트 통과")
    except Exception as e:
        print(f"❌ 위험한 표현식 차단 테스트 실패: {e}")

    try:
        test_calculator_function_safety()
        print("✅ 계산기 함수 안전성 테스트 통과")
    except Exception as e:
        print(f"❌ 계산기 함수 안전성 테스트 실패: {e}")

    try:
        test_allowed_characters_validation()
        print("✅ 허용 문자 검증 테스트 통과")
    except Exception as e:
        print(f"❌ 허용 문자 검증 테스트 실패: {e}")

    print("🔒 eval() 안전성 테스트 완료")
