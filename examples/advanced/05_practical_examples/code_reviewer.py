#!/usr/bin/env python3
"""
예제: AI 코드 리뷰어
난이도: 고급
설명: Python 코드를 분석하고 개선 제안을 제공하는 시스템
요구사항:
  - pyhub-llm (pip install "pyhub-llm[all]")
  - ast (내장 모듈)
  - OPENAI_API_KEY 환경 변수

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyhub.llm import LLM

# from pyhub.llm.cache import MemoryCache  # 캐시 기능은 현재 예제에서 사용하지 않음


class IssueSeverity(Enum):
    """이슈 심각도"""

    ERROR = "error"  # 버그나 오류
    WARNING = "warning"  # 잠재적 문제
    INFO = "info"  # 개선 제안
    STYLE = "style"  # 스타일 관련


class IssueCategory(Enum):
    """이슈 카테고리"""

    BUG = "bug"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    BEST_PRACTICE = "best_practice"


@dataclass
class CodeIssue:
    """코드 이슈"""

    line: int
    column: int
    severity: IssueSeverity
    category: IssueCategory
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class CodeMetrics:
    """코드 메트릭"""

    lines_of_code: int
    cyclomatic_complexity: int
    function_count: int
    class_count: int
    import_count: int
    comment_ratio: float
    test_coverage: Optional[float] = None


@dataclass
class ReviewResult:
    """리뷰 결과"""

    file_path: str
    issues: List[CodeIssue] = field(default_factory=list)
    metrics: Optional[CodeMetrics] = None
    score: float = 0.0
    summary: str = ""
    improvements: List[str] = field(default_factory=list)


class CodeReviewer:
    """AI 코드 리뷰어"""

    def __init__(self, model: str = "gpt-4o"):
        self.llm = LLM.create(model)
        # self.cache = MemoryCache()  # 캐시 기능은 현재 예제에서 사용하지 않음
        self.style_guide = self._load_style_guide()

    def _load_style_guide(self) -> Dict[str, Any]:
        """스타일 가이드 로드"""
        return {
            "max_line_length": 88,
            "max_function_length": 50,
            "max_complexity": 10,
            "naming_convention": "snake_case",
            "docstring_required": True,
            "type_hints_required": True,
        }

    def analyze_code_structure(self, code: str) -> Tuple[ast.AST, List[str]]:
        """코드 구조 분석"""
        try:
            tree = ast.parse(code)
            errors = []
            return tree, errors
        except SyntaxError as e:
            return None, [f"Syntax error at line {e.lineno}: {e.msg}"]

    def calculate_metrics(self, code: str, tree: Optional[ast.AST]) -> CodeMetrics:
        """코드 메트릭 계산"""
        lines = code.split("\n")
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith("#")])

        if not tree:
            return CodeMetrics(
                lines_of_code=lines_of_code,
                cyclomatic_complexity=0,
                function_count=0,
                class_count=0,
                import_count=0,
                comment_ratio=0.0,
            )

        # 함수와 클래스 수 계산
        function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        import_count = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))

        # 주석 비율 계산
        comment_lines = len([line for line in lines if line.strip().startswith("#")])
        comment_ratio = comment_lines / len(lines) if lines else 0.0

        # 순환 복잡도 계산 (간단한 버전)
        complexity = self._calculate_cyclomatic_complexity(tree)

        return CodeMetrics(
            lines_of_code=lines_of_code,
            cyclomatic_complexity=complexity,
            function_count=function_count,
            class_count=class_count,
            import_count=import_count,
            comment_ratio=comment_ratio,
        )

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """순환 복잡도 계산"""
        complexity = 1  # 기본 복잡도

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def check_style_issues(self, code: str, tree: Optional[ast.AST]) -> List[CodeIssue]:
        """스타일 이슈 검사"""
        issues = []
        lines = code.split("\n")

        # 라인 길이 검사
        for i, line in enumerate(lines, 1):
            if len(line) > self.style_guide["max_line_length"]:
                issues.append(
                    CodeIssue(
                        line=i,
                        column=self.style_guide["max_line_length"],
                        severity=IssueSeverity.STYLE,
                        category=IssueCategory.STYLE,
                        message=f"Line too long ({len(line)} > {self.style_guide['max_line_length']})",
                        suggestion="Break this line into multiple lines",
                    )
                )

        if tree:
            # 함수 길이 검사
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_lines = node.end_lineno - node.lineno + 1
                    if func_lines > self.style_guide["max_function_length"]:
                        issues.append(
                            CodeIssue(
                                line=node.lineno,
                                column=0,
                                severity=IssueSeverity.WARNING,
                                category=IssueCategory.MAINTAINABILITY,
                                message=f"Function '{node.name}' is too long ({func_lines} lines)",
                                suggestion="Consider breaking this function into smaller functions",
                            )
                        )

                    # 독스트링 검사
                    if self.style_guide["docstring_required"]:
                        docstring = ast.get_docstring(node)
                        if not docstring:
                            issues.append(
                                CodeIssue(
                                    line=node.lineno,
                                    column=0,
                                    severity=IssueSeverity.INFO,
                                    category=IssueCategory.STYLE,
                                    message=f"Function '{node.name}' lacks a docstring",
                                    suggestion="Add a docstring describing the function's purpose",
                                )
                            )

        return issues

    def check_security_issues(self, code: str, tree: Optional[ast.AST]) -> List[CodeIssue]:
        """보안 이슈 검사"""
        issues = []

        # 위험한 함수 사용 검사
        dangerous_functions = {
            "eval": "Avoid using eval() as it can execute arbitrary code",
            "exec": "Avoid using exec() as it can execute arbitrary code",
            "__import__": "Dynamic imports can be a security risk",
            "pickle.loads": "Pickle can execute arbitrary code during deserialization",
        }

        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = self._get_function_name(node)
                    if func_name in dangerous_functions:
                        issues.append(
                            CodeIssue(
                                line=node.lineno,
                                column=node.col_offset,
                                severity=IssueSeverity.WARNING,
                                category=IssueCategory.SECURITY,
                                message=dangerous_functions[func_name],
                                suggestion="Consider using a safer alternative",
                            )
                        )

        # 하드코딩된 비밀번호 검사
        secret_patterns = [
            (r'password\s*=\s*["\'].*["\']', "Hardcoded password detected"),
            (r'api_key\s*=\s*["\'].*["\']', "Hardcoded API key detected"),
            (r'secret\s*=\s*["\'].*["\']', "Hardcoded secret detected"),
        ]

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern, message in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(
                        CodeIssue(
                            line=i,
                            column=0,
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.SECURITY,
                            message=message,
                            suggestion="Use environment variables or secure configuration files",
                        )
                    )

        return issues

    def _get_function_name(self, node: ast.Call) -> str:
        """함수 이름 추출"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # value가 Name인 경우 직접 id를 가져옴
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            # value가 다른 Attribute인 경우 재귀적으로 처리
            elif isinstance(node.func.value, ast.Attribute):
                base = self._get_attribute_chain(node.func.value)
                return f"{base}.{node.func.attr}"
        return ""

    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """중첩된 attribute 체인 추출"""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_chain(node.value)}.{node.attr}"
        return node.attr

    def check_performance_issues(self, code: str, tree: Optional[ast.AST]) -> List[CodeIssue]:
        """성능 이슈 검사"""
        issues = []

        if tree:
            # 리스트 컴프리헨션 vs 루프
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    # 단순 append 패턴 검사
                    if self._is_simple_append_loop(node):
                        issues.append(
                            CodeIssue(
                                line=node.lineno,
                                column=0,
                                severity=IssueSeverity.INFO,
                                category=IssueCategory.PERFORMANCE,
                                message="Consider using list comprehension for better performance",
                                suggestion="Replace loop with list comprehension",
                            )
                        )

        return issues

    def _is_simple_append_loop(self, node: ast.For) -> bool:
        """단순 append 루프인지 확인"""
        # 간단한 휴리스틱
        if len(node.body) == 1 and isinstance(node.body[0], ast.Expr):
            expr = node.body[0].value
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
                return expr.func.attr == "append"
        return False

    def ai_review(self, code: str, context: str = "") -> Dict[str, Any]:
        """AI 기반 코드 리뷰"""
        # 캐시 확인 (캐시 기능은 현재 비활성화)
        # cache_key = f"ai_review_{hash(code)}"
        # cached = self.cache.get(cache_key)
        # if cached:
        #     return cached

        prompt = f"""다음 Python 코드를 리뷰하고 개선 제안을 제공하세요:

```python
{code[:2000]}  # 토큰 제한
```

{f"컨텍스트: {context}" if context else ""}

다음 관점에서 분석하세요:
1. 코드 품질과 가독성
2. 잠재적 버그나 오류
3. 성능 최적화 기회
4. 보안 취약점
5. 베스트 프랙티스 준수

JSON 형식으로 응답하세요:
{{
    "summary": "전체 평가 요약",
    "score": 0-100 점수,
    "issues": [
        {{
            "type": "bug|performance|security|style",
            "severity": "high|medium|low",
            "description": "이슈 설명",
            "suggestion": "개선 제안"
        }}
    ],
    "improvements": ["개선 제안 1", "개선 제안 2", ...]
}}"""

        reply = self.llm.ask(prompt)

        # JSON 파싱
        try:
            # JSON 블록 추출
            json_match = re.search(r"\{.*\}", reply.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"summary": reply.text, "score": 70, "issues": [], "improvements": []}
        except json.JSONDecodeError:
            result = {"summary": "AI 리뷰 파싱 실패", "score": 0, "issues": [], "improvements": []}

        # 캐시 저장 (캐시 기능은 현재 비활성화)
        # self.cache.set(cache_key, result)

        return result

    def review_file(self, file_path: str, context: str = "") -> ReviewResult:
        """파일 리뷰"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        if path.suffix != ".py":
            raise ValueError("Python 파일만 지원됩니다 (.py)")

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        return self.review_code(code, str(file_path), context)

    def review_code(self, code: str, file_name: str = "code.py", context: str = "") -> ReviewResult:
        """코드 리뷰 메인 함수"""
        print(f"🔍 코드 리뷰 시작: {file_name}")

        # 코드 구조 분석
        tree, syntax_errors = self.analyze_code_structure(code)

        # 결과 초기화
        result = ReviewResult(file_path=file_name)

        # 문법 오류가 있으면 추가
        for error in syntax_errors:
            result.issues.append(
                CodeIssue(line=1, column=0, severity=IssueSeverity.ERROR, category=IssueCategory.BUG, message=error)
            )

        # 메트릭 계산
        result.metrics = self.calculate_metrics(code, tree)
        print(f"  📊 메트릭: {result.metrics.lines_of_code} LOC, 복잡도 {result.metrics.cyclomatic_complexity}")

        # 각종 검사 수행
        if tree:
            result.issues.extend(self.check_style_issues(code, tree))
            result.issues.extend(self.check_security_issues(code, tree))
            result.issues.extend(self.check_performance_issues(code, tree))

        # AI 리뷰
        print("  🤖 AI 분석 중...")
        ai_result = self.ai_review(code, context)

        # AI 결과 통합
        result.summary = ai_result.get("summary", "")
        result.score = ai_result.get("score", 0)
        result.improvements = ai_result.get("improvements", [])

        # AI 이슈를 CodeIssue로 변환
        for ai_issue in ai_result.get("issues", []):
            severity_map = {"high": IssueSeverity.ERROR, "medium": IssueSeverity.WARNING, "low": IssueSeverity.INFO}
            category_map = {
                "bug": IssueCategory.BUG,
                "performance": IssueCategory.PERFORMANCE,
                "security": IssueCategory.SECURITY,
                "style": IssueCategory.STYLE,
            }

            result.issues.append(
                CodeIssue(
                    line=0,  # AI는 특정 라인을 지정하지 않음
                    column=0,
                    severity=severity_map.get(ai_issue.get("severity", "low"), IssueSeverity.INFO),
                    category=category_map.get(ai_issue.get("type", "style"), IssueCategory.BEST_PRACTICE),
                    message=ai_issue.get("description", ""),
                    suggestion=ai_issue.get("suggestion", ""),
                )
            )

        print(f"  ✅ 리뷰 완료: {len(result.issues)}개 이슈 발견")

        return result

    def generate_report(self, result: ReviewResult, format: str = "text") -> str:
        """리뷰 보고서 생성"""
        if format == "text":
            return self._generate_text_report(result)
        elif format == "markdown":
            return self._generate_markdown_report(result)
        elif format == "json":
            return self._generate_json_report(result)
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")

    def _generate_text_report(self, result: ReviewResult) -> str:
        """텍스트 보고서 생성"""
        lines = []
        lines.append(f"코드 리뷰 보고서: {result.file_path}")
        lines.append("=" * 50)
        lines.append(f"점수: {result.score}/100")
        lines.append(f"이슈: {len(result.issues)}개")
        lines.append("")

        if result.summary:
            lines.append("요약:")
            lines.append(result.summary)
            lines.append("")

        if result.metrics:
            lines.append("메트릭:")
            lines.append(f"  - 코드 라인: {result.metrics.lines_of_code}")
            lines.append(f"  - 복잡도: {result.metrics.cyclomatic_complexity}")
            lines.append(f"  - 함수 수: {result.metrics.function_count}")
            lines.append(f"  - 클래스 수: {result.metrics.class_count}")
            lines.append("")

        if result.issues:
            lines.append("이슈:")
            for issue in sorted(result.issues, key=lambda x: (x.severity.value, x.line)):
                lines.append(f"  [{issue.severity.value.upper()}] Line {issue.line}: {issue.message}")
                if issue.suggestion:
                    lines.append(f"    제안: {issue.suggestion}")
            lines.append("")

        if result.improvements:
            lines.append("개선 제안:")
            for i, improvement in enumerate(result.improvements, 1):
                lines.append(f"  {i}. {improvement}")

        return "\n".join(lines)

    def _generate_markdown_report(self, result: ReviewResult) -> str:
        """마크다운 보고서 생성"""
        lines = []
        lines.append(f"# 코드 리뷰 보고서: `{result.file_path}`")
        lines.append("")
        lines.append(f"**점수**: {result.score}/100 {'⭐' * (result.score // 20)}")
        lines.append("")

        if result.summary:
            lines.append("## 요약")
            lines.append(result.summary)
            lines.append("")

        if result.metrics:
            lines.append("## 메트릭")
            lines.append(f"- **코드 라인**: {result.metrics.lines_of_code}")
            lines.append(f"- **순환 복잡도**: {result.metrics.cyclomatic_complexity}")
            lines.append(f"- **함수 수**: {result.metrics.function_count}")
            lines.append(f"- **클래스 수**: {result.metrics.class_count}")
            lines.append("")

        if result.issues:
            lines.append("## 이슈")

            # 심각도별 그룹화
            by_severity = {}
            for issue in result.issues:
                if issue.severity not in by_severity:
                    by_severity[issue.severity] = []
                by_severity[issue.severity].append(issue)

            for severity in [IssueSeverity.ERROR, IssueSeverity.WARNING, IssueSeverity.INFO, IssueSeverity.STYLE]:
                if severity in by_severity:
                    emoji = {"error": "🔴", "warning": "🟡", "info": "🔵", "style": "⚪"}
                    lines.append(f"### {emoji[severity.value]} {severity.value.capitalize()}")
                    for issue in by_severity[severity]:
                        lines.append(f"- **Line {issue.line}**: {issue.message}")
                        if issue.suggestion:
                            lines.append(f"  - 💡 {issue.suggestion}")
                    lines.append("")

        if result.improvements:
            lines.append("## 개선 제안")
            for improvement in result.improvements:
                lines.append(f"- {improvement}")

        return "\n".join(lines)

    def _generate_json_report(self, result: ReviewResult) -> str:
        """JSON 보고서 생성"""
        data = {
            "file_path": result.file_path,
            "score": result.score,
            "summary": result.summary,
            "metrics": (
                {
                    "lines_of_code": result.metrics.lines_of_code,
                    "cyclomatic_complexity": result.metrics.cyclomatic_complexity,
                    "function_count": result.metrics.function_count,
                    "class_count": result.metrics.class_count,
                    "import_count": result.metrics.import_count,
                    "comment_ratio": result.metrics.comment_ratio,
                }
                if result.metrics
                else None
            ),
            "issues": [
                {
                    "line": issue.line,
                    "column": issue.column,
                    "severity": issue.severity.value,
                    "category": issue.category.value,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                }
                for issue in result.issues
            ],
            "improvements": result.improvements,
        }

        return json.dumps(data, ensure_ascii=False, indent=2)


def example_simple_review():
    """간단한 코드 리뷰 예제"""
    print("\n📝 간단한 코드 리뷰")
    print("-" * 50)

    sample_code = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    average = total / len(numbers)
    return average

# 테스트
nums = [1, 2, 3, 4, 5]
result = calculate_average(nums)
print("Average:", result)

password = "12345"  # 하드코딩된 비밀번호
"""

    reviewer = CodeReviewer()
    result = reviewer.review_code(sample_code, "example.py")

    # 보고서 출력
    print("\n" + reviewer.generate_report(result, format="text"))


def example_file_review():
    """파일 리뷰 예제"""
    print("\n📁 파일 리뷰")
    print("-" * 50)

    # 예제 파일 생성
    sample_code = '''
import os
import sys

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process_data(self, input_data):
        """데이터 처리 함수"""
        result = []
        for item in input_data:
            if item > 0:
                result.append(item * 2)
        
        # 매우 긴 라인으로 스타일 이슈 생성
        very_long_variable_name = "This is a very long string that exceeds the maximum line length recommended by PEP8"
        
        return result
    
    def risky_operation(self, user_input):
        # 보안 이슈: eval 사용
        return eval(user_input)

def main():
    processor = DataProcessor()
    data = [1, 2, 3, 4, 5]
    result = processor.process_data(data)
    print(result)

if __name__ == "__main__":
    main()
'''

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(sample_code)
        temp_path = f.name

    try:
        reviewer = CodeReviewer()
        result = reviewer.review_file(temp_path, context="데이터 처리 모듈")

        # 마크다운 보고서
        print("\n" + reviewer.generate_report(result, format="markdown"))

    finally:
        os.unlink(temp_path)


def example_comparative_review():
    """비교 리뷰 예제"""
    print("\n🔄 코드 개선 전후 비교")
    print("-" * 50)

    # 개선 전 코드
    code_before = """
def find_max(lst):
    max_val = lst[0]
    for i in range(len(lst)):
        if lst[i] > max_val:
            max_val = lst[i]
    return max_val
"""

    # 개선 후 코드
    code_after = '''
def find_max(lst: list[float]) -> float:
    """리스트에서 최댓값을 찾습니다.
    
    Args:
        lst: 숫자 리스트
        
    Returns:
        최댓값
        
    Raises:
        ValueError: 빈 리스트인 경우
    """
    if not lst:
        raise ValueError("빈 리스트입니다")
    
    return max(lst)
'''

    reviewer = CodeReviewer()

    print("개선 전:")
    result_before = reviewer.review_code(code_before, "before.py")
    print(f"  점수: {result_before.score}/100")
    print(f"  이슈: {len(result_before.issues)}개")

    print("\n개선 후:")
    result_after = reviewer.review_code(code_after, "after.py")
    print(f"  점수: {result_after.score}/100")
    print(f"  이슈: {len(result_after.issues)}개")

    print(f"\n개선 효과: +{result_after.score - result_before.score}점")


def main():
    """코드 리뷰어 예제 메인 함수"""

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)

    print("🔍 AI 코드 리뷰어 예제")
    print("=" * 50)

    try:
        # 1. 간단한 리뷰
        example_simple_review()

        # 2. 파일 리뷰
        example_file_review()

        # 3. 비교 리뷰
        example_comparative_review()

        print("\n✅ 모든 코드 리뷰 예제 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
