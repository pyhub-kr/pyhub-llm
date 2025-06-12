"""Optional imports 테스트

특정 provider가 설치되지 않은 상황에서도 기본 import가 작동하는지 확인
"""

import sys
from unittest.mock import patch



class TestOptionalImports:
    """Optional dependency import 테스트"""

    def test_import_without_anthropic(self):
        """anthropic이 설치되지 않은 상황에서 import 테스트"""
        # anthropic 모듈을 임시로 제거
        with patch.dict(sys.modules, {"anthropic": None, "anthropic.types": None}):
            # types 모듈을 다시 로드
            if "pyhub.llm.types" in sys.modules:
                del sys.modules["pyhub.llm.types"]

            # import가 실패하지 않아야 함
            # AnthropicChatModelType이 Union[Literal[...], str]로 대체되었는지 확인
            # 타입 체크를 위해 get_origin과 get_args 사용
            from typing import Union, get_args, get_origin

            from pyhub.llm import types

            assert get_origin(types.AnthropicChatModelType) == Union
            args = get_args(types.AnthropicChatModelType)
            assert str in args  # str이 포함되어 있어야 함

    def test_import_without_openai(self):
        """openai가 설치되지 않은 상황에서 import 테스트"""
        with patch.dict(sys.modules, {"openai": None, "openai.types": None}):
            if "pyhub.llm.types" in sys.modules:
                del sys.modules["pyhub.llm.types"]

            # _OpenAIChatModel이 Literal 타입으로 대체되었는지 확인
            from typing import Literal, get_origin

            from pyhub.llm import types

            assert get_origin(types._OpenAIChatModel) == Literal

    def test_llm_import_without_anthropic(self):
        """anthropic 없이 LLM import 테스트"""
        with patch.dict(sys.modules, {"anthropic": None, "anthropic.types": None}):
            # 관련 모듈들 제거
            modules_to_remove = ["pyhub.llm", "pyhub.llm.types", "pyhub.llm.base", "pyhub.llm.__init__"]
            for module in modules_to_remove:
                if module in sys.modules:
                    del sys.modules[module]

            # LLM import가 실패하지 않아야 함
            from pyhub.llm import LLM

            # OpenAI 모델로 인스턴스 생성 가능해야 함
            # (실제 API 호출은 하지 않음)
            llm = LLM.create("gpt-4o-mini")
            assert llm is not None
