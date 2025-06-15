#!/usr/bin/env python3
"""Django 관리 스크립트"""
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chatbot.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django를 import할 수 없습니다. 가상환경이 활성화되어 있고 " "Django가 설치되어 있는지 확인하세요."
        ) from exc
    execute_from_command_line(sys.argv)
