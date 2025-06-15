"""
Django 모델

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

import uuid

from django.contrib.auth.models import User
from django.db import models


class Conversation(models.Model):
    """대화 세션"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title or 'Untitled'} - {self.created_at}"


class Message(models.Model):
    """대화 메시지"""

    ROLE_CHOICES = [
        ("user", "사용자"),
        ("assistant", "AI"),
        ("system", "시스템"),
    ]

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    tokens_used = models.IntegerField(default=0)

    # 도구 호출 정보 (JSON)
    tool_calls = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class ChatHistory(models.Model):
    """채팅 통계 및 히스토리"""

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    total_messages = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    model_used = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "Chat histories"


class ImageAnalysis(models.Model):
    """이미지 분석 기록"""

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to="images/%Y/%m/%d/")
    question = models.TextField()
    analysis = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "Image analyses"
