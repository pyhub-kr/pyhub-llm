"""
Django REST Framework 시리얼라이저
"""

from rest_framework import serializers

from .models import ChatHistory, Conversation, ImageAnalysis, Message


class MessageSerializer(serializers.ModelSerializer):
    """메시지 시리얼라이저"""

    class Meta:
        model = Message
        fields = ["id", "role", "content", "created_at", "tokens_used", "tool_calls"]
        read_only_fields = ["created_at"]


class ConversationSerializer(serializers.ModelSerializer):
    """대화 시리얼라이저"""

    messages = MessageSerializer(many=True, read_only=True)
    message_count = serializers.IntegerField(source="messages.count", read_only=True)

    class Meta:
        model = Conversation
        fields = ["id", "title", "created_at", "updated_at", "is_active", "messages", "message_count"]
        read_only_fields = ["created_at", "updated_at"]


class ChatHistorySerializer(serializers.ModelSerializer):
    """채팅 히스토리 시리얼라이저"""

    conversation_title = serializers.CharField(source="conversation.title", read_only=True)

    class Meta:
        model = ChatHistory
        fields = [
            "id",
            "conversation",
            "conversation_title",
            "total_messages",
            "total_tokens",
            "model_used",
            "created_at",
        ]
        read_only_fields = ["created_at"]


class ImageAnalysisSerializer(serializers.ModelSerializer):
    """이미지 분석 시리얼라이저"""

    image_url = serializers.ImageField(source="image", read_only=True)

    class Meta:
        model = ImageAnalysis
        fields = ["id", "image", "image_url", "question", "analysis", "created_at"]
        read_only_fields = ["analysis", "created_at"]
