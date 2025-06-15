"""
Chat 앱 URL 설정
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'conversations', views.ConversationViewSet)
router.register(r'image-analysis', views.ImageAnalysisViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('chat/stream/', views.ChatStreamView.as_view(), name='chat-stream'),
    path('analyze/text/', views.TextAnalysisView.as_view(), name='text-analysis'),
    path('stats/', views.StatsView.as_view(), name='stats'),
]