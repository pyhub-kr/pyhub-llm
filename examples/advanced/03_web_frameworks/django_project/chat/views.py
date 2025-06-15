"""
Django 뷰
"""
import json
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.core.cache import cache
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

from pyhub.llm import LLM
from pyhub.llm.types import Message as LLMMessage
from .models import Conversation, Message, ChatHistory, ImageAnalysis
from .serializers import (
    ConversationSerializer, MessageSerializer, 
    ChatHistorySerializer, ImageAnalysisSerializer
)


# LLM 인스턴스 캐시
_llm_cache = {}


def get_llm(model="gpt-4o-mini", **kwargs):
    """LLM 인스턴스 가져오기"""
    cache_key = f"{model}_{hash(frozenset(kwargs.items()))}"
    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = LLM.create(model, **kwargs)
    return _llm_cache[cache_key]


class ConversationViewSet(viewsets.ModelViewSet):
    """대화 관리 ViewSet"""
    queryset = Conversation.objects.all()
    serializer_class = ConversationSerializer
    
    def get_queryset(self):
        """사용자별 대화 필터링"""
        if self.request.user.is_authenticated:
            return self.queryset.filter(user=self.request.user)
        # 비로그인 사용자는 세션 기반으로 처리 가능
        return self.queryset.none()
    
    @action(detail=True, methods=['post'])
    def send_message(self, request, pk=None):
        """메시지 전송"""
        conversation = self.get_object()
        message_content = request.data.get('message', '')
        model = request.data.get('model', 'gpt-4o-mini')
        
        if not message_content:
            return Response({'error': '메시지가 필요합니다.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # 사용자 메시지 저장
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=message_content
        )
        
        # 대화 컨텍스트 구성
        messages = conversation.messages.all()
        context_messages = []
        for msg in messages[-10:]:  # 최근 10개 메시지
            context_messages.append(LLMMessage(role=msg.role, content=msg.content))
        
        try:
            # LLM 호출
            llm = get_llm(model)
            
            # 간단한 컨텍스트 방식 사용
            context = "\n".join([f"{msg.role}: {msg.content}" for msg in context_messages])
            reply = llm.ask(context)
            
            # AI 응답 저장
            ai_message = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=reply.text,
                tokens_used=reply.usage.total if reply.usage else 0
            )
            
            # 대화 업데이트
            conversation.save()
            
            # 통계 업데이트
            ChatHistory.objects.create(
                user=request.user if request.user.is_authenticated else None,
                conversation=conversation,
                total_messages=messages.count(),
                total_tokens=sum(m.tokens_used for m in messages),
                model_used=model
            )
            
            return Response({
                'user_message': MessageSerializer(user_message).data,
                'ai_message': MessageSerializer(ai_message).data
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """대화 메시지 목록"""
        conversation = self.get_object()
        messages = conversation.messages.all()
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)


@method_decorator(csrf_exempt, name='dispatch')
class ChatStreamView(View):
    """스트리밍 채팅 뷰"""
    
    def post(self, request):
        """스트리밍 응답"""
        data = json.loads(request.body)
        message = data.get('message', '')
        conversation_id = data.get('conversation_id')
        model = data.get('model', 'gpt-4o-mini')
        
        if not message:
            return JsonResponse({'error': '메시지가 필요합니다.'}, status=400)
        
        def generate():
            """스트리밍 생성기"""
            try:
                # 대화 가져오기
                if conversation_id:
                    conversation = Conversation.objects.get(id=conversation_id)
                else:
                    conversation = Conversation.objects.create()
                
                # 사용자 메시지 저장
                Message.objects.create(
                    conversation=conversation,
                    role='user',
                    content=message
                )
                
                # LLM 스트리밍
                llm = get_llm(model)
                full_response = ""
                
                for chunk in llm.ask(message, stream=True):
                    full_response += chunk.text
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                
                # AI 응답 저장
                Message.objects.create(
                    conversation=conversation,
                    role='assistant',
                    content=full_response
                )
                
                yield f"data: {json.dumps({'done': True, 'conversation_id': str(conversation.id)})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingHttpResponse(
            generate(),
            content_type='text/event-stream'
        )


class TextAnalysisView(View):
    """텍스트 분석 뷰"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request):
        """텍스트 분석"""
        data = json.loads(request.body)
        text = data.get('text', '')
        analysis_type = data.get('analysis_type', 'all')
        
        if not text:
            return JsonResponse({'error': '텍스트가 필요합니다.'}, status=400)
        
        # 캐시 확인
        cache_key = f"analysis_{hash(text)}_{analysis_type}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return JsonResponse(cached_result)
        
        try:
            llm = get_llm("gpt-4o-mini")
            result = {}
            
            if analysis_type in ['sentiment', 'all']:
                sentiment_reply = llm.ask(
                    f"다음 텍스트의 감정을 긍정/중립/부정 중 하나로 분류하세요: {text}",
                    choices=["긍정", "중립", "부정"]
                )
                result['sentiment'] = sentiment_reply.choice
            
            if analysis_type in ['summary', 'all']:
                summary_reply = llm.ask(f"다음 텍스트를 한 문장으로 요약하세요: {text}")
                result['summary'] = summary_reply.text
            
            if analysis_type in ['keywords', 'all']:
                keywords_reply = llm.ask(f"다음 텍스트의 핵심 키워드 3개를 쉼표로 구분하여 나열하세요: {text}")
                result['keywords'] = [k.strip() for k in keywords_reply.text.split(',')][:3]
            
            # 캐시 저장 (1시간)
            cache.set(cache_key, result, 3600)
            
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class ImageAnalysisViewSet(viewsets.ModelViewSet):
    """이미지 분석 ViewSet"""
    queryset = ImageAnalysis.objects.all()
    serializer_class = ImageAnalysisSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    def create(self, request, *args, **kwargs):
        """이미지 업로드 및 분석"""
        image = request.FILES.get('image')
        question = request.data.get('question', '이 이미지를 설명해주세요.')
        
        if not image:
            return Response({'error': '이미지가 필요합니다.'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # 이미지 저장
            image_analysis = ImageAnalysis.objects.create(
                user=request.user if request.user.is_authenticated else None,
                image=image,
                question=question
            )
            
            # LLM으로 이미지 분석
            llm = get_llm("gpt-4o-mini")
            reply = llm.ask(question, files=[image_analysis.image.path])
            
            # 분석 결과 저장
            image_analysis.analysis = reply.text
            image_analysis.save()
            
            serializer = self.get_serializer(image_analysis)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# 관리 명령어 뷰
class StatsView(View):
    """통계 뷰"""
    
    def get(self, request):
        """채팅 통계"""
        from django.db.models import Sum, Count
        
        stats = ChatHistory.objects.aggregate(
            total_conversations=Count('conversation', distinct=True),
            total_messages=Sum('total_messages'),
            total_tokens=Sum('total_tokens')
        )
        
        # 모델별 사용량
        model_usage = ChatHistory.objects.values('model_used').annotate(
            count=Count('id'),
            tokens=Sum('total_tokens')
        ).order_by('-count')
        
        return JsonResponse({
            'stats': stats,
            'model_usage': list(model_usage),
            'active_conversations': Conversation.objects.filter(is_active=True).count(),
            'total_users': User.objects.count()
        })