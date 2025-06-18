"""
Django에서 ImageReply 활용 예제
================================

이 예제는 Django 애플리케이션에서 pyhub-llm의 이미지 생성 기능을 
활용하는 다양한 방법을 보여줍니다.

필요한 패키지:
- pip install "pyhub-llm[openai]"
- pip install django pillow

주요 기능:
- 텍스트 프롬프트로 이미지 생성
- 생성된 이미지를 다양한 방식으로 저장
- Django ORM과 통합
- 비동기 처리
- 이미지 분석 후 재생성
"""

# ===== models.py =====
from django.db import models
from django.contrib.auth.models import User

class GeneratedImage(models.Model):
    """생성된 이미지를 저장하는 모델"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    prompt = models.TextField(help_text="사용자가 입력한 원본 프롬프트")
    revised_prompt = models.TextField(blank=True, help_text="DALL-E가 개선한 프롬프트")
    image = models.ImageField(upload_to='generated/%Y/%m/%d/')
    image_url = models.URLField(blank=True, null=True, help_text="OpenAI에서 제공한 원본 URL")
    size = models.CharField(max_length=20, default='1024x1024')
    quality = models.CharField(max_length=20, default='standard')
    style = models.CharField(max_length=20, default='natural')
    created_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['user', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.prompt[:50]}... ({self.created_at.strftime('%Y-%m-%d %H:%M')})"


# ===== views.py =====
from django.http import HttpResponse, JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.core.files.base import ContentFile
from django.shortcuts import render, get_object_or_404
from pyhub.llm import OpenAILLM, LLM
from io import BytesIO
import json
import uuid
import base64
from PIL import Image as PILImage


@method_decorator(csrf_exempt, name='dispatch')
class GenerateImageAPIView(View):
    """이미지 생성 API 엔드포인트"""
    
    def post(self, request):
        """텍스트 프롬프트로 이미지 생성"""
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': '잘못된 JSON 형식입니다.'}, status=400)
        
        prompt = data.get('prompt', '').strip()
        size = data.get('size', '1024x1024')
        quality = data.get('quality', 'standard')
        style = data.get('style', 'natural')
        save_to_db = data.get('save_to_db', False)
        
        # 프롬프트 검증
        if not prompt:
            return JsonResponse({'error': '프롬프트가 필요합니다.'}, status=400)
        
        if len(prompt) > 1000:
            return JsonResponse({'error': '프롬프트는 1000자 이하여야 합니다.'}, status=400)
        
        # 크기 검증
        valid_sizes = ['1024x1024', '1024x1792', '1792x1024']
        if size not in valid_sizes:
            return JsonResponse({
                'error': f'유효하지 않은 크기입니다. 가능한 크기: {", ".join(valid_sizes)}'
            }, status=400)
        
        try:
            # DALL-E 3 모델로 이미지 생성
            dalle = OpenAILLM(model="dall-e-3")
            reply = dalle.generate_image(
                prompt,
                size=size,
                quality=quality,
                style=style
            )
            
            response_data = {
                'url': reply.url,
                'prompt': prompt,
                'revised_prompt': reply.revised_prompt,
                'size': reply.size,
                'model': reply.model
            }
            
            # DB에 저장 옵션
            if save_to_db and request.user.is_authenticated:
                # BytesIO에 이미지 저장
                buffer = BytesIO()
                reply.save(buffer)
                buffer.seek(0)
                
                # Django 모델에 저장
                generated = GeneratedImage.objects.create(
                    user=request.user,
                    prompt=prompt,
                    revised_prompt=reply.revised_prompt or prompt,
                    image_url=reply.url,
                    size=size,
                    quality=quality,
                    style=style,
                    metadata={
                        'model': reply.model,
                        'created_via': 'api'
                    }
                )
                
                # 이미지 파일 저장
                generated.image.save(
                    f'dalle_{uuid.uuid4().hex[:8]}.png',
                    ContentFile(buffer.getvalue()),
                    save=True
                )
                
                response_data['saved_id'] = generated.id
                response_data['saved_url'] = generated.image.url
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({'error': f'이미지 생성 실패: {str(e)}'}, status=500)


class ImageDownloadView(View):
    """생성된 이미지를 직접 다운로드"""
    
    def get(self, request, image_id):
        """저장된 이미지 다운로드"""
        image = get_object_or_404(GeneratedImage, id=image_id)
        
        # 권한 체크
        if image.user and image.user != request.user:
            return JsonResponse({'error': '권한이 없습니다.'}, status=403)
        
        response = HttpResponse(image.image, content_type='image/png')
        response['Content-Disposition'] = f'attachment; filename="generated_{image.id}.png"'
        return response
    
    def post(self, request):
        """새로 생성하여 바로 다운로드"""
        data = json.loads(request.body)
        prompt = data.get('prompt', '')
        
        if not prompt:
            return JsonResponse({'error': '프롬프트가 필요합니다.'}, status=400)
        
        try:
            # 이미지 생성
            dalle = OpenAILLM(model="dall-e-3")
            reply = dalle.generate_image(prompt, quality="hd")
            
            # HttpResponse에 직접 저장
            response = HttpResponse(content_type='image/png')
            response['Content-Disposition'] = f'attachment; filename="dalle_{uuid.uuid4().hex[:8]}.png"'
            reply.save(response)
            
            return response
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class BatchGenerateView(View):
    """여러 이미지 일괄 생성"""
    
    def post(self, request):
        data = json.loads(request.body)
        prompts = data.get('prompts', [])
        
        if not prompts or not isinstance(prompts, list):
            return JsonResponse({'error': '프롬프트 리스트가 필요합니다.'}, status=400)
        
        if len(prompts) > 5:
            return JsonResponse({'error': '한 번에 최대 5개까지 생성 가능합니다.'}, status=400)
        
        results = []
        dalle = OpenAILLM(model="dall-e-3")
        
        for prompt in prompts:
            try:
                reply = dalle.generate_image(prompt)
                
                # Base64로 인코딩하여 반환
                buffer = BytesIO()
                reply.save(buffer)
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode()
                
                results.append({
                    'prompt': prompt,
                    'success': True,
                    'data': f'data:image/png;base64,{image_data}',
                    'url': reply.url,
                    'revised_prompt': reply.revised_prompt
                })
            except Exception as e:
                results.append({
                    'prompt': prompt,
                    'success': False,
                    'error': str(e)
                })
        
        return JsonResponse({'results': results})


class ImageVariationView(View):
    """업로드된 이미지를 분석하고 변형 생성"""
    
    def post(self, request):
        uploaded_file = request.FILES.get('image')
        variation_type = request.POST.get('type', 'similar')  # similar, style_transfer, enhance
        
        if not uploaded_file:
            return JsonResponse({'error': '이미지 파일이 필요합니다.'}, status=400)
        
        try:
            # 이미지를 임시 저장
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                tmp.flush()  # 파일 시스템에 내용이 기록되도록 보장
                tmp_path = tmp.name
            
            # GPT-4V로 이미지 분석
            analyzer = LLM.create("gpt-4o-mini")
            
            # 변형 타입에 따른 프롬프트
            if variation_type == 'similar':
                analysis_prompt = """이 이미지를 분석하고 유사한 이미지를 생성하기 위한 
                                 상세한 DALL-E 프롬프트를 만들어주세요."""
            elif variation_type == 'style_transfer':
                analysis_prompt = """이 이미지의 내용은 유지하되 다른 예술 스타일로 
                                 변형하기 위한 DALL-E 프롬프트를 만들어주세요."""
            else:  # enhance
                analysis_prompt = """이 이미지를 더 고품질로 개선하거나 향상시키기 위한 
                                 DALL-E 프롬프트를 만들어주세요."""
            
            # 구조화된 응답으로 분석
            from pydantic import BaseModel, Field
            from typing import List
            
            class ImageAnalysis(BaseModel):
                description: str = Field(description="이미지 전체 설명")
                main_subject: str = Field(description="주요 피사체")
                style: str = Field(description="예술적 스타일")
                colors: List[str] = Field(description="주요 색상")
                mood: str = Field(description="분위기")
                dalle_prompt: str = Field(description="DALL-E 생성을 위한 프롬프트")
            
            analysis_reply = analyzer.ask(
                analysis_prompt,
                files=[tmp_path],
                schema=ImageAnalysis
            )
            
            analysis = analysis_reply.structured_data
            
            # DALL-E로 새 이미지 생성
            dalle = OpenAILLM(model="dall-e-3")
            new_image = dalle.generate_image(
                analysis.dalle_prompt,
                quality="hd"
            )
            
            # 결과를 Base64로 인코딩
            buffer = BytesIO()
            new_image.save(buffer)
            buffer.seek(0)
            new_image_data = base64.b64encode(buffer.getvalue()).decode()
            
            # 원본 이미지도 Base64로 인코딩 (비교용)
            with open(tmp_path, 'rb') as f:
                original_data = base64.b64encode(f.read()).decode()
            
            return JsonResponse({
                'analysis': {
                    'description': analysis.description,
                    'main_subject': analysis.main_subject,
                    'style': analysis.style,
                    'colors': analysis.colors,
                    'mood': analysis.mood
                },
                'original_image': f'data:image/png;base64,{original_data}',
                'generated_image': f'data:image/png;base64,{new_image_data}',
                'generated_url': new_image.url,
                'dalle_prompt': analysis.dalle_prompt,
                'revised_prompt': new_image.revised_prompt
            })
            
        except Exception as e:
            return JsonResponse({'error': f'처리 실패: {str(e)}'}, status=500)
        finally:
            # 임시 파일 정리
            import os
            if 'tmp_path' in locals():
                os.unlink(tmp_path)


# ===== 템플릿 뷰 =====
def image_generator_view(request):
    """이미지 생성기 웹 페이지"""
    user_images = []
    if request.user.is_authenticated:
        user_images = GeneratedImage.objects.filter(
            user=request.user
        )[:10]
    
    return render(request, 'image_generator.html', {
        'user_images': user_images,
        'sizes': ['1024x1024', '1024x1792', '1792x1024'],
        'qualities': ['standard', 'hd'],
        'styles': ['natural', 'vivid']
    })


# ===== Django 관리자 커스터마이징 (admin.py) =====
from django.contrib import admin
from django.utils.html import format_html

@admin.register(GeneratedImage)
class GeneratedImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'prompt_preview', 'image_preview', 'size', 'created_at']
    list_filter = ['size', 'quality', 'style', 'created_at']
    search_fields = ['prompt', 'revised_prompt']
    readonly_fields = ['image_preview_large', 'revised_prompt', 'created_at']
    
    def prompt_preview(self, obj):
        return obj.prompt[:50] + '...' if len(obj.prompt) > 50 else obj.prompt
    prompt_preview.short_description = '프롬프트'
    
    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" width="100" height="100" style="object-fit: cover;" />',
                obj.image.url
            )
        return '-'
    image_preview.short_description = '미리보기'
    
    def image_preview_large(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-width: 500px; max-height: 500px;" />',
                obj.image.url
            )
        return '-'
    image_preview_large.short_description = '이미지'


# ===== urls.py =====
from django.urls import path
from . import views

app_name = 'image_gen'

urlpatterns = [
    # API 엔드포인트
    path('api/generate/', GenerateImageAPIView.as_view(), name='generate'),
    path('api/download/', ImageDownloadView.as_view(), name='download'),
    path('api/download/<int:image_id>/', ImageDownloadView.as_view(), name='download_saved'),
    path('api/batch/', BatchGenerateView.as_view(), name='batch'),
    path('api/variation/', ImageVariationView.as_view(), name='variation'),
    
    # 웹 페이지
    path('', views.image_generator_view, name='generator'),
]


# ===== 사용 예시 (JavaScript) =====
"""
// 이미지 생성 요청
async function generateImage(prompt) {
    const response = await fetch('/image-gen/api/generate/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            prompt: prompt,
            size: '1024x1024',
            quality: 'hd',
            save_to_db: true
        })
    });
    
    const data = await response.json();
    if (data.url) {
        // 이미지 표시
        document.getElementById('result').innerHTML = 
            `<img src="${data.url}" alt="${data.prompt}">`;
    }
}

// 일괄 생성
async function batchGenerate(prompts) {
    const response = await fetch('/image-gen/api/batch/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompts })
    });
    
    const data = await response.json();
    data.results.forEach(result => {
        if (result.success) {
            // Base64 이미지 표시
            const img = document.createElement('img');
            img.src = result.data;
            document.getElementById('gallery').appendChild(img);
        }
    });
}

// 이미지 변형
async function createVariation(file) {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('type', 'style_transfer');
    
    const response = await fetch('/image-gen/api/variation/', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    // 원본과 생성된 이미지 비교 표시
    document.getElementById('comparison').innerHTML = `
        <div style="display: flex; gap: 20px;">
            <div>
                <h3>원본</h3>
                <img src="${data.original_image}" width="300">
            </div>
            <div>
                <h3>변형</h3>
                <img src="${data.generated_image}" width="300">
                <p>${data.analysis.description}</p>
            </div>
        </div>
    `;
}
"""


# ===== 캐싱 및 최적화 (선택사항) =====
from django.core.cache import cache
from django.views.decorators.cache import cache_page

class CachedImageGenerationView(View):
    """캐싱이 적용된 이미지 생성"""
    
    def post(self, request):
        data = json.loads(request.body)
        prompt = data.get('prompt', '')
        
        # 프롬프트 기반 캐시 키 생성 (안정적인 해시 사용)
        import hashlib
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        cache_key = f"image_gen:{prompt_hash}"
        cached_url = cache.get(cache_key)
        
        if cached_url:
            return JsonResponse({
                'url': cached_url,
                'cached': True
            })
        
        # 새로 생성
        dalle = OpenAILLM(model="dall-e-3")
        reply = dalle.generate_image(prompt)
        
        # 캐시에 저장 (1시간)
        cache.set(cache_key, reply.url, 3600)
        
        return JsonResponse({
            'url': reply.url,
            'cached': False
        })


# ===== 비동기 뷰 (Django 4.1+) =====
import asyncio
from django.http import JsonResponse

class AsyncBatchGenerateView(View):
    """비동기 일괄 이미지 생성"""
    
    async def post(self, request):
        data = json.loads(request.body)
        prompts = data.get('prompts', [])[:5]  # 최대 5개
        
        dalle = OpenAILLM(model="dall-e-3")
        
        # 동시에 여러 이미지 생성
        tasks = [
            dalle.generate_image_async(prompt) 
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        response_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                response_data.append({
                    'prompt': prompts[i],
                    'error': str(result)
                })
            else:
                response_data.append({
                    'prompt': prompts[i],
                    'url': result.url,
                    'revised_prompt': result.revised_prompt
                })
        
        return JsonResponse({'results': response_data})