"""
Django 이미지 생성 예제 - to_django_file() 활용

이 파일은 Django 프로젝트에서 AI 이미지 생성 기능을 구현하는 예제입니다.
v0.9.0+에서 추가된 to_django_file() 메서드를 활용합니다.

예제 실행 중 오류가 발생하면 me@pyhub.kr로 문의 부탁드립니다.
"""

from django.db import models
from django.views import View
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib.auth.models import User
from pyhub.llm import OpenAILLM
import json
import uuid
from datetime import timezone


# 모델 정의 (models.py에 추가)
class GeneratedImage(models.Model):
    """AI로 생성된 이미지"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    prompt = models.TextField()
    revised_prompt = models.TextField(blank=True)
    image = models.ImageField(upload_to='generated/%Y/%m/%d/')
    size = models.CharField(max_length=20)
    quality = models.CharField(max_length=10, default='standard')
    style = models.CharField(max_length=10, default='natural')
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.prompt[:50]}... - {self.created_at}"


# 뷰 정의 (views.py에 추가)
@method_decorator(csrf_exempt, name='dispatch')
class GenerateImageView(View):
    """이미지 생성 뷰"""
    
    def post(self, request):
        """텍스트 프롬프트로 이미지 생성"""
        try:
            data = json.loads(request.body)
            prompt = data.get('prompt', '')
            size = data.get('size', '1024x1024')
            quality = data.get('quality', 'standard')
            style = data.get('style', 'natural')
            
            if not prompt:
                return JsonResponse({'error': '프롬프트가 필요합니다.'}, status=400)
            
            # DALL-E 3로 이미지 생성
            dalle = OpenAILLM(model="dall-e-3")
            reply = dalle.generate_image(
                prompt,
                size=size,
                quality=quality,
                style=style
            )
            
            # to_django_file()로 간단하게 저장
            generated = GeneratedImage.objects.create(
                user=request.user if request.user.is_authenticated else None,
                prompt=prompt,
                revised_prompt=reply.revised_prompt or prompt,
                image=reply.to_django_file(),  # 자동으로 고유 파일명 생성
                size=reply.size,
                quality=quality,
                style=style,
                metadata={
                    'model': 'dall-e-3',
                    'original_url': reply.url
                }
            )
            
            return JsonResponse({
                'id': generated.id,
                'url': generated.image.url,
                'prompt': generated.prompt,
                'revised_prompt': generated.revised_prompt,
                'size': generated.size,
                'created_at': generated.created_at.isoformat()
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class GenerateVariationView(View):
    """이미지 변형 생성 뷰"""
    
    def post(self, request):
        """업로드된 이미지를 분석하여 변형 생성"""
        uploaded_file = request.FILES.get('image')
        style = request.POST.get('style', 'artistic')
        
        if not uploaded_file:
            return JsonResponse({'error': '이미지가 필요합니다.'}, status=400)
        
        try:
            # 1. 업로드된 이미지 분석
            from pyhub.llm import LLM
            analyzer = LLM.create("gpt-4o-mini")
            
            # 임시 파일로 저장
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            # 이미지 분석하여 프롬프트 생성
            analysis_reply = analyzer.ask(
                f"이 이미지를 {style} 스타일로 재해석하기 위한 상세한 DALL-E 프롬프트를 만들어주세요",
                files=[tmp_path]
            )
            
            variation_prompt = analysis_reply.text
            
            # 2. 변형 이미지 생성
            dalle = OpenAILLM(model="dall-e-3")
            image_reply = dalle.generate_image(variation_prompt, quality="hd")
            
            # 3. 생성된 이미지 저장
            generated = GeneratedImage.objects.create(
                user=request.user if request.user.is_authenticated else None,
                prompt=f"[{style} variation] {variation_prompt}",
                revised_prompt=image_reply.revised_prompt or variation_prompt,
                image=image_reply.to_django_file(f'variation_{uuid.uuid4().hex[:8]}.png'),
                size=image_reply.size,
                quality="hd",
                style=style,
                metadata={
                    'original_filename': uploaded_file.name,
                    'variation_style': style
                }
            )
            
            # 임시 파일 삭제
            import os
            os.unlink(tmp_path)
            
            return JsonResponse({
                'id': generated.id,
                'url': generated.image.url,
                'prompt': generated.prompt,
                'style': style,
                'created_at': generated.created_at.isoformat()
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class DownloadGeneratedImageView(View):
    """생성된 이미지 다운로드"""
    
    def get(self, request, image_id):
        """이미지를 다운로드 가능한 형태로 반환"""
        try:
            generated = GeneratedImage.objects.get(id=image_id)
            
            # 이미지 파일 읽기
            with generated.image.open('rb') as f:
                response = HttpResponse(f.read(), content_type='image/png')
                response['Content-Disposition'] = f'attachment; filename="generated_{generated.id}.png"'
                return response
                
        except GeneratedImage.DoesNotExist:
            return JsonResponse({'error': '이미지를 찾을 수 없습니다.'}, status=404)


# URL 설정 (urls.py에 추가)
"""
from django.urls import path
from .views import GenerateImageView, GenerateVariationView, DownloadGeneratedImageView

urlpatterns = [
    # ... 기존 URL 패턴들 ...
    
    # 이미지 생성 관련
    path('api/generate-image/', GenerateImageView.as_view(), name='generate-image'),
    path('api/generate-variation/', GenerateVariationView.as_view(), name='generate-variation'),
    path('api/download-image/<int:image_id>/', DownloadGeneratedImageView.as_view(), name='download-image'),
]
"""


# 관리자 설정 (admin.py에 추가)
"""
from django.contrib import admin
from django.utils.html import format_html
from .models import GeneratedImage

@admin.register(GeneratedImage)
class GeneratedImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'prompt_preview', 'image_preview', 'size', 'quality', 'created_at']
    list_filter = ['created_at', 'size', 'quality', 'style']
    search_fields = ['prompt', 'revised_prompt']
    readonly_fields = ['image_preview_large', 'metadata_display']
    
    def prompt_preview(self, obj):
        return obj.prompt[:50] + '...' if len(obj.prompt) > 50 else obj.prompt
    prompt_preview.short_description = 'Prompt'
    
    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" width="100" height="100" style="object-fit: cover;"/>',
                obj.image.url
            )
        return '-'
    image_preview.short_description = 'Preview'
    
    def image_preview_large(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" width="400"/><br/><a href="{}" target="_blank">전체 크기로 보기</a>',
                obj.image.url, obj.image.url
            )
        return '-'
    image_preview_large.short_description = 'Image'
    
    def metadata_display(self, obj):
        import json
        return format_html(
            '<pre>{}</pre>',
            json.dumps(obj.metadata, indent=2, ensure_ascii=False)
        )
    metadata_display.short_description = 'Metadata'
"""


# 사용 예제 (JavaScript/Frontend)
"""
// 이미지 생성 요청
async function generateImage(prompt) {
    const response = await fetch('/api/generate-image/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            prompt: prompt,
            size: '1024x1792',  // 세로 형식
            quality: 'hd',      // 고품질
            style: 'natural'    // 자연스러운 스타일
        })
    });
    
    const data = await response.json();
    if (data.url) {
        // 이미지 표시
        const img = document.createElement('img');
        img.src = data.url;
        document.body.appendChild(img);
    }
}

// 이미지 변형 생성
async function generateVariation(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('style', 'artistic');
    
    const response = await fetch('/api/generate-variation/', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    console.log('Generated variation:', data);
}
"""