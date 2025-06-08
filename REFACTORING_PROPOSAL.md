# 이미지 처리 메서드 리팩토링 제안

## 현재 상황

현재 이미지를 처리하는 3가지 방법이 있습니다:

1. `ask(input="질문", files=["image.jpg"])` - 범용 메서드
2. `describe_images(DescribeImageRequest)` - 병렬 처리 특화
3. `describe_image("image.jpg", prompt="질문")` - 간편 메서드

## 문제점

1. **기능 중복**: `ask(files=...)` 와 `describe_image()`가 비슷한 기능
2. **일관성 부족**: 각 메서드가 다른 방식으로 파라미터를 받음
3. **혼란**: 사용자가 어떤 메서드를 사용해야 할지 불명확

## 리팩토링 제안

### 옵션 1: 통합 접근법 (권장)

```python
# describe_image를 ask의 별칭으로 만들기
def describe_image(self, image, prompt="Describe this image in detail.", **kwargs):
    """이미지 설명을 위한 편의 메서드 (ask의 래퍼)"""
    return self.ask(
        input=prompt,
        files=[image],
        **kwargs
    )
```

**장점**:
- 코드 중복 제거
- 일관된 동작 보장
- ask의 모든 기능 활용 가능 (히스토리, 스트리밍 등)

**단점**:
- 현재 describe_images와의 연결 끊김

### 옵션 2: 역할 명확화

각 메서드의 역할을 명확히 구분:

- `ask(files=...)`: 텍스트와 이미지를 함께 사용하는 대화형 작업
- `describe_image()`: 단일 이미지 분석 전용 (히스토리 없음)
- `describe_images()`: 대량 이미지 병렬 처리 전용

```python
def describe_image(self, image, prompt="...", **kwargs):
    """단일 이미지 분석 전용 메서드"""
    # 히스토리 사용 안 함, 이미지 분석에 최적화
    return self.ask(
        input=prompt,
        files=[image],
        use_history=False,  # 항상 False
        save_history=False,  # 항상 False
        **kwargs
    )
```

### 옵션 3: describe_images 제거

`describe_images`의 병렬 처리 기능을 `ask`에 통합:

```python
# ask 메서드에 병렬 처리 옵션 추가
def ask(self, input, files=None, parallel=False, max_workers=4, ...):
    if files and len(files) > 1 and parallel:
        # 병렬 처리 로직
        ...
```

## 권장 사항

**단기적 해결책**: 옵션 1 채택
- `describe_image`를 `ask`의 간단한 래퍼로 만들기
- 문서에 각 메서드의 사용 시나리오 명확히 기술

**장기적 해결책**: 
- 사용 패턴 분석 후 불필요한 메서드 제거
- API 2.0에서 통합된 인터페이스 제공

## 구현 예시

```python
class BaseLLM:
    def describe_image(self, image, prompt="Describe this image in detail.", **kwargs):
        """
        이미지를 설명하는 편의 메서드.
        
        내부적으로 ask()를 호출하므로 모든 ask() 옵션 사용 가능.
        대량 이미지 병렬 처리가 필요한 경우 describe_images() 사용.
        
        Examples:
            # 기본 사용
            response = llm.describe_image("photo.jpg")
            
            # ask()와 동일한 결과
            response = llm.ask("Describe this image in detail.", files=["photo.jpg"])
        """
        return self.ask(input=prompt, files=[image], **kwargs)
```