# 🐍 Python 기초 확인하기

AI 프로그래밍을 시작하기 전에 Python이 제대로 설치되어 있는지 확인해봅시다.

## 🔍 Python 설치 확인

### 1. 터미널(명령 프롬프트) 열기
- **Windows**: Win + R → `cmd` 입력
- **Mac**: Spotlight(Cmd + Space) → `Terminal` 입력
- **Linux**: Ctrl + Alt + T

### 2. Python 버전 확인
```bash
python --version
```
또는
```bash
python3 --version
```

### 3. 결과 확인
```
Python 3.8.10
```
**3.8 이상**이면 OK! 👍

## 📥 Python이 없다면?

[python.org](https://www.python.org/downloads/)에서 다운로드하세요.

### 설치 시 주의사항
- ✅ "Add Python to PATH" 체크하기
- ✅ pip 포함 설치 선택하기

## 🧰 pip 확인하기

pip는 Python 패키지를 설치하는 도구입니다.

```bash
pip --version
```
또는
```bash
pip3 --version
```

결과:
```
pip 21.0.1 from ... (python 3.8)
```

## 📝 꼭 알아야 할 Python 기초

### 1. 변수와 출력
```python
# 변수에 값 저장하기
name = "김철수"
age = 25

# 출력하기
print(f"안녕하세요, {name}님! 나이는 {age}살이시네요.")
```

### 2. 함수 만들기
```python
# 함수 정의
def greet(name):
    """인사하는 함수"""
    return f"안녕하세요, {name}님!"

# 함수 사용
message = greet("영희")
print(message)  # 출력: 안녕하세요, 영희님!
```

### 3. 조건문
```python
age = 20

if age >= 20:
    print("성인입니다")
else:
    print("미성년자입니다")
```

### 4. 반복문
```python
# 리스트 만들기
fruits = ["사과", "바나나", "오렌지"]

# 반복하기
for fruit in fruits:
    print(f"나는 {fruit}를 좋아해요")
```

### 5. 딕셔너리 (중요!)
```python
# 딕셔너리는 키-값 쌍으로 데이터를 저장합니다
person = {
    "name": "김철수",
    "age": 25,
    "city": "서울"
}

# 값 가져오기
print(person["name"])  # 출력: 김철수

# 값 바꾸기
person["age"] = 26
```

### 6. 예외 처리
```python
try:
    # 뭔가 시도
    number = int("abc")  # 에러 발생!
except ValueError:
    # 에러가 발생하면 여기 실행
    print("숫자가 아니에요!")
```

## 🎮 연습 문제

다음 코드를 이해할 수 있다면 준비 완료!

```python
# AI 도우미에게 물어볼 질문들
questions = [
    "오늘 날씨 어때?",
    "파이썬이 뭐야?",
    "점심 메뉴 추천해줘"
]

# 각 질문을 출력하기
for i, question in enumerate(questions, 1):
    print(f"{i}. {question}")

# 사용자 선택 받기
try:
    choice = int(input("몇 번 질문을 선택하시겠어요? "))
    if 1 <= choice <= len(questions):
        selected = questions[choice - 1]
        print(f"선택한 질문: {selected}")
    else:
        print("잘못된 번호입니다!")
except ValueError:
    print("숫자를 입력해주세요!")
```

## ✅ 체크포인트

다음을 할 수 있다면 다음 단계로 진행하세요:
- [ ] Python 실행 가능
- [ ] pip로 패키지 설치 가능
- [ ] 위의 예제 코드 이해 가능

## 🚀 다음 단계

Python 준비가 끝났다면 [API 키 알아보기](api-keys-explained.md)로 이동하세요!