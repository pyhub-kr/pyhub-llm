# 🍳 AI 레시피 도우미

냉장고에 있는 재료로 무엇을 만들 수 있을까? AI가 도와드립니다!

## 🎯 프로젝트 소개

이 프로젝트는 다음과 같은 기능을 제공합니다:
- 🥗 보유한 재료로 요리 추천
- 📖 상세한 레시피 제공
- 🛒 부족한 재료 쇼핑 리스트
- 📊 영양 정보 분석
- 💾 좋아하는 레시피 저장

## 🚀 빠른 시작

```python
# 레시피 도우미 실행
helper = RecipeHelper()

# 재료로 요리 추천
my_ingredients = ["닭가슴살", "양파", "토마토", "치즈"]
recipe = helper.suggest_recipe(my_ingredients)
print(recipe)
```

## 📝 전체 코드

```python
# recipe_helper.py
from pyhub.llm import LLM
import json
from datetime import datetime
from typing import List, Dict

class RecipeHelper:
    """AI 기반 레시피 도우미"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.ai = LLM.create(model)
        self.saved_recipes = []  # 저장된 레시피
        self.shopping_list = []  # 쇼핑 리스트
        
    def suggest_recipe(self, ingredients: List[str], 
                      dietary_restrictions: str = None,
                      cuisine_type: str = None) -> str:
        """재료를 기반으로 레시피를 추천합니다"""
        
        # 프롬프트 구성
        prompt = f"""
        다음 재료들로 만들 수 있는 요리를 추천해주세요:
        재료: {', '.join(ingredients)}
        """
        
        # 식이 제한 추가
        if dietary_restrictions:
            prompt += f"\n식이 제한: {dietary_restrictions}"
        
        # 요리 종류 추가
        if cuisine_type:
            prompt += f"\n선호 요리: {cuisine_type}"
        
        prompt += """
        
        다음 형식으로 답변해주세요:
        
        🍽️ 요리명: [요리 이름]
        ⏱️ 조리 시간: [예상 시간]
        👥 인분: [몇 인분]
        
        📝 필요한 재료:
        - [재료1] - [양]
        - [재료2] - [양]
        
        👨‍🍳 조리 방법:
        1. [단계 1]
        2. [단계 2]
        ...
        
        💡 요리 팁:
        [유용한 팁]
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def analyze_nutrition(self, recipe_text: str) -> Dict:
        """레시피의 영양 정보를 분석합니다"""
        
        prompt = f"""
        다음 레시피의 대략적인 영양 정보를 분석해주세요:
        
        {recipe_text}
        
        1인분 기준으로 다음 정보를 추정해주세요:
        - 칼로리 (kcal)
        - 단백질 (g)
        - 탄수화물 (g)
        - 지방 (g)
        - 나트륨 (mg)
        
        JSON 형식으로만 답해주세요.
        """
        
        response = self.ai.ask(prompt)
        
        # JSON 파싱 시도
        try:
            # AI 응답에서 JSON 부분만 추출
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                nutrition = json.loads(json_match.group())
                return nutrition
        except:
            pass
        
        # 파싱 실패 시 기본값
        return {
            "calories": "분석 중",
            "protein": "분석 중",
            "carbs": "분석 중",
            "fat": "분석 중",
            "sodium": "분석 중"
        }
    
    def create_shopping_list(self, recipe_text: str, 
                           available_ingredients: List[str]) -> List[str]:
        """레시피에 필요한 재료 중 없는 것들의 쇼핑 리스트를 만듭니다"""
        
        prompt = f"""
        레시피:
        {recipe_text}
        
        현재 가지고 있는 재료:
        {', '.join(available_ingredients)}
        
        위 레시피를 만들기 위해 추가로 구매해야 할 재료들을 리스트로 만들어주세요.
        각 재료와 필요한 양을 함께 적어주세요.
        
        형식:
        - [재료명] ([필요량])
        """
        
        response = self.ai.ask(prompt)
        
        # 쇼핑 리스트 파싱
        shopping_items = []
        lines = response.text.split('\n')
        for line in lines:
            if line.strip().startswith('-'):
                item = line.strip()[1:].strip()
                shopping_items.append(item)
        
        self.shopping_list.extend(shopping_items)
        return shopping_items
    
    def modify_recipe(self, recipe_text: str, modification: str) -> str:
        """레시피를 수정합니다 (인분 조절, 재료 대체 등)"""
        
        prompt = f"""
        원본 레시피:
        {recipe_text}
        
        수정 요청: {modification}
        
        요청에 따라 레시피를 수정해주세요.
        원본과 같은 형식을 유지해주세요.
        """
        
        response = self.ai.ask(prompt)
        return response.text
    
    def save_recipe(self, recipe_text: str, name: str = None):
        """레시피를 저장합니다"""
        
        if not name:
            # 레시피 이름 추출
            lines = recipe_text.split('\n')
            for line in lines:
                if '요리명:' in line:
                    name = line.split('요리명:')[1].strip()
                    break
        
        recipe_data = {
            "name": name,
            "content": recipe_text,
            "saved_date": datetime.now().isoformat(),
            "tags": []
        }
        
        self.saved_recipes.append(recipe_data)
        
        # 파일로 저장
        with open("saved_recipes.json", "w", encoding="utf-8") as f:
            json.dump(self.saved_recipes, f, ensure_ascii=False, indent=2)
        
        return f"✅ '{name}' 레시피가 저장되었습니다!"
    
    def load_saved_recipes(self):
        """저장된 레시피를 불러옵니다"""
        try:
            with open("saved_recipes.json", "r", encoding="utf-8") as f:
                self.saved_recipes = json.load(f)
            return f"📚 {len(self.saved_recipes)}개의 레시피를 불러왔습니다."
        except FileNotFoundError:
            return "저장된 레시피가 없습니다."
    
    def search_recipes(self, keyword: str) -> List[Dict]:
        """저장된 레시피에서 검색합니다"""
        results = []
        for recipe in self.saved_recipes:
            if keyword.lower() in recipe["name"].lower() or \
               keyword.lower() in recipe["content"].lower():
                results.append(recipe)
        return results


class InteractiveRecipeHelper:
    """대화형 레시피 도우미"""
    
    def __init__(self):
        self.helper = RecipeHelper()
        self.current_recipe = None
        
    def start(self):
        """대화형 모드를 시작합니다"""
        print("""
🍳 AI 레시피 도우미에 오신 것을 환영합니다!
        
명령어:
- '재료' : 재료로 레시피 추천받기
- '검색' : 저장된 레시피 검색
- '영양' : 현재 레시피 영양 분석
- '수정' : 현재 레시피 수정
- '저장' : 현재 레시피 저장
- '쇼핑' : 쇼핑 리스트 만들기
- '도움말' : 명령어 보기
- '종료' : 프로그램 종료
        """)
        
        # 저장된 레시피 불러오기
        print(self.helper.load_saved_recipes())
        
        while True:
            command = input("\n🍽️ 무엇을 도와드릴까요? ").strip()
            
            if command == "종료":
                print("👋 맛있는 요리 하세요!")
                break
            elif command == "재료":
                self.handle_ingredients()
            elif command == "검색":
                self.handle_search()
            elif command == "영양":
                self.handle_nutrition()
            elif command == "수정":
                self.handle_modify()
            elif command == "저장":
                self.handle_save()
            elif command == "쇼핑":
                self.handle_shopping()
            elif command == "도움말":
                self.show_help()
            else:
                print("❓ 알 수 없는 명령어입니다. '도움말'을 입력해보세요.")
    
    def handle_ingredients(self):
        """재료 기반 레시피 추천을 처리합니다"""
        print("\n🥗 어떤 재료를 가지고 계신가요?")
        print("(쉼표로 구분해서 입력해주세요)")
        
        ingredients_input = input("재료: ").strip()
        ingredients = [i.strip() for i in ingredients_input.split(',')]
        
        # 옵션 확인
        dietary = input("식이 제한이 있나요? (없으면 엔터): ").strip()
        cuisine = input("선호하는 요리 종류는? (한식/중식/일식/양식 등, 없으면 엔터): ").strip()
        
        print("\n🔍 레시피를 찾고 있습니다...")
        
        # 레시피 추천
        recipe = self.helper.suggest_recipe(
            ingredients,
            dietary_restrictions=dietary if dietary else None,
            cuisine_type=cuisine if cuisine else None
        )
        
        self.current_recipe = recipe
        print("\n" + recipe)
        
        # 다음 액션 제안
        print("\n💡 다음 작업을 할 수 있습니다:")
        print("- '영양' : 영양 정보 분석")
        print("- '수정' : 레시피 수정 (인분 조절 등)")
        print("- '저장' : 이 레시피 저장")
        print("- '쇼핑' : 쇼핑 리스트 만들기")
    
    def handle_search(self):
        """레시피 검색을 처리합니다"""
        keyword = input("\n🔍 검색어를 입력하세요: ").strip()
        
        results = self.helper.search_recipes(keyword)
        
        if results:
            print(f"\n📚 '{keyword}' 검색 결과: {len(results)}개")
            for i, recipe in enumerate(results, 1):
                print(f"{i}. {recipe['name']} (저장일: {recipe['saved_date'][:10]})")
            
            # 레시피 선택
            try:
                choice = int(input("\n번호를 선택하세요 (0: 취소): "))
                if 1 <= choice <= len(results):
                    self.current_recipe = results[choice-1]['content']
                    print("\n" + self.current_recipe)
            except:
                print("취소되었습니다.")
        else:
            print(f"❌ '{keyword}'에 대한 검색 결과가 없습니다.")
    
    def handle_nutrition(self):
        """영양 정보 분석을 처리합니다"""
        if not self.current_recipe:
            print("❌ 먼저 레시피를 선택해주세요.")
            return
        
        print("\n📊 영양 정보를 분석하고 있습니다...")
        nutrition = self.helper.analyze_nutrition(self.current_recipe)
        
        print("\n🥗 영양 정보 (1인분 기준):")
        print(f"- 칼로리: {nutrition.get('calories', '?')} kcal")
        print(f"- 단백질: {nutrition.get('protein', '?')} g")
        print(f"- 탄수화물: {nutrition.get('carbs', '?')} g")
        print(f"- 지방: {nutrition.get('fat', '?')} g")
        print(f"- 나트륨: {nutrition.get('sodium', '?')} mg")
    
    def handle_modify(self):
        """레시피 수정을 처리합니다"""
        if not self.current_recipe:
            print("❌ 먼저 레시피를 선택해주세요.")
            return
        
        print("\n✏️ 어떻게 수정하시겠습니까?")
        print("예: '4인분으로 늘려줘', '매운맛 줄여줘', '닭고기를 두부로 바꿔줘'")
        
        modification = input("수정 사항: ").strip()
        
        print("\n🔄 레시피를 수정하고 있습니다...")
        modified_recipe = self.helper.modify_recipe(self.current_recipe, modification)
        
        self.current_recipe = modified_recipe
        print("\n" + modified_recipe)
    
    def handle_save(self):
        """레시피 저장을 처리합니다"""
        if not self.current_recipe:
            print("❌ 먼저 레시피를 선택해주세요.")
            return
        
        custom_name = input("\n💾 레시피 이름 (엔터: 자동): ").strip()
        
        result = self.helper.save_recipe(
            self.current_recipe,
            name=custom_name if custom_name else None
        )
        print(result)
    
    def handle_shopping(self):
        """쇼핑 리스트 생성을 처리합니다"""
        if not self.current_recipe:
            print("❌ 먼저 레시피를 선택해주세요.")
            return
        
        print("\n🛒 현재 가지고 있는 재료를 입력해주세요:")
        print("(쉼표로 구분)")
        
        available = input("보유 재료: ").strip()
        available_list = [i.strip() for i in available.split(',')] if available else []
        
        print("\n📝 쇼핑 리스트를 만들고 있습니다...")
        shopping_list = self.helper.create_shopping_list(
            self.current_recipe,
            available_list
        )
        
        if shopping_list:
            print("\n🛒 구매해야 할 재료:")
            for item in shopping_list:
                print(f"  {item}")
            
            # 쇼핑 리스트 저장 옵션
            save_list = input("\n쇼핑 리스트를 저장하시겠습니까? (y/n): ").lower()
            if save_list == 'y':
                with open("shopping_list.txt", "w", encoding="utf-8") as f:
                    f.write("🛒 쇼핑 리스트\n")
                    f.write(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                    for item in shopping_list:
                        f.write(f"□ {item}\n")
                print("✅ shopping_list.txt에 저장되었습니다!")
        else:
            print("✅ 추가로 구매할 재료가 없습니다!")
    
    def show_help(self):
        """도움말을 표시합니다"""
        print("""
📖 레시피 도우미 사용법:

🥗 재료로 레시피 찾기:
1. '재료' 명령어 입력
2. 보유한 재료들을 쉼표로 구분해서 입력
3. 식이 제한이나 선호 요리 종류 선택 (선택사항)

🔍 저장된 레시피 검색:
1. '검색' 명령어 입력
2. 검색어 입력 (요리명, 재료 등)
3. 결과에서 번호 선택

📊 영양 정보 확인:
- 현재 레시피의 영양 정보를 분석합니다

✏️ 레시피 수정:
- 인분 조절, 재료 대체 등 자유롭게 수정 가능

💾 레시피 저장:
- 마음에 드는 레시피를 저장해두고 나중에 검색 가능

🛒 쇼핑 리스트:
- 현재 가진 재료를 입력하면 부족한 재료 리스트 생성
        """)


# 추가 유틸리티 함수들
def create_meal_plan(helper: RecipeHelper, days: int = 7):
    """주간 식단을 생성합니다"""
    
    prompt = f"""
    {days}일간의 균형잡힌 식단을 만들어주세요.
    
    각 날짜별로:
    - 아침
    - 점심  
    - 저녁
    
    다양하고 영양가 있는 메뉴로 구성해주세요.
    한국인의 입맛에 맞게 만들어주세요.
    """
    
    response = helper.ai.ask(prompt)
    return response.text


def suggest_by_mood(helper: RecipeHelper, mood: str):
    """기분에 따른 요리 추천"""
    
    mood_foods = {
        "피곤": "에너지를 주는 영양가 높은",
        "우울": "기분을 좋게 만드는 달콤하거나 따뜻한",
        "스트레스": "스트레스 해소에 좋은 편안한",
        "행복": "특별한 날에 어울리는 화려한",
        "더워": "시원하고 상큼한",
        "추워": "따뜻하고 든든한"
    }
    
    food_type = mood_foods.get(mood, "맛있는")
    
    prompt = f"""
    {mood}한 기분일 때 먹기 좋은 {food_type} 요리 3가지를 추천해주세요.
    각 요리의 특징과 왜 이 기분에 좋은지 설명해주세요.
    """
    
    response = helper.ai.ask(prompt)
    return response.text


# 메인 실행 코드
if __name__ == "__main__":
    # 대화형 모드 실행
    app = InteractiveRecipeHelper()
    app.start()
```

## 🎮 사용 예시

### 1. 기본 사용법

```python
# 레시피 도우미 생성
helper = RecipeHelper()

# 재료로 요리 추천
ingredients = ["돼지고기", "김치", "두부"]
recipe = helper.suggest_recipe(ingredients, cuisine_type="한식")
print(recipe)
```

### 2. 영양 정보 분석

```python
# 레시피의 영양 정보 분석
nutrition = helper.analyze_nutrition(recipe)
print(f"칼로리: {nutrition['calories']} kcal")
```

### 3. 기분에 따른 추천

```python
# 피곤할 때 좋은 요리 추천
suggestion = suggest_by_mood(helper, "피곤")
print(suggestion)
```

### 4. 주간 식단 생성

```python
# 7일 식단 만들기
meal_plan = create_meal_plan(helper, days=7)
print(meal_plan)
```

## 💡 확장 아이디어

1. **알레르기 관리**: 특정 재료 자동 제외
2. **칼로리 계산기**: 목표 칼로리에 맞는 식단
3. **요리 타이머**: 단계별 타이머 기능
4. **사진 인식**: 재료 사진으로 인식
5. **커뮤니티**: 레시피 공유 기능

## 🔧 커스터마이징

### 선호도 설정 추가

```python
class PersonalizedRecipeHelper(RecipeHelper):
    def __init__(self):
        super().__init__()
        self.preferences = {
            "spicy_level": 3,  # 1-5
            "cooking_time": "30분 이내",
            "difficulty": "초급"
        }
```

### 데이터베이스 연동

```python
import sqlite3

def save_to_database(recipe_data):
    conn = sqlite3.connect('recipes.db')
    # ... 데이터베이스 저장 로직
```

## ✅ 학습 포인트

1. **클래스 설계**: 기능별로 메서드 분리
2. **프롬프트 엔지니어링**: 구조화된 출력 요청
3. **파일 입출력**: JSON으로 데이터 저장
4. **사용자 인터페이스**: 대화형 명령어 처리
5. **에러 처리**: 예외 상황 대응

## 🚀 다음 단계

레시피 도우미를 완성했다면:
- 웹 인터페이스 추가 (Flask/FastAPI)
- 모바일 앱으로 확장
- 음성 인식 기능 추가
- 다른 프로젝트 도전!

---

다음 프로젝트 [공부 도우미](../study-buddy/)도 확인해보세요! 📚