# Contributing to pyhub-llm

## PR 병합 전략

이 프로젝트는 선형 히스토리를 유지합니다. PR을 병합할 때 다음 방법 중 하나를 사용하세요:

### 1. Squash and Merge (권장)
하나의 기능/수정사항을 하나의 커밋으로 정리할 때 사용합니다.

```bash
gh pr merge --squash
```

### 2. Rebase and Merge
여러 의미있는 커밋들을 보존하면서 선형 히스토리를 유지할 때 사용합니다.

```bash
gh pr merge --rebase
```

### 3. 로컬에서 직접 작업
```bash
# feature 브랜치에서
git rebase main

# main 브랜치로 이동
git checkout main

# fast-forward merge
git merge --ff-only feature-branch
```

## 커밋 메시지 규칙

- `feat:` 새로운 기능
- `fix:` 버그 수정
- `docs:` 문서 수정
- `refactor:` 코드 리팩토링
- `test:` 테스트 추가/수정
- `chore:` 빌드, 설정 등

## 주의사항

- Merge commit (--merge)은 사용하지 않습니다
- PR 병합 후 브랜치는 자동으로 삭제됩니다