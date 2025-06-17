# pyhub-llm 문서

이 디렉토리는 MkDocs를 사용한 pyhub-llm의 문서를 포함합니다.

## 로컬에서 문서 보기

```bash
# 개발 서버 실행
mkdocs serve

# 또는 Python 모듈로 실행
python -m mkdocs serve
```

브라우저에서 http://127.0.0.1:8000 접속

## 문서 빌드

```bash
# HTML로 빌드
mkdocs build

# 또는
python -m mkdocs build
```

빌드된 파일은 `site/` 디렉토리에 생성됩니다.

## 문서 구조

```
docs/
├── index.md                    # 홈페이지
├── getting-started/           # 시작하기
│   ├── index.md              # 섹션 개요
│   ├── installation.md       # 설치 가이드
│   └── quickstart.md         # 빠른 시작
├── guides/                    # 가이드
│   ├── index.md              # 가이드 목록
│   ├── basic-usage.md        # 기본 사용법
│   ├── conversation.md       # 대화 관리
│   ├── providers.md          # 프로바이더
│   ├── structured-output.md  # 구조화된 출력
│   └── advanced.md           # 고급 기능
└── assets/                    # 정적 파일
    └── stylesheets/
        └── extra.css         # 커스텀 CSS
```

## 새 페이지 추가하기

1. `docs/` 디렉토리에 `.md` 파일 생성
2. `mkdocs.yml`의 `nav` 섹션에 추가
3. 마크다운으로 내용 작성

## 스타일 커스터마이징

- `docs/assets/stylesheets/extra.css`에서 커스텀 스타일 추가
- Material 테마의 기본 스타일을 덮어쓸 수 있음

## 플러그인

현재 설정된 주요 플러그인:

- **search**: 한국어 지원 검색
- **mkdocstrings**: Python 코드 자동 문서화
- **git-revision-date-localized**: 페이지 수정 날짜 표시
- **glightbox**: 이미지 확대 기능
- **minify**: HTML/CSS/JS 압축

## 기여하기

문서 개선을 위한 PR은 언제나 환영합니다!