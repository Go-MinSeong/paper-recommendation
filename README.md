# AIE Insight Bot

팀 맞춤형 AI/ML 논문 추천 자동화 시스템

## 개요

AIE Insight Bot은 HuggingFace의 주간 트렌딩 논문을 자동으로 수집하고, 팀원 개인의 관심사에 맞춰 맞춤형 논문을 추천하는 Slack 봇입니다.

## 주요 기능

### 1. 논문 수집 (Paper Collection)

```
┌─────────────────────┐
│   HuggingFace       │
│   Weekly Trending   │
│   (huggingface_hub) │
└──────────┬──────────┘
           │
           │ • 주간 트렌딩 논문
           │ • upvotes >= 10 필터
           │ • 최대 100개 수집
           ▼
┌─────────────────────┐
│   Vector DB         │
│   (Milvus)          │
│   + OpenAI Embed    │
└─────────────────────┘
```

- **HuggingFace Daily Papers**: `huggingface_hub` 라이브러리를 사용하여 주간 트렌딩 논문 수집
- **품질 필터링**: upvotes 10 이상인 논문만 수집
- **중복 방지**: paper_id 기반 중복 논문 자동 제외
- **벡터 임베딩**: OpenAI `text-embedding-3-small` 모델로 논문 벡터화

### 2. 관심사 기반 추천 (Interest-based Recommendation)

```
┌─────────────────────────────────────────────────┐
│  사용자 관심사                                    │
│  예: "VLM, Vision Language Model 연구"          │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  1. 관심사 → Embedding 변환                      │
│  2. Milvus 벡터 유사도 검색 (Cosine Similarity)  │
│  3. 이전 추천 기록 필터링 (중복 추천 방지)         │
│  4. Semantic Scholar에서 인용수/발행일 조회       │
│  5. GPT-4o-mini로 요약 생성                      │
│     - Core Summary: 핵심 내용 요약               │
│     - Contextualized Summary: 관심사 맥락 요약   │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  Slack 메시지로 추천 결과 전송                    │
└─────────────────────────────────────────────────┘
```

### 3. Slack 통합

#### 명령어

| 명령어 | 설명 |
|--------|------|
| `/set-interest <관심사>` | 개인 관심사 등록/수정 |
| `/my-interest` | 등록된 관심사 확인 |
| `/insight` | 수동 논문 추천 요청 |
| `/papers` | 현재 저장된 논문 목록 확인 |
| `/collect` | 수동 논문 수집 트리거 |

#### 자동 추천 설정

| 명령어 | 설명 |
|--------|------|
| `/auto-recommend on` | 자동 추천 활성화 |
| `/auto-recommend off` | 자동 추천 비활성화 |
| `/auto-recommend status` | 현재 설정 확인 |
| `/auto-recommend schedule <cron>` | 스케줄 설정 (예: `0 9 * * 1` = 매주 월요일 9시) |

### 4. 자동 스케줄링

- **논문 수집**: 설정된 주기로 자동 실행 (기본: 24시간)
- **자동 추천**: 사용자별 개인 스케줄에 따라 자동 추천 발송
- **시작 시 즉시 수집**: 서버 시작 시 즉시 논문 수집 실행

## 기술 스택

| 구분 | 기술 |
|------|------|
| Language | Python 3.11 |
| Framework | FastAPI |
| Vector DB | Milvus |
| LLM | OpenAI GPT-4o-mini |
| Embedding | OpenAI text-embedding-3-small |
| Slack | Slack Bolt SDK (Socket Mode) |
| Paper Source | HuggingFace Hub |
| Container | Docker, Docker Compose |

## 프로젝트 구조

```
paper-recommendation/
├── config/
│   ├── logger.py              # Loguru 로깅 설정
│   └── settings.py            # Pydantic Settings 설정
├── mcp_servers/
│   ├── paper_collector/       # 논문 수집
│   │   ├── huggingface_api.py # HuggingFace Hub 클라이언트
│   │   ├── semantic_scholar_api.py # Semantic Scholar 클라이언트
│   │   └── models.py          # Paper, PaperCollection 모델
│   ├── vector_store/          # 벡터 저장소
│   │   ├── milvus_client.py   # Milvus 클라이언트
│   │   ├── embeddings.py      # OpenAI Embeddings
│   │   └── service.py         # Vector Store 서비스
│   ├── interest_manager/      # 관심사 관리
│   │   ├── storage.py         # 관심사 저장소
│   │   └── models.py          # UserInterest 모델
│   ├── recommendation_history/# 추천 기록 관리
│   │   └── storage.py         # 추천 기록 저장소
│   └── auto_recommend/        # 자동 추천 설정
│       └── storage.py         # 자동 추천 저장소
├── src/
│   ├── recommender/
│   │   ├── engine.py          # 추천 엔진
│   │   └── summarizer.py      # GPT 요약 생성
│   ├── scheduler/
│   │   ├── collector.py       # 논문 수집 스케줄러
│   │   ├── recommendation.py  # 추천 스케줄러
│   │   └── auto_recommend.py  # 자동 추천 스케줄러
│   └── slack/
│       ├── app.py             # Slack App 설정
│       ├── handlers/          # 명령어 핸들러
│       └── formatters/        # 메시지 포맷터
├── data/                      # JSON 저장소
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── main.py                    # 애플리케이션 진입점
```

## 설치 및 실행

### 1. 환경 변수 설정

```bash
cp .env.example .env
```

필수 환경 변수:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_CHANNEL_ID=C...

# Milvus
MILVUS_HOST=milvus
MILVUS_PORT=19530
```

주요 설정:

```env
# 논문 수집
PAPER_COLLECTION_LIMIT=100     # 수집할 논문 수
PAPER_MIN_UPVOTES=10           # 최소 upvotes
PAPER_SOURCE=huggingface       # 논문 소스

# 추천
TOP_K_RECOMMENDATIONS=3        # 추천 논문 수
MIN_SIMILARITY_SCORE=0.1       # 최소 유사도 점수
```

### 2. Docker Compose 실행

```bash
# 전체 서비스 시작
docker compose up -d

# 로그 확인
docker compose logs -f app

# 서비스 중지
docker compose down
```

### 3. 개발 환경

```bash
# Milvus만 Docker로 실행
docker compose up -d milvus etcd minio

# 의존성 설치
pip install -e ".[dev]"

# 애플리케이션 실행
python main.py
```

## 설정 옵션

### 논문 수집 설정

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `PAPER_COLLECTION_LIMIT` | 100 | 수집할 최대 논문 수 |
| `PAPER_MIN_UPVOTES` | 10 | HuggingFace 최소 upvotes |
| `PAPER_SOURCE` | huggingface | 논문 소스 (huggingface, semantic_scholar, both) |
| `COLLECTION_INTERVAL_HOURS` | 24.0 | 수집 주기 (시간) |

### 추천 설정

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `TOP_K_RECOMMENDATIONS` | 3 | 추천할 논문 수 |
| `MIN_SIMILARITY_SCORE` | 0.1 | 최소 코사인 유사도 |
| `AUTO_RECOMMEND_ENABLED` | false | 전역 자동 추천 활성화 |

### LLM 설정

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `LLM_MODEL` | gpt-4o-mini | 요약 생성 모델 |
| `EMBEDDING_MODEL` | text-embedding-3-small | 임베딩 모델 |
| `OPENAI_MAX_CONCURRENT_REQUESTS` | 5 | 동시 API 요청 수 |

## 개발 가이드

### 코드 품질

```bash
# 포맷팅
ruff format .

# 린트
ruff check .

# 타입 체크
mypy src/
```

### 테스트

```bash
# 전체 테스트
pytest

# 커버리지 포함
pytest --cov=src --cov-report=html
```

## 아키텍처

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Slack      │────▶│   FastAPI    │────▶│   Milvus     │
│   (Socket)   │     │   (main.py)  │     │   (Vector)   │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ HuggingFace │  │  OpenAI     │  │  Semantic   │
    │ Hub API     │  │  (Embed/LLM)│  │  Scholar    │
    └─────────────┘  └─────────────┘  └─────────────┘
```

## 라이선스

MIT License
# Test change for PR review
