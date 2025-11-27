# AIE Insight Bot

팀 맞춤형 논문/데이터 리서치 자동화 시스템

## 목적

- 팀이 집중하는 주제(VLM, CV, 시스템 구축 등)에 대해 매주 자동으로 최신 논문 및 레퍼런스 자료를 요약하고 공유
- 팀 전체의 기술 흡수 속도를 높이고, "리서치 공유 문화"를 정착

## 주요 기능

1. **Paper Collection**: Hugging Face Papers API에서 최신 인기 논문 자동 수집
2. **Intelligent Embedding**: OpenAI Embeddings로 논문 벡터화 및 Milvus 저장
3. **Interest Management**: Slack 명령어로 팀원 관심사 등록
4. **Smart Recommendation**: 코사인 유사도 기반 맞춤형 논문 추천 (Top 3)
5. **Auto Summarization**: GPT-4o-mini로 핵심 요약 + 맞춤 요약 생성
6. **Slack Integration**: 자동으로 #test-noti 채널에 추천 결과 게시
7. **Analytics & Reporting**: 피드백 수집 및 월말 리포트 자동 생성

## 기술 스택

- **Language**: Python 3.11
- **Framework**: FastAPI
- **Vector Database**: Milvus
- **LLM**: OpenAI (GPT-4o-mini, text-embedding-3-small)
- **Integration**: Slack Bolt SDK
- **Containerization**: Docker, Docker Compose
- **Dependency Management**: uv
- **Code Quality**: Ruff, pytest, mypy

## 프로젝트 구조

```
paper_recommendation/
├── config/              # 설정 파일
├── mcp_servers/         # MCP 서버 모듈들
│   ├── paper_collector/ # 논문 수집
│   ├── vector_store/    # 벡터 저장소
│   ├── interest_manager/# 관심사 관리
│   └── analytics/       # 분석 및 리포트
├── src/
│   ├── recommender/     # 추천 엔진
│   ├── slack/           # Slack 봇
│   ├── scheduler/       # 스케줄러
│   └── orchestrator/    # MCP 클라이언트
├── data/                # 데이터 저장소
├── logs/                # 로그
└── tests/               # 테스트
```

## 설치 및 실행

### 1. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 필요한 값들을 설정
```

필수 환경 변수:
- `OPENAI_API_KEY`: OpenAI API 키
- `SLACK_BOT_TOKEN`: Slack 봇 토큰
- `SLACK_APP_TOKEN`: Slack 앱 토큰
- `SLACK_CHANNEL_ID`: Slack 채널 ID

### 2. Docker Compose로 실행

```bash
# 전체 서비스 시작 (Milvus + 애플리케이션)
docker-compose up -d

# 로그 확인
docker-compose logs -f app

# 서비스 중지
docker-compose down
```

### 3. 로컬 개발 환경

```bash
# 의존성 설치
pip install uv
uv pip install -e ".[dev]"

# Milvus만 Docker로 실행
docker-compose up -d milvus

# 애플리케이션 실행
python main.py
```

## 사용 방법

### Slack 명령어

1. **관심사 등록**
   ```
   /set_interest VLM을 이용한 CCTV 객체 검출
   ```

2. **수동 추천 요청**
   ```
   /insight
   ```

3. **월말 리포트 조회**
   ```
   /report
   ```

### 자동 스케줄

- 매주 월요일 09:00에 자동으로 논문 수집 및 추천 실행
- 결과는 #test-noti 채널에 자동 게시

## 개발 가이드

### 코드 포맷팅

```bash
# Ruff로 코드 포맷팅
ruff format .

# Lint 검사
ruff check .
```

### 테스트 실행

```bash
# 전체 테스트
pytest

# 커버리지 포함
pytest --cov=src --cov-report=html
```

### 타입 체크

```bash
mypy src/
```

## 기대 효과

- 팀 내 논문 검색 피로도 80% 이상 감소
- 최신 기술에 대한 공유 속도 및 품질 향상
- 월말 리서치 리포트 자동화로 회의 시간 절감
- 기술 흡수 → 실험 → 적용 → 공유의 AX 순환 구조 완성

## 라이선스

MIT License
