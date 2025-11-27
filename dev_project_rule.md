"AIE Insight Bot” — 팀 맞춤형 논문/데이터 리서치 자동화 시스템
목적
* 팀이 집중하는 주제(VLM, CV, 시스템 구축 등)에 대해 매주 자동으로 최신 논문 및 레퍼런스 자료를 요약하고 공유
* 팀 전체의 기술 흡수 속도를 높이고, “리서치 공유 문화”를 정착

영역 설명 입력 인터페이스 팀원들이 자연어 문장(예: “VLM을 이용한 CCTV 객체 검출”)으로 관심 주제 입력 데이터 소스 Hugging Face Papers API, ArXiv API, Semantic Scholar 등 백엔드 로직 주 1회(월요일 09:00) 인기 논문 Top30 임베딩 → 팀 관심 주제와 코사인 유사도 계산 → 상위 3개 요약 요약 방식 ① 핵심 요약 (기계적 요약) ② 맞춤 요약 (관심 주제 맥락 포함) 출력 방식 Slack Thread로 결과 전송 또는 ‘/insight’ 명령어로 요청 시 출력 성과 지표 클릭 수, 이모티콘 수, 논문 다운로드 수 등 → 월말 보고서 자동 생성 추가 기능(확장) “불확실성 지도” 제공 – 추천 근거의 신뢰도와 요약의 추정 정확도 표시

┌──────────────────────────────┐
│ [1] Paper Collector           │
│  - Hugging Face Papers API     │
│  - 최신 인기 논문 상위 30개 추출   │
│  - 요약, 제목, 링크 메타데이터 수집 │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ [2] Embedding & Storage       │
│  - 논문 Abstract + Title 임베딩 │
│  - Milvus / ChromaDB에 저장     │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ [3] User Interest Collector    │
│  - Slack 명령어 /set_interest │
│  - 입력 문장 임베딩 및 저장     │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ [4] Recommendation Engine      │
│  - cosine similarity 기반 상위 3개 추천 │
│  - 핵심요약 + 맞춤요약 생성             │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ [5] Slack Integration          │
│  - #soln_chpt_aie 채널 자동 쓰레드 생성 │
│  - 각 논문: 제목 / 링크 / 요약 / 반응버튼 │
│  - Reaction Event 수집              │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ [6] Feedback & Report Module  │
│  - 클릭률, 이모지 반응, 성공률 통계화     │
│  - 월말 자동 리포트 생성 /report)      │
└──────────────────────────────┘

기대효과
* 팀 내 논문 검색 피로도 80% 이상 감소
* 최신 기술에 대한 공유 속도 및 품질 향상
* 월말 리서치 리포트 자동화로 회의 시간 절감
* 기술 흡수 → 실험 → 적용 → 공유의 AX 순환 구조 완성

Update Todos

프로젝트 구조 및 초기 설정 (폴더 구조, requirements.txt, .env.example)

Paper Collector 모듈 구현 (Hugging Face Papers API 연동)

Embedding & Storage 모듈 구현 (ChromaDB 연동)

User Interest Collector 모듈 구현 (Slack 명령어 처리)

Recommendation Engine 구현 (코사인 유사도 기반 추천)

Slack Integration 구현 (메시지 전송 및 반응 수집)

Feedback & Report Module 구현 (통계 및 리포트 생성)

스케줄러 설정 (주 1회 자동 실행)

테스트 및 문서화



### 기술 스택 정리
Slack: #test-noti 채널
Vector DB: Milvus (익숙하신 걸로!)
Embedding: OpenAI Embeddings (text-embedding-3-small 사용 - 가장 저렴)
LLM: OpenAI GPT-4o-mini (가장 저렴하면서 성능 좋음)
배포: 로컬 서버


### 프로젝트 구조 설계
paper_recommendation/
├── .env.example
├── .env
├── .gitignore
├── docker-compose.yml           # 전체 서비스 오케스트레이션
├── Dockerfile                   # 메인 애플리케이션 Dockerfile
├── requirements.txt
├── README.md
├── dev_rule.md
│
├── docker/                      # Docker 관련 파일들
│   ├── milvus/
│   │   └── docker-compose.yml   # Milvus standalone 설정
│   ├── nginx/                   # (선택) 리버스 프록시
│   │   └── nginx.conf
│   └── scripts/
│       ├── init.sh              # 초기 설정 스크립트
│       └── healthcheck.sh
│
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── mcp_config.json
│
├── mcp_servers/                 # 각 MCP 서버는 독립 컨테이너로 실행 가능
│   │
│   ├── paper_collector/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── huggingface_api.py
│   │   └── README.md
│   │
│   ├── vector_store/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── milvus_client.py
│   │   ├── embeddings.py
│   │   └── README.md
│   │
│   ├── interest_manager/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── storage.py
│   │   └── README.md
│   │
│   └── analytics/
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── __init__.py
│       ├── server.py
│       ├── tracker.py
│       └── README.md
│
├── src/
│   ├── __init__.py
│   ├── recommender/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── summarizer.py
│   │
│   ├── slack/
│   │   ├── __init__.py
│   │   ├── bot.py
│   │   ├── commands.py
│   │   └── message_formatter.py
│   │
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── tasks.py
│   │
│   └── orchestrator/
│       ├── __init__.py
│       └── mcp_client.py
│
├── data/                        # Docker volume으로 마운트
│   ├── interests.json
│   └── feedback.json
│
├── logs/                        # Docker volume으로 마운트
│
├── tests/
│   └── __init__.py
│
└── main.py