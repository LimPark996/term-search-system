# 용어 추천 시스템

AI 기반 4단계 파이프라인을 통한 한국어 자연어 질의 처리 및 표준 용어 검색 시스템

## 시스템 개요

이 시스템은 자연어로 입력된 질의를 분석하여 가장 적합한 표준 용어를 검색하는 AI 기반 검색 시스템입니다. OpenAI GPT 모델과 하이브리드 검색 기법을 결합하여 높은 정확도의 용어 매칭을 제공합니다.

### 주요 특징
- **자연어 질의 처리**: 구어체 질문을 표준 검색어로 자동 변환
- **하이브리드 검색**: 의미적 유사도와 키워드 매칭을 결합한 정확한 검색
- **AI 기반 추천**: GPT를 활용한 맥락적 용어 선별 및 추천 이유 제공
- **캐시 시스템**: 임베딩 결과 캐싱으로 빠른 재실행 및 API 비용 절약

## 시스템 아키텍처

### 현재 구현: 4단계 용어 검색 파이프라인

```
자연어 질의 → Phase 1 → Phase 2&3 → Phase 4 → 용어 추천
              전처리     하이브리드    AI 재순위
                        검색         & 설명
```

**Phase 1: 입력 전처리**
```
INPUT: "웹사이트에서 다른 시스템이랑 데이터 주고받는 거 어떻게 해?"
       ↓ OpenAI GPT 프롬프팅
OUTPUT: "웹사이트 시스템간 데이터 교환 방법"
```

**Phase 2&3: 하이브리드 검색**

*사전 준비 (인덱스 구축):*
```python
# 1. 의미적 임베딩 생성 및 FAISS 인덱스 구축
semantic_texts = ["OAuth 인증: OAuth는 웹 API에서...", ...]
semantic_embeddings = generate_embeddings(semantic_texts)
faiss_index = faiss.IndexFlatIP(1536)
faiss_index.add(semantic_embeddings)  # 의미적 임베딩만 FAISS에 저장

# 2. 토큰별 임베딩 생성 및 메타데이터 저장
for each_term:
    tokens = tokenize("OAuth 인증: OAuth는...")  # ["OAuth", "인증", "웹", "API", ...]
    token_embeddings = generate_embeddings(tokens)
    
    token_metadata.append({
        'semantic_embedding': semantic_embedding,  # 전체 문장 임베딩
        'tokens': tokens,
        'token_embeddings': token_embeddings       # 토큰별 임베딩 리스트
    })
```

*검색 과정 (현재 구현):*
```python
# 1. 쿼리 임베딩 생성
query_semantic_emb = generate_embedding("실시간 보안 API")
query_tokens = ["실시간", "보안", "API"]
query_token_embs = generate_embeddings(query_tokens)

# 2. 전체 순회 검색 (FAISS 미사용)
for term_data in token_metadata:
    # 의미적 점수: 전체 문장 간 유사도
    semantic_score = cosine_similarity(query_semantic_emb, term_data['semantic_embedding'])
    
    # ColBERT 점수: 토큰별 최대 유사도의 평균
    colbert_score = calculate_colbert_score(query_token_embs, term_data['token_embeddings'])
    
    final_score = 0.6 × semantic_score + 0.4 × colbert_score
```

*점수 계산 예시:*
```
질의: "실시간 보안 API"
- 전체 임베딩: [질의 전체를 하나의 벡터로 변환]
- 토큰별 임베딩: ["실시간"벡터, "보안"벡터, "API"벡터]

용어: "OAuth 인증: OAuth는 웹 API에서 보안 인증을 위한..."
- 전체 임베딩: [용어+설명 전체를 하나의 벡터로 변환] 
- 문서 토큰들: ["OAuth", "인증", "웹", "API", "보안", "프로토콜"...]
- 문서 토큰 임베딩들: 각 토큰마다 개별 벡터

점수 계산:
1. 의미적 점수: 질의전체벡터 vs 용어전체벡터 유사도 = 0.45
2. ColBERT 점수: 각 쿼리토큰이 문서토큰들과 가장 유사한 값들의 평균
   - "실시간" vs 문서토큰들 → 최대값: 0.15
   - "보안" vs 문서토큰들 → 최대값: 0.92 (정확 매칭)
   - "API" vs 문서토큰들 → 최대값: 0.88 (정확 매칭)
   - 평균: (0.15 + 0.92 + 0.88) ÷ 3 = 0.65

최종 점수: 0.6 × 0.45 + 0.4 × 0.65 = 0.53
```

**주의:** 현재 구현에서는 FAISS 인덱스를 구축하지만 실제 검색 시에는 사용하지 않고 전체 용어를 순회하며 직접 유사도를 계산합니다. 대용량 데이터에서는 FAISS를 활용한 1차 필터링 후 ColBERT 점수 계산으로 최적화가 필요합니다.

**Phase 4: AI 기반 최종 선별**
```
INPUT: 후보 용어들 (임계값 0.3 이상)
       ↓ OpenAI GPT 분석
OUTPUT: 최종 추천 용어 + 추천 이유
```

### 향후 확장 계획: 실시간 데이터베이스 조회 시스템

현재 시스템을 기반으로 자연어 질의를 통한 실시간 DB 조회 시스템으로 확장 예정:

```
자연어 질의 → 용어 매칭 → 약어 변환 → SQL 생성 → 사용자 확인 → DB 실행
             (현재 구현)   (현재 구현)   (MCP 활용)
```

**3단계 처리 파이프라인 + 사용자 검증:**
1. **자연어 → 표준용어 매칭**: 현재 구현된 하이브리드 검색 활용
2. **표준용어 → 약어 변환**: 기존 용어 검색 시스템 활용  
3. **쿼리 생성 + 사용자 확인**: MCP 서버를 통한 SQL 생성 및 안전한 실행

## 파일 구조

### Python 모듈
- **`main.py`**: 사용자 인터페이스와 전체 파이프라인 실행 진입점
- **`recommendation_system.py`**: 4단계 파이프라인 통합 관리 컨트롤러
- **`query_preprocessor.py`**: Phase 1 - 자연어 질의를 정제된 검색어로 변환
- **`search_engine.py`**: Phase 2&3 - 임베딩 생성, FAISS 인덱싱, 하이브리드 검색
- **`ai_reranker.py`**: Phase 4 - GPT 기반 최종 후보 선별 및 추천 이유 생성
- **`data_loader.py`**: Google Sheets 용어 데이터 로드 및 Term 객체 변환
- **`text_utils.py`**: 한국어 토큰화 (정규식 + KoNLPy 형태소 분석)
- **`models.py`**: Term과 SearchResult 데이터클래스 정의
- **`config.py`**: API 키, 가중치, 임계값 등 시스템 설정 관리

### 캐시 파일 (자동 생성)
- **`semantic_embeddings.npy`**: 용어별 의미적 임베딩 벡터 (FAISS 인덱스 재구축용)
- **`token_metadata.pkl`**: 토큰별 메타데이터 및 임베딩 정보 (ColBERT 검색용)
- **`faiss_index.bin`**: FAISS 벡터 검색 인덱스 (구축되지만 현재 미사용)

*캐시 파일들은 OpenAI API 호출 결과를 저장하여 2회차부터 API 비용 없이 즉시 검색 가능. 단, FAISS 인덱스는 구축되지만 실제 검색에서는 미사용 상태*

## 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 설정
`.env.example`을 복사하여 `.env` 파일 생성 후 API 키 입력:
```bash
cp .env.example .env
```

`.env` 파일 편집:
```
SPREADSHEET_ID=your_spreadsheet_id_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Google Sheets 인증
- Google Cloud Console에서 OAuth 클라이언트 ID 생성
- `credentials.json` 파일을 프로젝트 루트에 저장
- 자세한 설정 방법은 OAuth 문제 해결 가이드 참조

## 사용법

### 기본 실행
```bash
python main.py
```

### 사용 예시
```
=== 용어 추천 시스템 ===
자연어 질의를 입력하세요: 고객 정보를 안전하게 저장하는 방법

🔍 처리 중: '고객 정보를 안전하게 저장하는 방법'
📝 원본 질의: 고객 정보를 안전하게 저장하는 방법
🎯 정제된 검색어: 고객정보 보안 저장 방법
🔍 후보 개수: 15개

🏆 AI 최종 추천 (3개):
1위. 개인정보보호
    설명: 개인의 사생활과 관련된 정보를 보호하는 것
    점수: 0.847 (의미: 0.782, 키워드: 0.945)
    매칭 키워드: 정보, 보호
    AI 추천 이유: 고객 정보 보안 저장의 핵심 개념으로 가장 적합

2위. 데이터암호화
    설명: 데이터를 암호화하여 보안을 강화하는 기술
    점수: 0.823 (의미: 0.756, 키워드: 0.923)
    매칭 키워드: 데이터, 보안
    AI 추천 이유: 안전한 저장을 위한 구체적인 기술적 방법
```

## 설정 값 조정

`config.py`에서 시스템 동작을 조정할 수 있습니다:

```python
# 검색 파라미터
SEARCH_THRESHOLD = 0.3      # 후보 선별 최소 점수
SEMANTIC_WEIGHT = 0.6       # 의미적 유사도 가중치
COLBERT_WEIGHT = 0.4        # 키워드 매칭 가중치
```

## 기술 스택

- **Python 3.8+**
- **OpenAI GPT API**: 자연어 처리 및 임베딩 생성
- **FAISS**: 고속 벡터 유사도 검색
- **KoNLPy**: 한국어 형태소 분석
- **Google Sheets API**: 용어 데이터 관리
- **pandas**: 데이터 처리

## 성능 특징

- **초기 실행**: 용어 데이터 임베딩 생성으로 3-5분 소요
- **캐시 이후**: 즉시 검색 (< 1초)
- **정확도**: 의미적 검색과 키워드 매칭의 하이브리드로 높은 정확도
- **확장성 제한**: 현재 전체 순회 검색으로 대용량 데이터 처리시 성능 저하 가능
- **최적화 여지**: FAISS 인덱스를 활용한 1차 필터링으로 대폭 성능 개선 가능

## 현재 한계사항

### 1. 용어 추천 범위 제한
- **현재**: 질의에 대한 관련 **용어**만 제공 가능
- **한계**: 질의에 대한 실제 **데이터 값** 제공 불가능
- **예시**: 
  - ✅ 가능: "직원 정보 관리" → "직원정보", "사원번호", "인사관리" 등 용어 추천
  - ❌ 불가능: "IT부서 평균 연봉이 얼마야?" → 실제 연봉 데이터 값 조회

### 2. 정적 검색 시스템
- **현재**: 사전에 정의된 용어사전 기반 검색
- **한계**: 실시간 데이터베이스 조회 및 동적 데이터 분석 불가능

### 3. 해결 방안: MCP 기반 쿼리 생성 시스템 (향후 계획)
현재 용어 추천 시스템을 기반으로 실시간 데이터베이스 조회 시스템으로 확장 예정:

```
사용자 질의: "IT부서 평균 연봉이 얼마야?"
              ↓
1단계: 용어 매칭 (현재 시스템 활용)
- "평균" → 집계 함수 필요
- "연봉" → SALARY_AMT 컬럼
- "IT부서" → DEPT_CD 필터링

2단계: SQL 생성 (MCP 서버 활용)
- 테이블 스키마 분석
- 관계 추론 및 JOIN 로직
- 보안 검증된 쿼리 생성

3단계: 사용자 확인 후 실행
- 생성된 SQL 사용자 확인
- 안전한 실행 및 결과 제공
```

이를 통해 자연어 질의로 실제 데이터 값까지 조회 가능한 통합 시스템 구축 계획