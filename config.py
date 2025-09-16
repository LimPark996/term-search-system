import os
from typing import Dict, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ===== 환경변수에서 민감한 정보 로드 =====
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# 환경변수 확인
if not SPREADSHEET_ID:
    print("⚠️ SPREADSHEET_ID 환경변수가 설정되지 않았습니다.")
    
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

# ===== 컬럼 매핑 =====
COLUMN_MAPPING = {
    'term_abbr': '공통표준용어영문약어명',
    'term_name': '공통표준용어명', 
    'term_desc': '공통표준용어설명',
    'domain': '공통표준도메인명',
    'word_name': '공통표준단어명',
    'word_abbr': '공통표준단어영문약어명'
}

# ===== 검색 파라미터 =====
SEARCH_THRESHOLD = 0.3
SEMANTIC_WEIGHT = 0.6
COLBERT_WEIGHT = 0.4