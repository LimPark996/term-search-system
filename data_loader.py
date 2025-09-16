import logging
import gspread
import pandas as pd
from typing import List
from models import Term
from config import SPREADSHEET_ID, COLUMN_MAPPING

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.gc = self._setup_gsheets()
    
    def _setup_gsheets(self):
        try:
            logger.info("Google Sheets 연결 중...")
            return gspread.oauth(scopes=[
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ],
            credentials_filename='credentials.json',  # 현재 폴더의 파일 사용
            authorized_user_filename='token.json'     # 현재 폴더에 토큰 저장
            )
        except Exception as e:
            logger.error(f"Google Sheets 인증 실패: {e}")
            return None
    
    def load_from_sheets(self) -> List[Term]:
        """Google Sheets에서 용어 로드"""
        if not self.gc:
            logger.warning("Google Sheets 연결 실패, Excel 파일 사용")
            return []
        
        try:
            spreadsheet = self.gc.open_by_key(SPREADSHEET_ID)
            worksheet = spreadsheet.worksheets()[0]
            
            all_values = worksheet.get_all_values()
            if not all_values:
                return []
            
            headers = all_values[0]
            rows = all_values[1:]
            df = pd.DataFrame(rows, columns=headers)
            df = df.dropna(how='all')
            
            return self._convert_to_terms(df)
            
        except Exception as e:
            logger.error(f"Google Sheets 로드 실패: {e}")
            return []
       
    def _convert_to_terms(self, df: pd.DataFrame) -> List[Term]:
        """DataFrame을 Term 객체로 변환"""
        terms = []
        for idx, row in df.iterrows():
            term = Term(
                id=str(idx),
                name=str(row.get(COLUMN_MAPPING['term_name'], '')).strip(),
                description=str(row.get(COLUMN_MAPPING['term_desc'], '')).strip(),
                abbreviation=str(row.get(COLUMN_MAPPING['term_abbr'], '')).strip(),
                domain=str(row.get(COLUMN_MAPPING.get('domain', ''), '')).strip(),
                metadata=row.to_dict()
            )
            
            if term.name and term.description:
                terms.append(term)
        
        logger.info(f"용어 {len(terms)}개 변환 완료")
        return terms
    