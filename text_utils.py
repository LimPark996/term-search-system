import re
from typing import List
from konlpy.tag import Okt

class TextUtils:
    def __init__(self):
        self.okt = Okt()
    
    def tokenize(self, text: str) -> List[str]:
        """한국어 텍스트 토큰화"""
        # 1) 정규식 기반
        basic_tokens = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        basic_tokens = [token for token in basic_tokens if len(token) > 1]
        
        # 2) 형태소 분석
        pos_result = self.okt.pos(text)
        morpheme_tokens = [
            word for word, pos in pos_result 
            if pos in ['Noun', 'Verb', 'Adjective', 'Alpha', 'Number'] and len(word) > 1
        ]
        
        # 3) 결합 (중복 제거)
        all_tokens = list(set(basic_tokens + morpheme_tokens))
        return all_tokens