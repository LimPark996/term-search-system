import logging
from openai import OpenAI
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

class QueryPreprocessor:
    """Phase 1: OpenAI로 자연어 질의를 정제된 검색어로 변환"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def refine_query(self, natural_query: str) -> str:
        """자연어 질의를 정제된 검색어로 변환"""
        prompt = f"""다음 자연어 질의에서 핵심 키워드를 추출하고 명확한 검색어로 변환해주세요.

입력 질의: "{natural_query}"

조건:
- 구어체, 은어, 줄임말을 표준 용어로 변환
- 핵심 개념을 명확히 식별
- 불필요한 조사, 감정 표현 제거
- 검색에 적합한 키워드 조합으로 구성
- 20-30자 내외로 간결하게 작성

예시:
"고객 번호 관리하는 거 어떻게 해?" → "고객번호 관리 방법"
"직원 정보 저장할 때 뭐가 필요해?" → "직원정보 저장 항목"
"웹사이트에서 다른 시스템이랑 데이터 주고받는 거" → "웹사이트 시스템간 데이터 교환"

변환된 검색어만 출력하세요:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            
            if response and response.choices:
                refined_query = response.choices[0].message.content.strip()
                logger.info(f"쿼리 정제: '{natural_query}' → '{refined_query}'")
                return refined_query
            else:
                logger.warning("OpenAI 응답 없음, 원본 쿼리 사용")
                return natural_query
                
        except Exception as e:
            logger.error(f"쿼리 정제 실패: {e}")
            return natural_query
