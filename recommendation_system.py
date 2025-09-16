import logging
from data_loader import DataLoader
from query_preprocessor import QueryPreprocessor
from search_engine import HybridSearchEngine
from ai_reranker import AIReranker

logger = logging.getLogger(__name__)

class TermRecommendationSystem:
    """4단계 파이프라인 통합 시스템"""
    
    def __init__(self):
        logger.info("시스템 초기화 시작...")
        
        # Phase 1: 쿼리 전처리기
        self.query_preprocessor = QueryPreprocessor()
        
        # 데이터 로드
        self.data_loader = DataLoader()
        self.terms = self.data_loader.load_from_sheets()
        
        if not self.terms:
            raise Exception("용어 데이터를 로드할 수 없습니다.")
        
        # Phase 2 & 3: 하이브리드 검색 엔진
        self.search_engine = HybridSearchEngine(self.terms)
        
        # Phase 4: AI 재순위화
        self.ai_reranker = AIReranker()
        
        logger.info(f"시스템 초기화 완료: {len(self.terms)}개 용어")

    def recommend(self, natural_query: str) -> dict[str, any]:
        """완전한 4단계 파이프라인 실행"""
        logger.info(f"추천 파이프라인 시작: '{natural_query}'")
        
        try:
            # Phase 1: 쿼리 전처리
            logger.info("Phase 1: 쿼리 전처리")
            refined_query = self.query_preprocessor.refine_query(natural_query)
            
            # Phase 2 & 3: 하이브리드 검색
            logger.info("Phase 2-3: 하이브리드 검색")
            candidates = self.search_engine.search(refined_query, k=20)
            
            if not candidates:
                return {
                    'success': False,
                    'message': '검색 조건에 맞는 용어를 찾을 수 없습니다.',
                    'original_query': natural_query,
                    'refined_query': refined_query
                }
            
            # Phase 4: AI 기반 최종 선별
            logger.info("Phase 4: AI 기반 최종 선별")
            final_recommendations = self.ai_reranker.rerank_and_explain(
                natural_query, refined_query, candidates
            )
            
            return {
                'success': True,
                'original_query': natural_query,
                'refined_query': refined_query,
                'candidates_count': len(candidates),
                'final_recommendations': final_recommendations
            }
            
        except Exception as e:
            logger.error(f"추천 파이프라인 실패: {e}")
            return {
                'success': False,
                'message': f'시스템 오류: {str(e)}',
                'original_query': natural_query,
                'error': str(e)
            }
