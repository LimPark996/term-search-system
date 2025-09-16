import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple
from openai import OpenAI
import logging
from models import Term, SearchResult
from text_utils import TextUtils
from config import OPENAI_API_KEY, SEMANTIC_WEIGHT, COLBERT_WEIGHT, SEARCH_THRESHOLD

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """Phase 2 & 3: 임베딩 생성 및 하이브리드 검색"""
    
    def __init__(self, terms: List[Term]):
        self.terms = terms
        self.text_utils = TextUtils()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # 캐시 파일 경로
        self.semantic_embeddings_file = "semantic_embeddings.npy"
        self.token_metadata_file = "token_metadata.pkl"
        self.faiss_index_file = "faiss_index.bin"
        
        # 인덱스 및 메타데이터
        self.semantic_index = None
        self.token_metadata = []
        
        self._build_or_load_index()
    
    def _build_or_load_index(self):
        """인덱스 구축 또는 로드"""
        if self._cache_exists():
            logger.info("캐시된 인덱스 로드 중...")
            self._load_from_cache()
        else:
            logger.info("새 인덱스 구축 중...")
            self._build_fresh_index()
            self._save_to_cache()
    
    def _cache_exists(self) -> bool:
        """캐시 파일 존재 여부 확인"""
        return (os.path.exists(self.semantic_embeddings_file) and 
                os.path.exists(self.token_metadata_file) and
                os.path.exists(self.faiss_index_file))
    
    def _build_fresh_index(self):
        """새로운 인덱스 구축"""
        # 1) 의미적 임베딩 생성
        logger.info("의미적 임베딩 생성 중...")
        semantic_texts = [f"{term.name}: {term.description}" for term in self.terms]
        semantic_embeddings = self._generate_embeddings(semantic_texts)
        
        # 2) FAISS 인덱스 생성
        if semantic_embeddings:
            embeddings_array = np.array(semantic_embeddings, dtype=np.float32)
            dimension = embeddings_array.shape[1]
            
            self.semantic_index = faiss.IndexFlatIP(dimension)
            self.semantic_index.add(embeddings_array)
            logger.info(f"FAISS 인덱스 생성: {embeddings_array.shape}")
        
        # 3) 토큰 메타데이터 생성
        logger.info("토큰 메타데이터 생성 중...")

        # 3-1) 모든 용어의 토큰 수집 및 매핑 정보 생성
        all_tokens = []
        token_mapping = [] # (term_index, start_idx, end_idx) 저장
        
        for i, term in enumerate(self.terms):
            combined_text = semantic_texts[i]
            tokens = self.text_utils.tokenize(combined_text)
            
            if tokens:
                start_idx = len(all_tokens)
                all_tokens.extend(tokens)
                end_idx = len(all_tokens)
                token_mapping.append({
                    'term_index': i,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'tokens': tokens,
                    'combined_text': combined_text
                })
            else:
                token_mapping.append({
                    'term_index': i,
                    'start_idx': -1,
                    'end_idx': -1,
                    'tokens': [],
                    'combined_text': combined_text
                })
        
        logger.info(f"전체 토큰 수집 완료: {len(all_tokens)}개 토큰")

        # 3-2) 모든 토큰의 임베딩을 한번에 생성
        all_token_embeddings = []
        if all_tokens:
            logger.info("모든 토큰의 임베딩 일괄 생성 중...")
            all_token_embeddings = self._generate_embeddings(all_tokens)

        # 3-3) 각 용어별로 토큰 임베딩 분배
        self.token_metadata = []
        for mapping_info in token_mapping:
            term_index = mapping_info['term_index']
            start_idx = mapping_info['start_idx']
            end_idx = mapping_info['end_idx']

            if start_idx >= 0: # 토큰이 있는 경우
                token_embeddings = all_token_embeddings[start_idx:end_idx]
            else: # 토큰이 없는 경우
                token_embeddings = []

            self.token_metadata.append({
                'term_index': term_index,
                'tokens': mapping_info['tokens'],
                'token_embeddings': token_embeddings,
                'combined_text': mapping_info['combined_text'],
                'semantic_embedding': semantic_embeddings[term_index]
            })

        logger.info(f"토큰 메타데이터 생성 완료: {len(self.token_metadata)}개")

    def _generate_embeddings(self, texts: List[str], batch_size: int = 10) -> List[np.ndarray]:
        """OpenAI 임베딩 생성"""
        if not texts:
            return []
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"임베딩 배치 {batch_num}/{total_batches} 처리 중...")
            
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model="text-embedding-3-small",
                    dimensions=1536
                )
                
                for emb_data in response.data:
                    embedding = np.array(emb_data.embedding, dtype=np.float32)
                    # L2 정규화
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    embeddings.append(embedding)
                    
            except Exception as e:
                logger.error(f"임베딩 생성 실패 (배치 {batch_num}): {e}")
                # 실패시 영벡터 추가
                for _ in batch_texts:
                    embeddings.append(np.zeros(1536, dtype=np.float32))
        
        logger.info(f"임베딩 생성 완료: {len(embeddings)}개")
        return embeddings
    
    def _save_to_cache(self):
        """캐시에 저장"""
        try:
            logger.info("캐시 저장 시작...")
            
            # 1. 토큰 메타데이터 저장 (핵심 데이터)
            with open(self.token_metadata_file, 'wb') as f:
                pickle.dump(self.token_metadata, f)
            logger.info("토큰 메타데이터 저장 완료")
            
            # 2. FAISS 인덱스 저장
            if self.semantic_index:
                faiss.write_index(self.semantic_index, self.faiss_index_file)
            logger.info("FAISS 인덱스 저장 완료")
            
            # 3. 의미적 임베딩 저장 (메타데이터에서 추출)
            semantic_embeddings = [data['semantic_embedding'] for data in self.token_metadata]
            np.save(self.semantic_embeddings_file, np.array(semantic_embeddings))
            logger.info("의미적 임베딩 저장 완료")
            
            logger.info("캐시 저장 완료")
            
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_from_cache(self):
        """캐시에서 로드"""
        try:
            # FAISS 인덱스 로드
            self.semantic_index = faiss.read_index(self.faiss_index_file)
            
            # 토큰 메타데이터 로드
            with open(self.token_metadata_file, 'rb') as f:
                self.token_metadata = pickle.load(f)
            
            logger.info(f"캐시 로드 완료: {len(self.token_metadata)}개 용어")
        except Exception as e:
            logger.error(f"캐시 로드 실패: {e}")
            self._build_fresh_index()
    
    def search(self, refined_query: str, k: int = 20) -> List[SearchResult]:
        """Phase 3: 하이브리드 검색 수행"""
        # 쿼리 전처리
        query_tokens = self.text_utils.tokenize(refined_query)
        
        # 쿼리 임베딩 생성
        query_semantic_emb = self._generate_embeddings([refined_query])[0]
        query_token_embs = []
        if query_tokens:
            query_token_embs = self._generate_embeddings(query_tokens)
        
        # 후보 스코어링
        candidates = []
        
        for term_data in self.token_metadata:
            term = self.terms[term_data['term_index']]
            
            # 1) 의미적 점수 계산
            term_semantic_emb = term_data['semantic_embedding']
            semantic_score = self._cosine_similarity(query_semantic_emb, term_semantic_emb)
            
            # 2) ColBERT 점수 계산
            colbert_score = self._calculate_colbert_score(query_token_embs, term_data['token_embeddings'])
            
            # 3) 하이브리드 점수 계산
            final_score = SEMANTIC_WEIGHT * semantic_score + COLBERT_WEIGHT * colbert_score

            # 임계값 필터링
            if final_score >= SEARCH_THRESHOLD:
                matched_tokens = [
                    token for token in query_tokens 
                    if token in term_data['tokens']
                ]
                
                result = SearchResult(
                    term=term,
                    semantic_score=semantic_score,
                    colbert_score=colbert_score,
                    final_score=final_score,
                    rank=0,
                    matched_tokens=matched_tokens
                )
                candidates.append(result)
        
        # 정렬 및 상위 k개 선택
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # 순위 부여
        for rank, candidate in enumerate(candidates[:k], 1):
            candidate.rank = rank
        
        logger.info(f"하이브리드 검색 완료: {len(candidates[:k])}개 후보")
        return candidates[:k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
    
    def _calculate_colbert_score(self, query_embeds: List[np.ndarray], doc_embeds: List[np.ndarray]) -> float:
        """ColBERT 스타일 점수 계산"""
        if not query_embeds or not doc_embeds:
            return 0.0
        
        total_score = 0
        for query_embed in query_embeds:
            max_sim = max(
                self._cosine_similarity(query_embed, doc_embed) 
                for doc_embed in doc_embeds
            )
            total_score += max_sim
        
        return total_score / len(query_embeds)
