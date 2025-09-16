from typing import List
from openai import OpenAI
import logging
from models import SearchResult
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

class AIReranker:
    """Phase 4: GPT로 최종 추천 및 이유 생성"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def rerank_and_explain(self, original_query: str, refined_query: str, 
                      candidates: List[SearchResult]) -> List[SearchResult]:
    
        # 후보 목록을 인덱스와 함께 포맷팅
        candidates_text = self._format_candidates_with_index(candidates[:10])
        
        prompt = f"""사용자 질의: '{original_query}'
    정제된 검색어: '{refined_query}'

    후보 용어들:
    {candidates_text}

    위 후보 용어들 중에서 사용자 질의에 가장 적합한 용어 3개를 선택하세요.

    다음 JSON 형식으로만 응답하세요:
    {{
    "recommendations": [
        {{
        "candidate_index": 0,
        "rank": 1,
        "reasoning": "구체적인 추천 이유"
        }},
        {{
        "candidate_index": 2,
        "rank": 2, 
        "reasoning": "구체적인 추천 이유"
        }},
        {{
        "candidate_index": 5,
        "rank": 3,
        "reasoning": "구체적인 추천 이유"
        }}
    ]
    }}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            if response and response.choices:
                ai_response = response.choices[0].message.content.strip()
                return self._parse_structured_response(ai_response, candidates)
            else:
                return candidates[:3]
                
        except Exception as e:
            logger.error(f"AI 재순위화 실패: {e}")
            return candidates[:3]
    
    def _format_candidates_with_index(self, candidates: List[SearchResult]) -> str:
        """후보 목록을 인덱스와 함께 포맷팅"""
        formatted = []
        for i, candidate in enumerate(candidates):
            text = f"{i}. {candidate.term.name}"
            text += f"\n   설명: {candidate.term.description}"
            text += f"\n   점수: {candidate.final_score:.3f}"
            if candidate.matched_tokens:
                text += f"\n   매칭 키워드: {', '.join(candidate.matched_tokens)}"
            formatted.append(text)
        
        return "\n\n".join(formatted)
    
    def _parse_structured_response(self, ai_response: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """JSON 구조화된 응답 파싱"""
        try:
            import json
            
            # JSON 추출 (GPT가 추가 텍스트를 넣을 수 있어서)
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("JSON 형식을 찾을 수 없음, 상위 3개 반환")
                return candidates[:3]
            
            json_str = ai_response[json_start:json_end]
            data = json.loads(json_str)
            
            final_results = []
            recommendations = data.get('recommendations', [])
            
            for rec in recommendations:
                candidate_index = rec.get('candidate_index')
                reasoning = rec.get('reasoning', 'AI 추천')
                
                # 인덱스 유효성 검사
                if (candidate_index is not None and 
                    0 <= candidate_index < len(candidates)):
                    
                    candidate = candidates[candidate_index]
                    candidate.ai_reasoning = reasoning
                    final_results.append(candidate)
            
            # 순위 재부여
            for rank, result in enumerate(final_results, 1):
                result.rank = rank
            
            logger.info(f"구조화된 파싱 성공: {len(final_results)}개")
            return final_results[:3]
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            return candidates[:3]
        except Exception as e:
            logger.error(f"구조화된 응답 파싱 실패: {e}")
            return candidates[:3]