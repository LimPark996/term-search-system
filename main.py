import logging
from recommendation_system import TermRecommendationSystem

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    try:
        print("=== 용어 추천 시스템 ===")
        print("시스템 초기화 중...")
        
        system = TermRecommendationSystem()
        print("초기화 완료!\n")
        
        while True:
            print("=" * 60)
            natural_query = input("자연어 질의를 입력하세요 (종료: 'quit'): ").strip()
            
            if natural_query.lower() in ['quit', 'exit', '종료']:
                print("시스템을 종료합니다.")
                break
            
            if not natural_query:
                continue
            
            # 4단계 파이프라인 실행
            print(f"\n🔍 처리 중: '{natural_query}'")
            print("-" * 60)
            
            result = system.recommend(natural_query)
            
            if result['success']:
                print("✅ 추천 완료!")
                print(f"\n📝 원본 질의: {result['original_query']}")
                print(f"🎯 정제된 검색어: {result['refined_query']}")
                print(f"🔍 후보 개수: {result['candidates_count']}개")
                
                recommendations = result['final_recommendations']
                if recommendations:
                    print(f"\n🏆 AI 최종 추천 ({len(recommendations)}개):")
                    print("=" * 70)
                    
                    for rec in recommendations:
                        print(f"\n{rec.rank}위. {rec.term.name}")
                        print(f"    설명: {rec.term.description}")
                        print(f"    점수: {rec.final_score:.3f} (의미: {rec.semantic_score:.3f}, 키워드: {rec.colbert_score:.3f})")
                        if rec.matched_tokens:
                            print(f"    매칭 키워드: {', '.join(rec.matched_tokens)}")
                        if rec.ai_reasoning:
                            print(f"    AI 추천 이유: {rec.ai_reasoning}")
                
            else:
                print("❌ 추천 실패!")
                print(f"사유: {result['message']}")
            
            print("\n")
    
    except KeyboardInterrupt:
        print("\n\n시스템이 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
