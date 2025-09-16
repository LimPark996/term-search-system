from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

@dataclass
class Term:
    id: str
    name: str
    description: str
    abbreviation: str
    domain: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SearchResult:
    term: Term
    semantic_score: float
    colbert_score: float
    final_score: float
    rank: int
    matched_tokens: Optional[List[str]] = None
    ai_reasoning: Optional[str] = None
