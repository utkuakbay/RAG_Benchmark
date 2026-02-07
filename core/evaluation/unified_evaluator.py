"""
Unified Evaluator - Tüm metrikleri birleştiren ana değerlendirici.

Final Score = 60% Semantic + 30% BERT + 10% ROUGE
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from .semantic_scorer import SemanticScorer
from .bert_scorer import BERTScorer
from .rouge_scorer import ROUGEScorer
from .keyword_scorer import KeywordScorer


@dataclass
class EvaluationResult:
    """Değerlendirme sonucu veri sınıfı."""
    final_score: float = 0.0
    semantic_score: float = 0.0
    bert_f1: float = 0.0
    rouge_l: float = 0.0
    keyword_f1: float = 0.0
    
    # Detaylı skorlar
    bert_precision: float = 0.0
    bert_recall: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    keyword_precision: float = 0.0
    keyword_recall: float = 0.0
    
    # Keyword detayları
    matched_keywords: list = None
    missed_keywords: list = None
    
    # Hata durumu
    error: str = None
    
    def __post_init__(self):
        if self.matched_keywords is None:
            self.matched_keywords = []
        if self.missed_keywords is None:
            self.missed_keywords = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir."""
        return {
            'final_score': self.final_score,
            'semantic_score': self.semantic_score,
            'bert_f1': self.bert_f1,
            'rouge_l': self.rouge_l,
            'keyword_f1': self.keyword_f1,
            'details': {
                'bert': {
                    'precision': self.bert_precision,
                    'recall': self.bert_recall,
                    'f1': self.bert_f1
                },
                'rouge': {
                    'rouge1': self.rouge_1,
                    'rouge2': self.rouge_2,
                    'rougeL': self.rouge_l
                },
                'keyword': {
                    'precision': self.keyword_precision,
                    'recall': self.keyword_recall,
                    'f1': self.keyword_f1,
                    'matched': self.matched_keywords,
                    'missed': self.missed_keywords
                }
            },
            'error': self.error
        }


class UnifiedEvaluator:
    """
    Tüm metrikleri birleştiren ana değerlendirici.
    
    Ağırlıklar:
    - Semantic Similarity: %60 (PRIMARY)
    - BERTScore F1: %30 (SECONDARY)
    - ROUGE-L: %10 (TERTIARY)
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.60,
        bert_weight: float = 0.30,
        rouge_weight: float = 0.10,
        semantic_model: str = 'all-MiniLM-L6-v2',
        bert_model: str = 'bert-base-multilingual-cased'
    ):
        """
        Args:
            semantic_weight: Semantic similarity ağırlığı
            bert_weight: BERTScore ağırlığı
            rouge_weight: ROUGE ağırlığı
            semantic_model: Semantic scorer için model
            bert_model: BERTScore için model
        """
        self.semantic_weight = semantic_weight
        self.bert_weight = bert_weight
        self.rouge_weight = rouge_weight
        
        # Scorer'ları başlat
        print("\n📊 Evaluation modülleri başlatılıyor...")
        
        self.semantic_scorer = None
        self.bert_scorer = None
        self.rouge_scorer = None
        self.keyword_scorer = None
        
        try:
            self.semantic_scorer = SemanticScorer(model_name=semantic_model)
        except Exception as e:
            print(f"⚠️ SemanticScorer başlatılamadı: {e}")
        
        try:
            self.bert_scorer = BERTScorer(model_type=bert_model)
        except Exception as e:
            print(f"⚠️ BERTScorer başlatılamadı: {e}")
        
        try:
            self.rouge_scorer = ROUGEScorer()
        except Exception as e:
            print(f"⚠️ ROUGEScorer başlatılamadı: {e}")
        
        try:
            self.keyword_scorer = KeywordScorer()
        except Exception as e:
            print(f"⚠️ KeywordScorer başlatılamadı: {e}")
        
        print("✅ UnifiedEvaluator hazır\n")
    
    def evaluate(
        self, 
        generated_answer: str, 
        ideal_answer: str
    ) -> EvaluationResult:
        """
        Tüm metrikleri hesapla ve weighted average al.
        
        Args:
            generated_answer: Model tarafından üretilen cevap
            ideal_answer: İdeal/beklenen cevap
            
        Returns:
            EvaluationResult objesi
        """
        result = EvaluationResult()
        
        try:
            # Semantic Similarity (PRIMARY)
            if self.semantic_scorer and self.semantic_scorer.is_available():
                try:
                    result.semantic_score = self.semantic_scorer.calculate_score(
                        generated_answer, ideal_answer
                    )
                except Exception as e:
                    print(f"⚠️ Semantic scoring hatası: {e}")
            
            # BERTScore (SECONDARY)
            if self.bert_scorer and self.bert_scorer.is_available():
                try:
                    bert_scores = self.bert_scorer.calculate_score(
                        generated_answer, ideal_answer
                    )
                    result.bert_f1 = bert_scores.get('f1', 0.0)
                    result.bert_precision = bert_scores.get('precision', 0.0)
                    result.bert_recall = bert_scores.get('recall', 0.0)
                except Exception as e:
                    print(f"⚠️ BERT scoring hatası: {e}")
            
            # ROUGE (TERTIARY)
            if self.rouge_scorer and self.rouge_scorer.is_available():
                try:
                    rouge_scores = self.rouge_scorer.calculate_score(
                        generated_answer, ideal_answer
                    )
                    result.rouge_l = rouge_scores.get('rougeL', 0.0)
                    result.rouge_1 = rouge_scores.get('rouge1', 0.0)
                    result.rouge_2 = rouge_scores.get('rouge2', 0.0)
                except Exception as e:
                    print(f"⚠️ ROUGE scoring hatası: {e}")
            
            # Keyword Matching (SUPPLEMENTARY)
            if self.keyword_scorer and self.keyword_scorer.is_available():
                try:
                    keyword_scores = self.keyword_scorer.calculate_score(
                        generated_answer, ideal_answer
                    )
                    result.keyword_f1 = keyword_scores.get('f1', 0.0)
                    result.keyword_precision = keyword_scores.get('precision', 0.0)
                    result.keyword_recall = keyword_scores.get('recall', 0.0)
                    result.matched_keywords = keyword_scores.get('matched_keywords', [])
                    result.missed_keywords = keyword_scores.get('missed_keywords', [])
                except Exception as e:
                    print(f"⚠️ Keyword scoring hatası: {e}")
            
            # WEIGHTED FINAL SCORE
            result.final_score = self._calculate_weighted_score(result)
            
            return result
            
        except Exception as e:
            print(f"⚠️ UnifiedEvaluator critical error: {e}")
            result.error = str(e)
            return result
    
    def _calculate_weighted_score(self, result: EvaluationResult) -> float:
        """
        Ağırlıklı final skoru hesapla.
        
        Eğer bazı metrikler çalışmadıysa, ağırlıkları normalize et.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        
        # Semantic
        if result.semantic_score > 0:
            weighted_sum += result.semantic_score * self.semantic_weight
            total_weight += self.semantic_weight
        
        # BERT
        if result.bert_f1 > 0:
            weighted_sum += result.bert_f1 * self.bert_weight
            total_weight += self.bert_weight
        
        # ROUGE
        if result.rouge_l > 0:
            weighted_sum += result.rouge_l * self.rouge_weight
            total_weight += self.rouge_weight
        
        # Normalize
        if total_weight > 0:
            return weighted_sum / total_weight
        
        return 0.0
    
    def get_available_metrics(self) -> Dict[str, bool]:
        """Hangi metriklerin kullanılabilir olduğunu döndür."""
        return {
            'semantic': self.semantic_scorer is not None and self.semantic_scorer.is_available(),
            'bert': self.bert_scorer is not None and self.bert_scorer.is_available(),
            'rouge': self.rouge_scorer is not None and self.rouge_scorer.is_available(),
            'keyword': self.keyword_scorer is not None and self.keyword_scorer.is_available()
        }
    
    def print_result(self, result: EvaluationResult) -> None:
        """Sonucu formatlanmış şekilde yazdır."""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║              EVALUATION RESULTS                          ║
╠══════════════════════════════════════════════════════════╣
║ FINAL SCORE:      {result.final_score:>6.1f}/100                        ║
║                                                          ║
║ Primary Metrics:                                         ║
║   • Semantic:     {result.semantic_score:>6.1f}/100                        ║
║   • BERTScore:    {result.bert_f1:>6.1f}/100                        ║
║   • ROUGE-L:      {result.rouge_l:>6.1f}/100                        ║
║                                                          ║
║ Secondary Metrics:                                       ║
║   • Keyword F1:   {result.keyword_f1:>6.1f}/100                        ║
╚══════════════════════════════════════════════════════════╝
""")
