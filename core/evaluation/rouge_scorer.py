"""
ROUGE Scorer - TERTIARY metric (%10)

N-gram overlap metriği.
"""

from typing import Dict

try:
    from rouge_score import rouge_scorer as rouge_lib
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️ rouge-score yüklü değil. ROUGE scoring devre dışı.")


class ROUGEScorer:
    """
    ROUGE metriği - N-gram overlap.
    
    Klasik NLP metriği, kelime örtüşmesini ölçer.
    """
    
    def __init__(self):
        """ROUGE scorer'ı başlat."""
        self.scorer = None
        self.available = ROUGE_AVAILABLE
        
        if not self.available:
            print("⚠️ ROUGEScorer başlatılamadı (rouge-score eksik)")
            return
        
        try:
            self.scorer = rouge_lib.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            print("✅ ROUGEScorer hazır")
            
        except Exception as e:
            print(f"⚠️ ROUGEScorer başlatma hatası: {e}")
            self.scorer = None
            self.available = False
    
    def calculate_score(
        self, 
        generated_answer: str, 
        ideal_answer: str
    ) -> Dict[str, float]:
        """
        ROUGE skorları hesapla.
        
        Args:
            generated_answer: Model tarafından üretilen cevap
            ideal_answer: İdeal/beklenen cevap
            
        Returns:
            dict: ROUGE-1, ROUGE-2, ROUGE-L skorları (0-100 arası)
        """
        default_result = {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }
        
        if self.scorer is None:
            return default_result
        
        try:
            # Boş string kontrolü
            if not generated_answer or not ideal_answer:
                return default_result
            
            scores = self.scorer.score(ideal_answer, generated_answer)
            
            return {
                'rouge1': scores['rouge1'].fmeasure * 100,
                'rouge2': scores['rouge2'].fmeasure * 100,
                'rougeL': scores['rougeL'].fmeasure * 100
            }
            
        except Exception as e:
            print(f"⚠️ ROUGE hesaplama hatası: {e}")
            return default_result
    
    def is_available(self) -> bool:
        """Scorer kullanılabilir mi?"""
        return self.scorer is not None
