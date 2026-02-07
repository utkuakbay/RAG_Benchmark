"""
BERTScore Scorer - SECONDARY metric (%30)

Context-aware token matching.
"""

from typing import Dict, Optional

try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("⚠️ bert-score yüklü değil. BERTScore devre dışı.")


class BERTScorer:
    """
    BERTScore metriği - Context-aware token matching.
    
    ROUGE'dan çok daha güçlü, kelime bağlamını anlar.
    """
    
    def __init__(self, model_type: str = 'bert-base-multilingual-cased'):
        """
        Args:
            model_type: BERT modeli
                - bert-base-multilingual-cased: Çok dilli, Türkçe destekli (ÖNERİLEN)
                - dbmdz/bert-base-turkish-cased: Özel Türkçe model (daha iyi ama yavaş)
                - xlm-roberta-base: Çok dilli XLM model (alternatif)
        """
        self.model_type = model_type
        self.available = BERT_SCORE_AVAILABLE
        
        if self.available:
            print(f"✅ BERTScorer hazır: {model_type}")
        else:
            print("⚠️ BERTScorer başlatılamadı (bert-score eksik)")
    
    def calculate_score(
        self, 
        generated_answer: str, 
        ideal_answer: str,
        lang: str = 'tr'
    ) -> Dict[str, float]:
        """
        BERTScore hesapla.
        
        Args:
            generated_answer: Model tarafından üretilen cevap
            ideal_answer: İdeal/beklenen cevap
            lang: Dil kodu (tr = Türkçe)
            
        Returns:
            dict: precision, recall, f1 skorları (0-100 arası)
        """
        default_result = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        
        if not self.available:
            return default_result
        
        try:
            # Boş string kontrolü
            if not generated_answer or not ideal_answer:
                return default_result
            
            P, R, F1 = bert_score_fn(
                [generated_answer],
                [ideal_answer],
                model_type=self.model_type,
                lang=lang,
                verbose=False,
                rescale_with_baseline=False
            )
            
            return {
                'precision': P.item() * 100,
                'recall': R.item() * 100,
                'f1': F1.item() * 100
            }
            
        except Exception as e:
            print(f"⚠️ BERTScore hesaplama hatası: {e}")
            return default_result
    
    def is_available(self) -> bool:
        """Scorer kullanılabilir mi?"""
        return self.available
