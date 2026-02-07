"""
Keyword Scorer - Faktüel doğruluk kontrolü

Kritik anahtar kelimelerin yakalanma oranını ölçer.
"""

import re
from typing import Dict, Set, List


class KeywordScorer:
    """
    Kritik anahtar kelimelerin yakalanma oranı.
    
    Faktüel sorular için önemli (tarih, isim, sayı).
    """
    
    def __init__(self):
        """Keyword scorer'ı başlat."""
        # Türkçe stop words (yaygın kelimeler - filtrelenecek)
        self.stop_words = {
            'bir', 've', 'veya', 'için', 'ile', 'ise', 'olan', 'bu', 'şu', 'o',
            'da', 'de', 'ta', 'te', 'mi', 'mı', 'mu', 'mü', 'gibi', 'kadar', 'daha',
            'çok', 'az', 'en', 'ama', 'fakat', 'ancak', 'lakin', 'ne', 'nasıl', 
            'neden', 'niçin', 'nerede', 'hangi', 'kim', 'var', 'yok', 'ben', 'sen', 
            'biz', 'siz', 'onlar', 'benim', 'senin', 'bizim', 'sizin', 'onların',
            'bu', 'şu', 'bunun', 'şunun', 'bunlar', 'şunlar', 'olan', 'olarak',
            'oldu', 'olmuş', 'olacak', 'eder', 'etti', 'etmiş', 'edecek',
            'dir', 'dır', 'dur', 'dür', 'tir', 'tır', 'tur', 'tür',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'to', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there'
        }
        
        print("✅ KeywordScorer hazır")
    
    def extract_keywords(self, text: str, min_length: int = 3) -> Set[str]:
        """
        Metinden anahtar kelimeleri çıkar.
        
        Args:
            text: Metin
            min_length: Minimum kelime uzunluğu
            
        Returns:
            Anahtar kelime seti
        """
        try:
            if not text:
                return set()
            
            # Temizle ve küçült
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Stop words ve kısa kelimeleri çıkar
            keywords = {
                w for w in words 
                if len(w) >= min_length and w not in self.stop_words
            }
            
            return keywords
            
        except Exception as e:
            print(f"⚠️ Keyword extraction hatası: {e}")
            return set()
    
    def calculate_score(
        self, 
        generated_answer: str, 
        ideal_answer: str
    ) -> Dict[str, any]:
        """
        Anahtar kelime yakalama oranı.
        
        Args:
            generated_answer: Model tarafından üretilen cevap
            ideal_answer: İdeal/beklenen cevap
            
        Returns:
            dict: Precision, Recall, F1 ve detaylar
        """
        default_result = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'matched_keywords': [],
            'missed_keywords': []
        }
        
        try:
            ideal_keywords = self.extract_keywords(ideal_answer)
            generated_keywords = self.extract_keywords(generated_answer)
            
            # Eğer ideal'de keyword yoksa, perfect score
            if not ideal_keywords:
                return {
                    'precision': 100.0,
                    'recall': 100.0,
                    'f1': 100.0,
                    'matched_keywords': [],
                    'missed_keywords': []
                }
            
            # Kesişim
            matched = ideal_keywords & generated_keywords
            missed = ideal_keywords - generated_keywords
            
            # Precision: Üretilen kelimelerin kaçı doğru?
            precision = (len(matched) / len(generated_keywords) * 100) if generated_keywords else 0.0
            
            # Recall: İdeal kelimelerin kaçı yakalandı?
            recall = (len(matched) / len(ideal_keywords) * 100) if ideal_keywords else 0.0
            
            # F1
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'matched_keywords': list(matched),
                'missed_keywords': list(missed)
            }
            
        except Exception as e:
            print(f"⚠️ Keyword scoring hatası: {e}")
            return default_result
    
    def is_available(self) -> bool:
        """Scorer kullanılabilir mi?"""
        return True  # Her zaman kullanılabilir
