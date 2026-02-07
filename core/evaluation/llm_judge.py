"""
LLM Judge - Cross-validation için LLM tabanlı değerlendirme.

Primary metriklerin güvenilirliğini doğrulamak için %20 sample değerlendirir.
"""

import random
from typing import Dict, Any, List, Optional

try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy yüklü değil. Correlation analizi devre dışı.")


class LLMJudge:
    """
    LLM-as-a-Judge - Cross-validation için.
    
    Tüm soruları değil, sadece random %20'sini değerlendirerek
    primary metriklerin güvenilirliğini test eder.
    """
    
    # Değerlendirme prompt'u
    JUDGE_PROMPT = """Bir değerlendirici olarak, verilen soruya modelin cevabının kalitesini puanla.

SORU: {question}

İDEAL CEVAP: {ideal_answer}

MODEL CEVABI: {model_answer}

Aşağıdaki kriterlere göre 0-100 arası puan ver:
1. Doğruluk (40%): Cevap faktüel olarak doğru mu?
2. Tamlık (30%): İdeal cevaptaki tüm bilgiler var mı?
3. Alakalılık (20%): Cevap soruyla alakalı mı?
4. Netlik (10%): Cevap açık ve anlaşılır mı?

SADECE bir sayı döndür (0-100 arası). Açıklama yapma.
PUAN:"""

    def __init__(
        self, 
        llm: Optional[Any] = None,
        sample_rate: float = 0.20
    ):
        """
        Args:
            llm: Değerlendirme için kullanılacak LLM (None ise devre dışı)
            sample_rate: Değerlendirilecek oran (default: %20)
        """
        self.llm = llm
        self.sample_rate = sample_rate
        self.available = llm is not None
        
        if self.available:
            print(f"✅ LLMJudge hazır (sample rate: %{sample_rate*100:.0f})")
        else:
            print("⚠️ LLMJudge: LLM sağlanmadı, devre dışı")
    
    def set_llm(self, llm: Any) -> None:
        """LLM'i sonradan ayarla."""
        self.llm = llm
        self.available = llm is not None
        if self.available:
            print(f"✅ LLMJudge LLM ayarlandı")
    
    def evaluate_single(
        self,
        question: str,
        ideal_answer: str,
        model_answer: str
    ) -> Dict[str, Any]:
        """
        Tek bir cevabı değerlendir.
        
        Args:
            question: Soru
            ideal_answer: İdeal cevap
            model_answer: Model cevabı
            
        Returns:
            dict: score ve detaylar
        """
        if not self.available:
            return {"score": 0.0, "error": "LLM not available"}
        
        try:
            # Prompt oluştur
            prompt = self.JUDGE_PROMPT.format(
                question=question,
                ideal_answer=ideal_answer,
                model_answer=model_answer
            )
            
            # LLM'e sor
            response = self.llm.invoke(prompt)
            
            # Response'dan sayıyı çıkar
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Sayıyı bul
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', response_text)
            
            if numbers:
                score = float(numbers[0])
                score = max(0.0, min(100.0, score))  # 0-100 sınırla
            else:
                score = 0.0
            
            return {
                "score": score,
                "raw_response": response_text
            }
            
        except Exception as e:
            print(f"⚠️ LLM Judge değerlendirme hatası: {e}")
            return {"score": 0.0, "error": str(e)}
    
    def cross_validate(
        self, 
        results: List[Dict[str, Any]],
        sample_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Primary metrikleri doğrulamak için sample-based değerlendirme.
        
        Args:
            results: Tüm test sonuçları
                - Her biri: {question, ideal_answer, answer, final_score, ...}
            sample_rate: Değerlendirilecek oran (None ise self.sample_rate)
            
        Returns:
            dict: Korelasyon analizi sonuçları
        """
        if not self.available:
            return {
                "error": "LLM not available",
                "correlation": 0.0,
                "sample_size": 0
            }
        
        if not results:
            return {
                "error": "No results to validate",
                "correlation": 0.0,
                "sample_size": 0
            }
        
        rate = sample_rate if sample_rate is not None else self.sample_rate
        
        # Random sample seç
        total = len(results)
        sample_size = max(2, int(total * rate))  # En az 2 sample
        sample_size = min(sample_size, total)  # Total'dan fazla olamaz
        
        sample_indices = random.sample(range(total), sample_size)
        
        print(f"\n🎯 LLM Judge Cross-Validation")
        print(f"   Total: {total} soru")
        print(f"   Sample: {sample_size} soru (%{rate*100:.0f})")
        
        correlations = []
        
        for idx in sample_indices:
            result = results[idx]
            
            try:
                # LLM Judge puanı
                llm_result = self.evaluate_single(
                    question=result.get('question', ''),
                    ideal_answer=result.get('ideal_answer', ''),
                    model_answer=result.get('answer', '')
                )
                llm_score = llm_result.get('score', 0)
                
                # Primary metric puanı
                primary_score = result.get('final_score', 0)
                
                # Farkı kaydet
                diff = abs(llm_score - primary_score)
                correlations.append({
                    'question_id': idx,
                    'llm_score': llm_score,
                    'primary_score': primary_score,
                    'diff': diff
                })
                
                print(f"   Soru {idx+1}: LLM={llm_score:.1f}, Primary={primary_score:.1f}, Diff={diff:.1f}")
                
            except Exception as e:
                print(f"   ⚠️ Soru {idx+1} değerlendirilemedi: {e}")
                continue
        
        if len(correlations) < 2:
            print("⚠️ Yeterli sample değerlendirilemedi")
            return {
                'avg_diff': 0.0,
                'max_diff': 0.0,
                'correlation': 0.0,
                'p_value': 1.0,
                'sample_size': len(correlations),
                'details': correlations
            }
        
        # İstatistikler
        avg_diff = sum(c['diff'] for c in correlations) / len(correlations)
        max_diff = max(c['diff'] for c in correlations)
        
        # Pearson correlation
        correlation = 0.0
        p_value = 1.0
        
        if SCIPY_AVAILABLE and len(correlations) >= 2:
            try:
                llm_scores = [c['llm_score'] for c in correlations]
                primary_scores = [c['primary_score'] for c in correlations]
                correlation, p_value = pearsonr(llm_scores, primary_scores)
            except Exception as e:
                print(f"⚠️ Korelasyon hesaplama hatası: {e}")
        
        print(f"\n📊 KORELASYON ANALİZİ:")
        print(f"   Ortalama Fark: {avg_diff:.1f} puan")
        print(f"   Maksimum Fark: {max_diff:.1f} puan")
        print(f"   Pearson r: {correlation:.3f} (p={p_value:.4f})")
        
        if correlation > 0.85:
            print(f"   ✅ ÇOK YÜKSEK korelasyon! Primary metrics güvenilir.")
        elif correlation > 0.70:
            print(f"   ✅ İYİ korelasyon. Primary metrics kullanılabilir.")
        elif correlation > 0.50:
            print(f"   ⚠️ ORTA korelasyon. Daha fazla sample gerekebilir.")
        else:
            print(f"   ❌ DÜŞÜK korelasyon! Primary metrics'i gözden geçir.")
        
        return {
            'avg_diff': avg_diff,
            'max_diff': max_diff,
            'correlation': correlation,
            'p_value': p_value,
            'sample_size': len(correlations),
            'details': correlations
        }
    
    def is_available(self) -> bool:
        """Judge kullanılabilir mi?"""
        return self.available
