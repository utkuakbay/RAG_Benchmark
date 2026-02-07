"""
Semantic Similarity Scorer - PRIMARY metric (%60)

Anlamsal benzerlik hesaplama modülü.
"""

from typing import List, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers yüklü değil. Semantic scoring devre dışı.")


class SemanticScorer:
    """
    Anlamsal benzerlik puanlayıcı (PRIMARY metric).
    
    İki cevabın anlamsal benzerliğini cosine similarity ile ölçer.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Args:
            model_name: Embedding modeli
                - all-MiniLM-L6-v2: Hızlı, hafif (önerilen)
                - all-mpnet-base-v2: Daha güçlü ama yavaş
        """
        self.model_name = model_name
        self.model = None
        self.device = 'cpu'
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("⚠️ SemanticScorer başlatılamadı (sentence-transformers eksik)")
            return
        
        try:
            # GPU varsa kullan
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(model_name, device=self.device)
            print(f"✅ SemanticScorer yüklendi: {model_name} (device: {self.device})")
            
        except Exception as e:
            print(f"⚠️ SemanticScorer yükleme hatası: {e}")
            print(f"   CPU ile tekrar deneniyor...")
            
            try:
                self.device = 'cpu'
                self.model = SentenceTransformer(model_name, device='cpu')
                print(f"✅ SemanticScorer yüklendi: {model_name} (device: cpu)")
            except Exception as e2:
                print(f"❌ SemanticScorer yüklenemedi: {e2}")
                self.model = None
    
    def calculate_score(self, generated_answer: str, ideal_answer: str) -> float:
        """
        İki cevabın anlamsal benzerliğini hesapla.
        
        Args:
            generated_answer: Model tarafından üretilen cevap
            ideal_answer: İdeal/beklenen cevap
            
        Returns:
            float: 0-100 arası puan
        """
        if self.model is None:
            return 0.0
        
        try:
            # Boş string kontrolü
            if not generated_answer or not ideal_answer:
                return 0.0
            
            # Vektörlere çevir
            embeddings = self.model.encode(
                [generated_answer, ideal_answer],
                convert_to_tensor=True
            )
            
            # Cosine similarity
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
            score = similarity.item() * 100
            
            # 0-100 arasında sınırla
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            print(f"⚠️ Semantic scoring hatası: {e}")
            return 0.0
    
    def batch_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Toplu puanlama (daha hızlı).
        
        Args:
            pairs: [(generated, ideal), (generated, ideal), ...]
            
        Returns:
            Puan listesi
        """
        if self.model is None:
            return [0.0] * len(pairs)
        
        try:
            if not pairs:
                return []
            
            generated = [p[0] for p in pairs]
            ideal = [p[1] for p in pairs]
            
            # Batch encoding
            gen_embeddings = self.model.encode(generated, convert_to_tensor=True)
            ideal_embeddings = self.model.encode(ideal, convert_to_tensor=True)
            
            # Batch cosine similarity
            similarities = util.pytorch_cos_sim(gen_embeddings, ideal_embeddings)
            
            # Diagonal'ı al (her cevap kendi ideal'i ile)
            scores = [
                max(0.0, min(100.0, similarities[i][i].item() * 100))
                for i in range(len(pairs))
            ]
            
            return scores
            
        except Exception as e:
            print(f"⚠️ Batch semantic scoring hatası: {e}")
            return [0.0] * len(pairs)
    
    def is_available(self) -> bool:
        """Scorer kullanılabilir mi?"""
        return self.model is not None
