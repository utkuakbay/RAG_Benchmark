"""
Model Manager - LLM modellerini yöneten sınıf.

API modelleri (Gemini, GPT, Claude) ve Ollama local modelleri destekler.
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Config import
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import MODEL_CONFIG, calculate_cost


@dataclass
class ModelMetrics:
    """Model performans metriklerini tutan veri sınıfı."""
    model_name: str
    response: str
    duration: float
    retrieval_time: float = 0.0
    llm_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    retrieved_docs: int = 0
    
    # Evaluation metrikleri
    final_score: float = 0.0
    semantic_score: float = 0.0
    bert_f1: float = 0.0
    rouge_l: float = 0.0
    keyword_f1: float = 0.0
    
    # Hardware metrikleri
    ram_before: float = 0.0
    ram_after: float = 0.0
    cpu_usage: float = 0.0


class ModelManager:
    """
    LLM modellerini yöneten sınıf.
    
    Desteklenen modeller:
    - API: Gemini, GPT, Claude
    - Local (Ollama): Llama, Mistral, Phi, Qwen
    """
    
    def __init__(self):
        """ModelManager başlatıcı."""
        self.models: Dict[str, Any] = {}
        self.available_models: List[str] = []
        self.ollama_available = False
        
        # Ollama bağlantısını test et
        self._check_ollama_connection()
    
    def _check_ollama_connection(self) -> bool:
        """Ollama servisinin çalışıp çalışmadığını kontrol et."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.ollama_available = True
                print("✅ Ollama servisi çalışıyor")
                
                # Mevcut modelleri listele
                data = response.json()
                models = [m.get('name', '') for m in data.get('models', [])]
                if models:
                    print(f"   Yüklü modeller: {', '.join(models)}")
                return True
        except Exception as e:
            print(f"⚠️ Ollama servisi bulunamadı: {e}")
            print("   Ollama'yı başlatmak için: 'ollama serve' komutunu çalıştırın")
        
        self.ollama_available = False
        return False
    
    def setup_models(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        use_ollama: bool = True,
        ollama_models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Modelleri yükle ve hazırla.
        
        Args:
            gemini_api_key: Google Gemini API key
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic Claude API key
            use_ollama: Ollama modellerini kullan
            ollama_models: Kullanılacak Ollama modelleri listesi
                           None ise sadece Llama kullanılır
        
        Returns:
            Yüklenen modeller sözlüğü
        """
        print(f"\n{'='*60}")
        print("MODELLER YÜKLENİYOR")
        print(f"{'='*60}")
        
        self.models = {}
        
        # API Modelleri
        if gemini_api_key:
            self._load_gemini(gemini_api_key)
        
        if openai_api_key:
            self._load_openai(openai_api_key)
        
        if anthropic_api_key:
            self._load_claude(anthropic_api_key)
        
        # Ollama Modelleri
        if use_ollama and self.ollama_available:
            # Varsayılan olarak sadece Llama (ilk aşama için)
            models_to_load = ollama_models or ["Llama"]
            
            for model_name in models_to_load:
                self._load_ollama_model(model_name)
        
        print(f"{'='*60}")
        print(f"Toplam {len(self.models)} model yüklendi: {', '.join(self.models.keys())}")
        print(f"{'='*60}\n")
        
        self.available_models = list(self.models.keys())
        
        if not self.models:
            raise Exception("Hiçbir model yüklenemedi! En az bir geçerli API key veya Ollama gerekli.")
        
        return self.models
    
    def _load_gemini(self, api_key: str) -> bool:
        """Gemini modelini yükle."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            config = MODEL_CONFIG.get("Gemini", {})
            
            model = ChatGoogleGenerativeAI(
                model=config.get("model_name", "gemini-2.0-flash-exp"),
                google_api_key=api_key,
                temperature=config.get("temperature", 0.0)
            )
            
            self.models["Gemini"] = model
            print("✅ Gemini model yüklendi")
            return True
            
        except Exception as e:
            print(f"❌ Gemini yükleme hatası: {e}")
            return False
    
    def _load_openai(self, api_key: str) -> bool:
        """OpenAI modelini yükle."""
        try:
            from langchain_openai import ChatOpenAI
            
            config = MODEL_CONFIG.get("GPT", {})
            
            model = ChatOpenAI(
                model=config.get("model_name", "gpt-3.5-turbo"),
                openai_api_key=api_key,
                temperature=config.get("temperature", 0.0)
            )
            
            self.models["GPT"] = model
            print("✅ GPT model yüklendi")
            return True
            
        except Exception as e:
            print(f"❌ GPT yükleme hatası: {e}")
            return False
    
    def _load_claude(self, api_key: str) -> bool:
        """Claude modelini yükle."""
        try:
            from langchain_anthropic import ChatAnthropic
            
            config = MODEL_CONFIG.get("Claude", {})
            
            model = ChatAnthropic(
                model=config.get("model_name", "claude-3-5-haiku-latest"),
                anthropic_api_key=api_key,
                temperature=config.get("temperature", 0.0)
            )
            
            self.models["Claude"] = model
            print("✅ Claude model yüklendi")
            return True
            
        except Exception as e:
            print(f"❌ Claude yükleme hatası: {e}")
            return False
    
    def _load_ollama_model(self, model_name: str) -> bool:
        """Ollama modelini yükle."""
        if model_name not in MODEL_CONFIG:
            print(f"⚠️ {model_name} konfigürasyonu bulunamadı")
            return False
        
        config = MODEL_CONFIG[model_name]
        
        if config.get("type") != "local":
            print(f"⚠️ {model_name} local model değil")
            return False
        
        try:
            from langchain_ollama import ChatOllama
            
            ollama_model_name = config.get("model_name")
            
            # Model'in Ollama'da yüklü olup olmadığını kontrol et
            if not self._check_ollama_model_exists(ollama_model_name):
                print(f"⚠️ {model_name} ({ollama_model_name}) Ollama'da yüklü değil")
                print(f"   Yüklemek için: ollama pull {ollama_model_name}")
                return False
            
            model = ChatOllama(
                model=ollama_model_name,
                temperature=config.get("temperature", 0.0)
            )
            
            self.models[model_name] = model
            print(f"✅ {model_name} ({ollama_model_name}) yüklendi")
            return True
            
        except ImportError:
            print("❌ langchain-ollama yüklü değil. 'pip install langchain-ollama' çalıştırın")
            return False
        except Exception as e:
            print(f"❌ {model_name} yükleme hatası: {e}")
            return False
    
    def _check_ollama_model_exists(self, model_name: str) -> bool:
        """Ollama'da model yüklü mü kontrol et."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m.get('name', '') for m in data.get('models', [])]
                
                # Tam eşleşme veya prefix eşleşmesi
                for m in models:
                    if m == model_name or m.startswith(model_name.split(':')[0]):
                        return True
            return False
        except:
            return False
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Model objesi döndür."""
        return self.models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """Kullanılabilir modellerin listesini döndür."""
        return self.available_models
    
    def get_local_models(self) -> List[str]:
        """Sadece local (Ollama) modelleri döndür."""
        return [
            name for name in self.available_models
            if MODEL_CONFIG.get(name, {}).get("type") == "local"
        ]
    
    def get_api_models(self) -> List[str]:
        """Sadece API modellerini döndür."""
        return [
            name for name in self.available_models
            if MODEL_CONFIG.get(name, {}).get("type") == "api"
        ]
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Basit token tahmini (yaklaşık 4 karakter = 1 token).
        
        Args:
            text: Token sayısı tahmin edilecek metin
            
        Returns:
            Tahmini token sayısı
        """
        if not text:
            return 0
        return len(text) // 4
    
    @staticmethod
    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Tahmini maliyeti hesapla.
        
        Args:
            model_name: Model adı
            input_tokens: Input token sayısı
            output_tokens: Output token sayısı
            
        Returns:
            Tahmini maliyet (USD)
        """
        return calculate_cost(model_name, input_tokens, output_tokens)
    
    def invoke_model(
        self,
        model_name: str,
        prompt: str
    ) -> tuple:
        """
        Modeli çağır ve yanıt al.
        
        Args:
            model_name: Model adı
            prompt: Prompt metni
            
        Returns:
            (response_text, duration, input_tokens, output_tokens)
        """
        model = self.get_model(model_name)
        
        if model is None:
            raise Exception(f"Model bulunamadı: {model_name}")
        
        start_time = time.time()
        
        try:
            response = model.invoke(prompt)
            duration = time.time() - start_time
            
            # Response text'i çıkar
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Token tahmini
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(response_text)
            
            return response_text, duration, input_tokens, output_tokens
            
        except Exception as e:
            duration = time.time() - start_time
            raise Exception(f"Model çağrı hatası ({model_name}): {e}")
