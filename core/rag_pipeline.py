"""
RAG Pipeline - FAISS vektör veritabanı ve retriever yönetimi.

İki senaryo destekler:
1. Shared Embedding: Tüm modeller aynı embedding'i kullanır
2. Model-Specific Embedding: Her model kendi embedding'ini kullanır

Özellikler:
- GPU otomatik algılama (CUDA varsa kullanır)
- Vectorstore cache (diske kaydetme/yükleme)
"""

import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Config import
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import (
    SHARED_EMBEDDING, 
    EMBEDDING_MODELS, 
    MODEL_CONFIG,
    get_model_embedding
)

# Cache dizini
CACHE_DIR = Path(__file__).parent.parent / "cache" / "vectorstores"


class RAGPipeline:
    """
    RAG Pipeline - FAISS vektör veritabanı ve retriever yönetimi.
    
    İki senaryo:
    1. Shared: Tüm modeller aynı embedding'i kullanır
    2. Model-Specific: Her local model kendi embedding'ini kullanır
    
    Özellikler:
    - GPU otomatik algılama
    - Vectorstore cache (disk'e kaydet/yükle)
    """
    
    def __init__(self, documents: List[Document], use_cache: bool = True):
        """
        Args:
            documents: Chunk'lanmış doküman listesi
            use_cache: Vectorstore cache kullanılsın mı
        """
        self.documents = documents
        self.use_cache = use_cache
        
        # GPU algılama
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Shared embedding ve vectorstore
        self.shared_embedding = None
        self.shared_vectorstore = None
        
        # Model-specific embedding'ler ve vectorstore'lar
        self.model_embeddings: Dict[str, Any] = {}
        self.model_vectorstores: Dict[str, Any] = {}
        
        # Doküman hash'i (cache için)
        self._doc_hash = self._calculate_doc_hash()
        
        # Cache dizinini oluştur
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📚 RAGPipeline başlatıldı: {len(documents)} doküman")
        print(f"   Device: {self.device.upper()}")
        print(f"   Cache: {'Aktif' if use_cache else 'Kapalı'}")
    
    def _calculate_doc_hash(self) -> str:
        """Doküman içeriğinden hash hesapla (cache key için)."""
        content = "".join([doc.page_content[:100] for doc in self.documents[:10]])
        content += str(len(self.documents))
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _get_cache_path(self, embedding_name: str) -> Path:
        """Cache dosya yolunu döndür."""
        safe_name = embedding_name.replace("/", "_").replace(":", "_")
        return CACHE_DIR / f"{safe_name}_{self._doc_hash}"
    
    def _load_from_cache(self, embedding_name: str, embedding) -> Optional[FAISS]:
        """Cache'den vectorstore yükle."""
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(embedding_name)
        if cache_path.exists():
            try:
                vectorstore = FAISS.load_local(
                    str(cache_path), 
                    embedding,
                    allow_dangerous_deserialization=True
                )
                print(f"   ✅ Cache'den yüklendi: {cache_path.name}")
                return vectorstore
            except Exception as e:
                print(f"   ⚠️ Cache yüklenemedi: {e}")
        return None
    
    def _save_to_cache(self, embedding_name: str, vectorstore: FAISS) -> None:
        """Vectorstore'u cache'e kaydet."""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(embedding_name)
        try:
            vectorstore.save_local(str(cache_path))
            print(f"   💾 Cache'e kaydedildi: {cache_path.name}")
        except Exception as e:
            print(f"   ⚠️ Cache kaydedilemedi: {e}")
    
    def setup_shared_embedding(self, embedding_model: str = None) -> None:
        """
        Scenario 1: Shared embedding kurulumu.
        
        Tüm modeller bu embedding'i kullanacak.
        
        Args:
            embedding_model: Kullanılacak embedding modeli (None = default)
        """
        model_name = embedding_model or SHARED_EMBEDDING
        
        print(f"\n{'='*60}")
        print(f"SHARED EMBEDDING OLUŞTURULUYOR")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device.upper()}")
        
        try:
            start_time = time.time()
            
            # Embedding modeli oluştur (GPU destekli)
            self.shared_embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Önce cache'den yüklemeyi dene
            cached = self._load_from_cache(model_name, self.shared_embedding)
            if cached:
                self.shared_vectorstore = cached
                elapsed_time = time.time() - start_time
                print(f"✅ Shared vectorstore hazır ({elapsed_time:.2f} saniye)")
                print(f"   İndekslenen doküman: {len(self.documents)}")
                print(f"{'='*60}\n")
                return
            
            # Cache yoksa yeni oluştur
            print("   📊 Yeni vectorstore oluşturuluyor...")
            self.shared_vectorstore = FAISS.from_documents(
                documents=self.documents,
                embedding=self.shared_embedding
            )
            
            # Cache'e kaydet
            self._save_to_cache(model_name, self.shared_vectorstore)
            
            elapsed_time = time.time() - start_time
            
            print(f"✅ Shared vectorstore oluşturuldu ({elapsed_time:.2f} saniye)")
            print(f"   İndekslenen doküman: {len(self.documents)}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            raise Exception(f"Shared embedding oluşturma hatası: {str(e)}")
    
    def setup_model_specific_embedding(self, model_name: str) -> None:
        """
        Scenario 2: Model-specific embedding kurulumu.
        
        Args:
            model_name: Model adı (Llama, Mistral, Phi, Qwen)
        """
        # Model için embedding al
        embedding_model = get_model_embedding(model_name, scenario=2)
        
        if embedding_model is None:
            print(f"⚠️ {model_name} için Scenario 2 embedding tanımlı değil")
            return
        
        # Zaten oluşturulmuş mu kontrol et
        if model_name in self.model_vectorstores:
            print(f"✅ {model_name} vectorstore zaten mevcut (bellekte)")
            return
        
        print(f"\n📊 {model_name} için özel embedding oluşturuluyor...")
        print(f"   Model: {embedding_model}")
        print(f"   Device: {self.device.upper()}")
        
        try:
            start_time = time.time()
            
            # Embedding modeli oluştur (GPU destekli)
            embedding = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Önce cache'den yüklemeyi dene
            cached = self._load_from_cache(embedding_model, embedding)
            if cached:
                self.model_embeddings[model_name] = embedding
                self.model_vectorstores[model_name] = cached
                elapsed_time = time.time() - start_time
                print(f"   ✅ {model_name} vectorstore hazır ({elapsed_time:.2f} saniye)")
                return
            
            # Cache yoksa yeni oluştur
            print("   📊 Yeni vectorstore oluşturuluyor...")
            vectorstore = FAISS.from_documents(
                documents=self.documents,
                embedding=embedding
            )
            
            # Cache'e kaydet
            self._save_to_cache(embedding_model, vectorstore)
            
            elapsed_time = time.time() - start_time
            
            # Kaydet
            self.model_embeddings[model_name] = embedding
            self.model_vectorstores[model_name] = vectorstore
            
            print(f"   ✅ {model_name} vectorstore hazır ({elapsed_time:.2f} saniye)")
            
        except Exception as e:
            print(f"   ❌ {model_name} embedding hatası: {str(e)}")
    
    def setup_all_model_specific_embeddings(self, model_names: List[str] = None) -> None:
        """
        Tüm local modeller için model-specific embedding'leri oluştur.
        
        Args:
            model_names: Model listesi (None = tüm local modeller)
        """
        if model_names is None:
            # Sadece local modelleri al
            model_names = [
                name for name, config in MODEL_CONFIG.items()
                if config.get("type") == "local"
            ]
        
        print(f"\n{'='*60}")
        print(f"MODEL-SPECIFIC EMBEDDINGS OLUŞTURULUYOR")
        print(f"{'='*60}")
        print(f"Modeller: {', '.join(model_names)}")
        
        for model_name in model_names:
            self.setup_model_specific_embedding(model_name)
        
        print(f"{'='*60}\n")
    
    def get_retriever(
        self, 
        model_name: str = None, 
        scenario: int = 1, 
        k: int = 3
    ):
        """
        Retriever objesi döndür.
        
        Args:
            model_name: Model adı (Scenario 2 için gerekli)
            scenario: 1 = Shared, 2 = Model-specific
            k: En alakalı k dokümanı getir
            
        Returns:
            Retriever objesi
        """
        if scenario == 1:
            # Shared embedding
            if self.shared_vectorstore is None:
                raise Exception("Shared vectorstore henüz oluşturulmamış!")
            
            return self.shared_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        
        elif scenario == 2:
            # Model-specific embedding
            if model_name is None:
                raise Exception("Scenario 2 için model_name gerekli!")
            
            if model_name not in self.model_vectorstores:
                # Otomatik oluştur
                self.setup_model_specific_embedding(model_name)
            
            if model_name not in self.model_vectorstores:
                raise Exception(f"{model_name} için vectorstore oluşturulamadı!")
            
            return self.model_vectorstores[model_name].as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        
        else:
            raise Exception(f"Geçersiz scenario: {scenario}")
    
    def retrieve(
        self,
        query: str,
        model_name: str = None,
        scenario: int = 1,
        k: int = 3
    ) -> tuple:
        """
        Sorgu için dokümanları retrieve et.
        
        Args:
            query: Sorgu metni
            model_name: Model adı
            scenario: 1 = Shared, 2 = Model-specific
            k: En alakalı k dokümanı getir
            
        Returns:
            (documents, retrieval_time)
        """
        retriever = self.get_retriever(model_name, scenario, k)
        
        start_time = time.time()
        docs = retriever.invoke(query)
        retrieval_time = time.time() - start_time
        
        return docs, retrieval_time
    
    def get_context(self, docs: List[Document]) -> str:
        """Dokümanlardan context metni oluştur."""
        return "\n\n".join([doc.page_content for doc in docs])
    
    def get_doc_ids(self, docs: List[Document]) -> List[int]:
        """Dokümanların row_index'lerini döndür."""
        return [doc.metadata.get("row_index", -1) for doc in docs]
    
    def get_available_scenarios(self, model_name: str) -> List[int]:
        """
        Model için kullanılabilir senaryoları döndür.
        
        Args:
            model_name: Model adı
            
        Returns:
            Kullanılabilir senaryo numaraları listesi
        """
        scenarios = []
        
        # Scenario 1 her zaman kullanılabilir (shared varsa)
        if self.shared_vectorstore is not None:
            scenarios.append(1)
        
        # Scenario 2 sadece local modeller için
        config = MODEL_CONFIG.get(model_name, {})
        if config.get("type") == "local" and config.get("scenario_2_embedding"):
            scenarios.append(2)
        
        return scenarios
