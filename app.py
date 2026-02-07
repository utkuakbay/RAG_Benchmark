# requirements.txt
# streamlit
# pandas
# langchain
# langchain-google-genai
# langchain-openai
# langchain-community
# faiss-cpu
# sentence-transformers
# python-dotenv

import streamlit as st
import pandas as pd
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


@dataclass
class ModelMetrics:
    """Model performans metriklerini tutan veri sınıfı."""
    model_name: str
    response: str
    duration: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    retrieved_docs: int


@dataclass
class TestStatistics:
    """Test istatistiklerini tutan veri sınıfı."""
    total_questions: int = 0
    total_duration: float = 0.0
    model_metrics: Dict[str, List[ModelMetrics]] = field(default_factory=dict)
    
    def add_metric(self, metric: ModelMetrics):
        """Metrik ekle."""
        if metric.model_name not in self.model_metrics:
            self.model_metrics[metric.model_name] = []
        self.model_metrics[metric.model_name].append(metric)
    
    def get_average_duration(self, model_name: str) -> float:
        """Ortalama yanıt süresini hesapla."""
        if model_name not in self.model_metrics:
            return 0.0
        metrics = self.model_metrics[model_name]
        return sum(m.duration for m in metrics) / len(metrics) if metrics else 0.0
    
    def get_total_tokens(self, model_name: str) -> int:
        """Toplam token kullanımını hesapla."""
        if model_name not in self.model_metrics:
            return 0
        metrics = self.model_metrics[model_name]
        return sum(m.total_tokens for m in metrics)
    
    def get_total_cost(self, model_name: str) -> float:
        """Toplam maliyeti hesapla."""
        if model_name not in self.model_metrics:
            return 0.0
        metrics = self.model_metrics[model_name]
        return sum(m.estimated_cost for m in metrics)


class CSVProcessor:
    """CSV dosyalarını dinamik olarak okuyup chunking işlemi yapan sınıf."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Chunk boyutu
            chunk_overlap: Chunk'lar arası örtüşme miktarı
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_and_chunk(self, uploaded_file) -> List[Document]:
        """
        CSV dosyasını yükler, her satırı dinamik olarak işler ve chunk'lara böler.
        
        Args:
            uploaded_file: Streamlit tarafından yüklenen dosya
            
        Returns:
            Document listesi (chunk'lanmış)
        """
        try:
            # CSV'yi oku
            df = pd.read_csv(uploaded_file)
            print(f"\n{'='*60}")
            print(f"CSV YÜKLENDİ")
            print(f"{'='*60}")
            print(f"Toplam Satır Sayısı: {len(df)}")
            print(f"Sütunlar: {', '.join(df.columns.tolist())}")
            print(f"{'='*60}\n")
            
            # Her satırı dinamik olarak metne dönüştür
            documents = []
            for idx, row in df.iterrows():
                # Dinamik olarak tüm sütunları birleştir
                row_text_parts = []
                for column in df.columns:
                    value = row[column]
                    # NaN değerleri atla
                    if pd.notna(value):
                        row_text_parts.append(f"{column}: {value}")
                
                # Satır metnini oluştur
                row_text = " | ".join(row_text_parts)
                
                # Document objesi oluştur
                doc = Document(
                    page_content=row_text,
                    metadata={"row_index": idx, "source": "corpus.csv"}
                )
                documents.append(doc)
            
            print(f"Chunk öncesi doküman sayısı: {len(documents)}")
            
            # Chunk işlemi uygula
            chunked_documents = self.text_splitter.split_documents(documents)
            
            print(f"Chunk sonrası doküman sayısı: {len(chunked_documents)}")
            print(f"Ortalama chunk uzunluğu: {sum(len(doc.page_content) for doc in chunked_documents) / len(chunked_documents):.0f} karakter\n")
            
            return chunked_documents
            
        except Exception as e:
            raise Exception(f"CSV işleme hatası: {str(e)}")


class RAGPipeline:
    """FAISS vektör veritabanı ve retriever yönetimi yapan sınıf."""
    
    def __init__(self, documents: List[Document], embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            documents: Chunk'lanmış doküman listesi
            embedding_model_name: Kullanılacak embedding modeli
        """
        self.documents = documents
        self.embedding_model_name = embedding_model_name
        self.embeddings = None
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Vektör veritabanını başlatır."""
        try:
            print(f"{'='*60}")
            print(f"VEKTÖR VERİTABANI OLUŞTURULUYOR")
            print(f"{'='*60}")
            print(f"Embedding Model: {self.embedding_model_name}")
            
            # Embedding modeli oluştur
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # FAISS vektör veritabanı oluştur
            start_time = time.time()
            self.vectorstore = FAISS.from_documents(
                documents=self.documents,
                embedding=self.embeddings
            )
            elapsed_time = time.time() - start_time
            
            print(f"Vektör veritabanı oluşturuldu ({elapsed_time:.2f} saniye)")
            print(f"İndekslenen doküman sayısı: {len(self.documents)}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            raise Exception(f"Vektör veritabanı oluşturma hatası: {str(e)}")
    
    def get_retriever(self, k: int = 3):
        """
        Retriever objesi döndürür.
        
        Args:
            k: En alakalı k dokümanı getir
            
        Returns:
            Retriever objesi
        """
        if self.vectorstore is None:
            raise Exception("Vektör veritabanı henüz oluşturulmamış!")
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )


class ModelManager:
    """LLM modellerini yöneten sınıf."""
    
    # Model fiyatlandırması (USD per 1M tokens - Güncel fiyatlar)
    PRICING = {
        "Gemini": {
            "input": 0.075 / 1_000_000,  # $0.075 per 1M input tokens
            "output": 0.30 / 1_000_000,  # $0.30 per 1M output tokens
        },
        "GPT": {
            "input": 0.50 / 1_000_000,  # GPT-3.5-turbo: $0.50 per 1M input tokens
            "output": 1.50 / 1_000_000,  # GPT-3.5-turbo: $1.50 per 1M output tokens
        }
    }
    
    @staticmethod
    def get_models(gemini_api_key: Optional[str] = None, 
                   openai_api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Model sözlüğü oluşturur.
        
        Args:
            gemini_api_key: Google Gemini API key
            openai_api_key: OpenAI API key
            
        Returns:
            Model ismi -> Model objesi sözlüğü
        """
        models = {}
        
        try:
            if gemini_api_key:
                print(f"{'='*60}")
                print(f"MODELLER YÜKLENİYOR")
                print(f"{'='*60}")
                
                # Gemini modelini yükle
                gemini_model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=gemini_api_key,
                    temperature=0.0
                )
                models["Gemini"] = gemini_model
                print(f"✓ Gemini model yüklendi")
            else:
                print(f"⚠ Gemini API key bulunamadı, atlanıyor...")
            
            if openai_api_key:
                # OpenAI modelini yükle
                openai_model = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    openai_api_key=openai_api_key,
                    temperature=0.0
                )
                models["GPT"] = openai_model
                print(f"✓ GPT model yüklendi")
            else:
                print(f"⚠ OpenAI API key bulunamadı, atlanıyor...")
            
            print(f"{'='*60}\n")
            
            if not models:
                raise Exception("Hiçbir model yüklenemedi! En az bir geçerli API key gerekli.")
            
            return models
            
        except Exception as e:
            raise Exception(f"Model yükleme hatası: {str(e)}")
    
    @staticmethod
    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Tahmini maliyeti hesaplar.
        
        Args:
            model_name: Model adı
            input_tokens: Input token sayısı
            output_tokens: Output token sayısı
            
        Returns:
            Tahmini maliyet (USD)
        """
        if model_name not in ModelManager.PRICING:
            return 0.0
        
        pricing = ModelManager.PRICING[model_name]
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
        return cost


class BenchmarkRunner:
    """Test çalıştırma ve detaylı loglama yapan sınıf."""
    
    def __init__(self):
        """BenchmarkRunner başlatıcı."""
        self.statistics = TestStatistics()
        self.prompt_template = ChatPromptTemplate.from_template(
            """Aşağıdaki bağlamı kullanarak soruyu cevapla. 
Bilgi bağlamda yoksa 'Bilgi bulunamadı' de.

Bağlam: {context}

Soru: {question}"""
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Basit token tahmini (yaklaşık 4 karakter = 1 token).
        
        Args:
            text: Token sayısı tahmin edilecek metin
            
        Returns:
            Tahmini token sayısı
        """
        return len(text) // 4
    
    def _run_single_query(self, 
                         question: str, 
                         model_name: str, 
                         model: Any, 
                         retriever: Any) -> ModelMetrics:
        """
        Tek bir soru için model çalıştırır ve metrikleri toplar.
        
        Args:
            question: Soru
            model_name: Model adı
            model: Model objesi
            retriever: Retriever objesi
            
        Returns:
            ModelMetrics objesi
        """
        # Manuel RAG chain oluştur (LangChain 1.0 uyumlu)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Chain: Retriever -> Format -> Prompt -> LLM -> Parse
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | model
            | StrOutputParser()
        )
        
        # Sorguyu çalıştır ve süreyi ölç
        start_time = time.time()
        
        try:
            # Dokümanları al (metrik için)
            retrieved_docs = retriever.invoke(question)
            
            # RAG chain'i çalıştır
            response = rag_chain.invoke(question)
            
            duration = time.time() - start_time
            
            # Token hesaplama
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            input_text = f"{context_text}\n{question}"
            
            input_tokens = self._estimate_tokens(input_text)
            output_tokens = self._estimate_tokens(response)
            total_tokens = input_tokens + output_tokens
            
            # Maliyeti hesapla
            estimated_cost = ModelManager.calculate_cost(
                model_name, 
                input_tokens, 
                output_tokens
            )
            
            # Metrik objesi oluştur
            metrics = ModelMetrics(
                model_name=model_name,
                response=response,
                duration=duration,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost,
                retrieved_docs=len(retrieved_docs)
            )
            
            return metrics
            
        except Exception as e:
            # Hata durumunda boş metrik döndür
            duration = time.time() - start_time
            return ModelMetrics(
                model_name=model_name,
                response=f"HATA: {str(e)}",
                duration=duration,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_cost=0.0,
                retrieved_docs=0
            )
    
    def _print_question_header(self, question_num: int, total_questions: int, 
                              question: str, ideal_answer: str):
        """Soru başlığını yazdırır."""
        print(f"\n{'='*80}")
        print(f"SORU {question_num}/{total_questions}: {question}")
        print(f"İDEAL CEVAP: {ideal_answer}")
        print(f"{'='*80}\n")
    
    def _print_model_results(self, metrics: ModelMetrics):
        """Model sonuçlarını yazdırır."""
        print(f"--- MODEL: {metrics.model_name} ---")
        print(f"CEVAP: {metrics.response}")
        print(f"SÜRE: {metrics.duration:.2f} saniye")
        print(f"TOKEN KULLANIMI:")
        print(f"  • Input Tokens: {metrics.input_tokens}")
        print(f"  • Output Tokens: {metrics.output_tokens}")
        print(f"  • Toplam: {metrics.total_tokens} tokens")
        print(f"TAHMİNİ MALİYET: ${metrics.estimated_cost:.6f}")
        print(f"RETRIEVE EDİLEN DOKÜMAN: {metrics.retrieved_docs} adet\n")
    
    def _print_summary(self):
        """Test özet istatistiklerini yazdırır."""
        print(f"\n{'='*80}")
        print(f"╔{'='*78}╗")
        print(f"║{' '*24}TEST TAMAMLANDI - ÖZET{' '*31}║")
        print(f"╚{'='*78}╝")
        print(f"\nGENEL BİLGİLER:")
        print(f"  • Toplam Soru Sayısı: {self.statistics.total_questions}")
        print(f"  • Toplam Test Süresi: {self.statistics.total_duration:.2f} saniye")
        
        for model_name in self.statistics.model_metrics.keys():
            avg_duration = self.statistics.get_average_duration(model_name)
            total_tokens = self.statistics.get_total_tokens(model_name)
            total_cost = self.statistics.get_total_cost(model_name)
            
            print(f"\n{model_name.upper()} PERFORMANSI:")
            print(f"  • Ortalama Yanıt Süresi: {avg_duration:.2f} saniye")
            print(f"  • Toplam Token Kullanımı: {total_tokens}")
            print(f"  • Tahmini Toplam Maliyet: ${total_cost:.6f}")
        
        print(f"{'='*80}\n")
    
    def run_test(self, 
                 test_df: pd.DataFrame, 
                 retriever: Any, 
                 models: Dict[str, Any]) -> TestStatistics:
        """
        Test setini çalıştırır ve detaylı loglar.
        
        Args:
            test_df: Test DataFrame (soru, ideal_cevap sütunları)
            retriever: Retriever objesi
            models: Model sözlüğü
            
        Returns:
            TestStatistics objesi
        """
        try:
            # Sütun kontrolü
            required_columns = ["soru", "ideal_cevap"]
            for col in required_columns:
                if col not in test_df.columns:
                    raise Exception(f"Test CSV'sinde '{col}' sütunu bulunamadı!")
            
            print(f"\n{'='*80}")
            print(f"TEST BAŞLIYOR")
            print(f"{'='*80}")
            print(f"Toplam Soru Sayısı: {len(test_df)}")
            print(f"Test Edilecek Modeller: {', '.join(models.keys())}")
            print(f"{'='*80}\n")
            
            self.statistics.total_questions = len(test_df)
            test_start_time = time.time()
            
            # Her soru için döngü
            for idx, row in test_df.iterrows():
                question = row["soru"]
                ideal_answer = row["ideal_cevap"]
                
                # Soru başlığını yazdır
                self._print_question_header(
                    idx + 1, 
                    len(test_df), 
                    question, 
                    ideal_answer
                )
                
                # Her model için test çalıştır
                for model_name, model in models.items():
                    metrics = self._run_single_query(
                        question=question,
                        model_name=model_name,
                        model=model,
                        retriever=retriever
                    )
                    
                    # Metriği kaydet
                    self.statistics.add_metric(metrics)
                    
                    # Sonuçları yazdır
                    self._print_model_results(metrics)
            
            # Toplam süreyi hesapla
            self.statistics.total_duration = time.time() - test_start_time
            
            # Özet istatistikleri yazdır
            self._print_summary()
            
            return self.statistics
            
        except Exception as e:
            raise Exception(f"Test çalıştırma hatası: {str(e)}")


def main():
    """Ana Streamlit uygulaması."""
    
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="RAG Benchmark V1",
        page_icon="📊",
        layout="wide"
    )
    
    # Başlık
    st.title("Dinamik CSV RAG Benchmark (V1)")
    st.markdown("""
    Bu uygulama, **Google Gemini** ve **OpenAI GPT** modellerinin RAG performansını 
    dinamik CSV korpusu üzerinde karşılaştırmanıza olanak sağlar.
    """)
    
    # Sidebar - API Keys
    st.sidebar.header("🔑 API Anahtarları")
    st.sidebar.markdown("API anahtarlarınızı girin veya .env dosyasından yüklensin.")
    
    # .env'den yükle veya input al
    default_gemini_key = os.getenv("GOOGLE_API_KEY", "")
    default_openai_key = os.getenv("OPENAI_API_KEY", "")
    
    gemini_api_key = st.sidebar.text_input(
        "Google Gemini API Key",
        value=default_gemini_key,
        type="password",
        help="Gemini API anahtarınızı girin"
    )
    
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=default_openai_key,
        type="password",
        help="OpenAI API anahtarınızı girin"
    )
    
    # Ayarlar
    st.sidebar.header("⚙️ Ayarlar")
    chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200, 50)
    retriever_k = st.sidebar.slider("Retriever K", 1, 10, 3, 1)
    
    # Dosya yükleyiciler
    st.header("📁 Dosya Yükleme")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Corpus CSV")
        corpus_file = st.file_uploader(
            "Kütüphane/Corpus CSV dosyasını yükleyin",
            type=["csv"],
            key="corpus",
            help="Dinamik yapıda herhangi bir CSV dosyası yükleyebilirsiniz"
        )
        if corpus_file:
            st.success(f"✓ {corpus_file.name} yüklendi")
    
    with col2:
        st.subheader("2. Test CSV")
        test_file = st.file_uploader(
            "Test soruları CSV dosyasını yükleyin",
            type=["csv"],
            key="test",
            help="'soru' ve 'ideal_cevap' sütunları zorunludur"
        )
        if test_file:
            st.success(f"✓ {test_file.name} yüklendi")
    
    # Test başlatma butonu
    st.header("🚀 Test")
    
    if st.button("Testi Başlat", type="primary", use_container_width=True):
        # Validasyonlar
        if not corpus_file:
            st.error("❌ Lütfen corpus CSV dosyası yükleyin!")
            return
        
        if not test_file:
            st.error("❌ Lütfen test CSV dosyası yükleyin!")
            return
        
        if not gemini_api_key and not openai_api_key:
            st.error("❌ En az bir API key gerekli!")
            return
        
        try:
            # İşlem adımları
            with st.spinner("📄 Corpus işleniyor..."):
                # CSV Processor
                csv_processor = CSVProcessor(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                documents = csv_processor.load_and_chunk(corpus_file)
                st.info(f"✓ {len(documents)} chunk oluşturuldu")
            
            with st.spinner("🔍 Vektör veritabanı oluşturuluyor..."):
                # RAG Pipeline
                rag_pipeline = RAGPipeline(documents)
                retriever = rag_pipeline.get_retriever(k=retriever_k)
                st.info(f"✓ Vektör veritabanı hazır")
            
            with st.spinner("🤖 Modeller yükleniyor..."):
                # Model Manager
                models = ModelManager.get_models(
                    gemini_api_key=gemini_api_key if gemini_api_key else None,
                    openai_api_key=openai_api_key if openai_api_key else None
                )
                st.info(f"✓ {len(models)} model yüklendi: {', '.join(models.keys())}")
            
            with st.spinner("🧪 Test çalıştırılıyor... (Konsol çıktılarını kontrol edin)"):
                # Test CSV'yi oku
                test_df = pd.read_csv(test_file)
                
                # Benchmark Runner
                benchmark_runner = BenchmarkRunner()
                statistics = benchmark_runner.run_test(
                    test_df=test_df,
                    retriever=retriever,
                    models=models
                )
            
            # Başarı mesajı
            st.success("✅ Test başarıyla tamamlandı!")
            st.balloons()
            
            # Özet bilgileri göster
            st.header("📈 Test Özeti")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Toplam Soru", statistics.total_questions)
            with col2:
                st.metric("Toplam Süre", f"{statistics.total_duration:.2f}s")
            with col3:
                st.metric("Test Edilen Model", len(models))
            
            # Model bazlı metrikler
            for model_name in statistics.model_metrics.keys():
                st.subheader(f"🤖 {model_name} Metrikleri")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_duration = statistics.get_average_duration(model_name)
                    st.metric("Ort. Yanıt Süresi", f"{avg_duration:.2f}s")
                with col2:
                    total_tokens = statistics.get_total_tokens(model_name)
                    st.metric("Toplam Token", f"{total_tokens:,}")
                with col3:
                    total_cost = statistics.get_total_cost(model_name)
                    st.metric("Tahmini Maliyet", f"${total_cost:.6f}")
            
            st.info("💡 Detaylı loglar için konsol çıktılarını kontrol edin!")
            
        except Exception as e:
            st.error(f"❌ Hata oluştu: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

