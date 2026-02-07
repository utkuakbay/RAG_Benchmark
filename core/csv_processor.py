"""
CSV Processor - CSV dosyalarını dinamik olarak okuyup chunking işlemi yapan sınıf.
"""

import pandas as pd
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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
            uploaded_file: Streamlit tarafından yüklenen dosya veya dosya yolu
            
        Returns:
            Document listesi (chunk'lanmış)
        """
        try:
            # CSV'yi oku
            if isinstance(uploaded_file, str):
                df = pd.read_csv(uploaded_file)
            else:
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
            
            if chunked_documents:
                avg_length = sum(len(doc.page_content) for doc in chunked_documents) / len(chunked_documents)
                print(f"Ortalama chunk uzunluğu: {avg_length:.0f} karakter\n")
            
            return chunked_documents
            
        except Exception as e:
            raise Exception(f"CSV işleme hatası: {str(e)}")
    
    def load_test_csv(self, uploaded_file) -> pd.DataFrame:
        """
        Test CSV dosyasını yükle.
        
        Args:
            uploaded_file: Streamlit tarafından yüklenen dosya veya dosya yolu
            
        Returns:
            Test DataFrame'i
        """
        try:
            if isinstance(uploaded_file, str):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Sütun kontrolü
            required_columns = ["soru", "ideal_cevap"]
            for col in required_columns:
                if col not in df.columns:
                    raise Exception(f"Test CSV'sinde '{col}' sütunu bulunamadı!")
            
            print(f"\n📋 Test CSV yüklendi: {len(df)} soru")
            
            return df
            
        except Exception as e:
            raise Exception(f"Test CSV yükleme hatası: {str(e)}")
