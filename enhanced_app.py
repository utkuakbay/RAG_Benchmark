"""
Enhanced RAG Benchmark System - Streamlit UI

Multi-Model RAG Benchmark:
- API Modeller: Gemini, GPT, Claude
- Local Modeller (Ollama): Llama, Mistral, Phi, Qwen

İki Senaryo:
1. Shared Embedding (Fair Arena)
2. Model-Specific Embedding (Real World)
"""

import streamlit as st
import pandas as pd
import time
import os
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv

# Core imports
from core.csv_processor import CSVProcessor
from core.rag_pipeline import RAGPipeline
from core.model_manager import ModelManager
from core.benchmark_runner import BenchmarkRunner
from core.hardware_monitor import HardwareMonitor
from core.evaluation.unified_evaluator import UnifiedEvaluator

# Config imports
from config.model_config import MODEL_CONFIG, HARDWARE_CONFIG

# Load environment variables
load_dotenv()


def init_session_state():
    """Session state'i başlat."""
    if 'benchmark_results' not in st.session_state:
        st.session_state.benchmark_results = None
    if 'test_completed' not in st.session_state:
        st.session_state.test_completed = False
    if 'hw_monitor' not in st.session_state:
        st.session_state.hw_monitor = HardwareMonitor()


def display_system_status():
    """Sistem durumunu göster."""
    hw_monitor = st.session_state.hw_monitor
    stats = hw_monitor.get_system_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Renk göstergeleri fonksiyonel - kalıyor
        ram_color = "🟢" if stats.ram_percent < 70 else "🟡" if stats.ram_percent < 85 else "🔴"
        st.metric(
            f"{ram_color} RAM Kullanımı", 
            f"{stats.ram_percent:.1f}%",
            f"{stats.ram_used_gb:.1f} / {stats.ram_total_gb:.1f} GB"
        )
    
    with col2:
        cpu_color = "🟢" if stats.cpu_percent < 70 else "🟡" if stats.cpu_percent < 85 else "🔴"
        st.metric(
            f"{cpu_color} CPU Kullanımı",
            f"{stats.cpu_percent:.1f}%"
        )
    
    with col3:
        st.metric(
            "Kullanılabilir RAM",
            f"{stats.ram_available_gb:.1f} GB"
        )
    
    # Uyarılar - fonksiyonel
    if stats.ram_percent > 85:
        st.warning("RAM kullanımı yüksek! Test sırasında sorun yaşanabilir.")
    elif stats.ram_available_gb < 4:
        st.warning("Kullanılabilir RAM düşük. Diğer uygulamaları kapatmanızı öneririz.")


def display_ollama_status(model_manager: ModelManager):
    """Ollama durumunu göster."""
    if model_manager.ollama_available:
        st.success("Ollama servisi çalışıyor")
    else:
        st.error("""
        Ollama servisi bulunamadı!
        
        Çözüm:
        1. Ollama'nın kurulu olduğundan emin olun
        2. PowerShell'de `ollama serve` komutunu çalıştırın
        3. Sayfayı yenileyin
        """)


def display_results_table(summary_df: pd.DataFrame):
    """Sonuç tablosunu göster."""
    if summary_df.empty:
        st.warning("Sonuç bulunamadı.")
        return
    
    # Metrik açıklamaları
    with st.expander("ℹ️ Metrikler nasıl hesaplanıyor?", expanded=False):
        st.markdown("""
        ### Değerlendirme Formülü
        ```
        Avg Score = (Semantic × 0.60) + (BERT × 0.30) + (ROUGE × 0.10)
        ```
        
        ---
        
        | Metrik | Ağırlık | Açıklama |
        |--------|---------|----------|
        | **Avg Semantic** | 60% | Anlamsal benzerlik. Cevap ve ideal cevap vektöre çevrilip cosine similarity hesaplanır. Kelimeler farklı olsa bile anlamı yakalar. |
        | **Avg BERT** | 30% | BERTScore. Her kelime bağlamsal embedding'e çevrilir. Eş anlamlı kelimeleri (doktor↔hekim) tanır. |
        | **Avg ROUGE** | 10% | ROUGE-L. En uzun ortak kelime dizisini bulur. Sadece kelime eşleşmesi, anlam yakalamaz. |
        
        ---
        
        | Diğer Metrikler | Açıklama |
        |-----------------|----------|
        | **Avg Time (s)** | Model başına ortalama yanıt süresi |
        | **Total Tokens** | Toplam kullanılan token sayısı |
        | **Total Cost ($)** | API modelleri için tahmini maliyet |
        | **Avg RAM (%)** | Test sırasındaki ortalama RAM kullanımı |
        | **Errors** | Başarısız olan soru sayısı |
        """)
    
    # Formatla
    display_df = summary_df.copy()
    
    # Sayısal sütunları formatla
    for col in ['Avg Score', 'Avg Semantic', 'Avg BERT', 'Avg ROUGE']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
    
    if 'Avg Time (s)' in display_df.columns:
        display_df['Avg Time (s)'] = display_df['Avg Time (s)'].apply(lambda x: f"{x:.2f}")
    
    if 'Total Cost ($)' in display_df.columns:
        display_df['Total Cost ($)'] = display_df['Total Cost ($)'].apply(lambda x: f"${x:.6f}")
    
    if 'Avg RAM (%)' in display_df.columns:
        display_df['Avg RAM (%)'] = display_df['Avg RAM (%)'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, width='stretch')


def display_comparison_table(comparison_df: pd.DataFrame):
    """Delta karşılaştırma tablosunu göster."""
    if comparison_df.empty:
        st.info("Karşılaştırma için her iki senaryo da çalıştırılmalı.")
        return
    
    st.subheader("Scenario 1 vs Scenario 2 — Delta Analizi")
    
    # Formatla
    display_df = comparison_df.copy()
    
    for col in ['Shared Score', 'Own Score', 'Score Delta']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
    
    for col in ['Shared Time', 'Own Time', 'Time Delta']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}s")
    
    st.dataframe(display_df, width='stretch')
    
    # Özet
    improved = len(comparison_df[comparison_df['Improvement'] == 'Yes'])
    total = len(comparison_df)
    
    if improved > 0:
        st.success(f"{improved}/{total} modelde kendi embedding'i ile performans artışı görüldü.")
    else:
        st.info("Shared embedding tüm modellerde yeterli performans sağlıyor.")


def display_charts(summary_df: pd.DataFrame):
    """Benchmark sonuçları için grafikler göster."""
    if summary_df.empty:
        return
    
    # Renk paleti (dark tema uyumlu)
    colors = {
        'primary': '#F5A623',      # Turuncu/Altın
        'secondary': '#4ECDC4',    # Turkuaz
        'success': '#2ECC71',      # Yeşil
        'warning': '#E74C3C',      # Kırmızı
        'background': '#0E1117',   # Dark background
        'text': '#FAFAFA'          # Light text
    }
    
    # Grafik layout ayarları
    layout_config = dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text'], size=12),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    
    st.subheader("Performans Grafikleri")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Skor Grafiği
        fig_score = go.Figure()
        
        # Scenario 1 ve 2'yi ayır
        s1_data = summary_df[summary_df['Scenario'] == 1].copy()
        s2_data = summary_df[summary_df['Scenario'] == 2].copy()
        
        if not s1_data.empty:
            score_values_s1 = pd.to_numeric(s1_data['Avg Score'], errors='coerce')
            fig_score.add_trace(go.Bar(
                name='Scenario 1',
                x=s1_data['Model'],
                y=score_values_s1,
                marker_color=colors['primary'],
                text=score_values_s1.round(1),
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate='%{x}<br>Skor: %{y:.1f}<extra></extra>'
            ))
        
        if not s2_data.empty:
            score_values_s2 = pd.to_numeric(s2_data['Avg Score'], errors='coerce')
            fig_score.add_trace(go.Bar(
                name='Scenario 2',
                x=s2_data['Model'],
                y=score_values_s2,
                marker_color=colors['secondary'],
                text=score_values_s2.round(1),
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate='%{x}<br>Skor: %{y:.1f}<extra></extra>'
            ))
        
        fig_score.update_layout(
            title=dict(text='Model Performans Skorları', font=dict(size=16)),
            xaxis_title='',
            yaxis_title='Skor',
            barmode='group',
            yaxis=dict(range=[0, 105], gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            **layout_config
        )
        
        st.plotly_chart(fig_score, key="chart_score")
    
    with col2:
        # Süre Grafiği
        fig_time = go.Figure()
        
        # Süre verilerini sayısal olarak al
        if not s1_data.empty:
            time_values_s1 = pd.to_numeric(s1_data['Avg Time (s)'], errors='coerce')
            fig_time.add_trace(go.Bar(
                name='Scenario 1',
                x=s1_data['Model'],
                y=time_values_s1,
                marker_color=colors['primary'],
                text=time_values_s1.round(2),
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate='%{x}<br>Süre: %{y:.2f}s<extra></extra>'
            ))
        
        if not s2_data.empty:
            time_values_s2 = pd.to_numeric(s2_data['Avg Time (s)'], errors='coerce')
            fig_time.add_trace(go.Bar(
                name='Scenario 2',
                x=s2_data['Model'],
                y=time_values_s2,
                marker_color=colors['secondary'],
                text=time_values_s2.round(2),
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate='%{x}<br>Süre: %{y:.2f}s<extra></extra>'
            ))
        
        fig_time.update_layout(
            title=dict(text='Yanıt Süreleri (saniye)', font=dict(size=16)),
            xaxis_title='',
            yaxis_title='Süre (s)',
            barmode='group',
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', rangemode='tozero'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            **layout_config
        )
        
        st.plotly_chart(fig_time, key="chart_time")
    
    # İkinci satır: Semantic Skor ve RAM Kullanımı
    col3, col4 = st.columns(2)
    
    with col3:
        # Semantic Skor (tek senaryo için daha iyi görünüm)
        if not s1_data.empty:
            fig_semantic = go.Figure()
            
            # Sayısal dönüşüm
            semantic_values = pd.to_numeric(s1_data['Avg Semantic'], errors='coerce')
            
            # Renkleri skora göre ayarla
            colors_list = [colors['success'] if x >= 80 else colors['primary'] if x >= 60 else colors['warning'] 
                          for x in semantic_values]
            
            fig_semantic.add_trace(go.Bar(
                x=s1_data['Model'],
                y=semantic_values,
                marker_color=colors_list,
                text=semantic_values.round(1),
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate='%{x}<br>Semantic: %{y:.1f}<extra></extra>'
            ))
            
            fig_semantic.update_layout(
                title=dict(text='Anlamsal Benzerlik Skorları', font=dict(size=16)),
                xaxis_title='',
                yaxis_title='Semantic Score',
                yaxis=dict(range=[0, 105], gridcolor='rgba(255,255,255,0.1)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                showlegend=False,
                **layout_config
            )
            
            st.plotly_chart(fig_semantic, key="chart_semantic")
    
    with col4:
        # Maliyet Grafiği (sadece API modelleri için anlamlı)
        cost_values = pd.to_numeric(summary_df['Total Cost ($)'], errors='coerce')
        api_data = summary_df[cost_values > 0].copy()
        
        if not api_data.empty:
            fig_cost = go.Figure()
            
            cost_vals = pd.to_numeric(api_data['Total Cost ($)'], errors='coerce')
            
            fig_cost.add_trace(go.Bar(
                x=api_data['Model'],
                y=cost_vals * 1000,  # Milicent olarak göster
                marker_color=colors['primary'],
                text=[f"${x:.4f}" for x in cost_vals],
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate='%{x}<br>Maliyet: %{text}<extra></extra>'
            ))
            
            fig_cost.update_layout(
                title=dict(text='API Maliyetleri', font=dict(size=16)),
                xaxis_title='',
                yaxis_title='Maliyet (x1000 $)',
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', rangemode='tozero'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                showlegend=False,
                **layout_config
            )
            
            st.plotly_chart(fig_cost, key="chart_cost")
        else:
            # RAM Kullanımı göster
            if 'Avg RAM (%)' in s1_data.columns:
                fig_ram = go.Figure()
                
                # Sayısal dönüşüm
                ram_values = pd.to_numeric(s1_data['Avg RAM (%)'], errors='coerce')
                
                fig_ram.add_trace(go.Bar(
                    x=s1_data['Model'],
                    y=ram_values,
                    marker_color=colors['secondary'],
                    text=ram_values.round(1),
                    textposition='outside',
                    textfont=dict(size=11),
                    hovertemplate='%{x}<br>RAM: %{y:.1f}%<extra></extra>'
                ))
                
                fig_ram.update_layout(
                    title=dict(text='Ortalama RAM Kullanımı (%)', font=dict(size=16)),
                    xaxis_title='',
                    yaxis_title='RAM (%)',
                    yaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    showlegend=False,
                    **layout_config
                )
                
                st.plotly_chart(fig_ram, key="chart_ram")


def display_radar_chart(summary_df: pd.DataFrame):
    """Model karşılaştırma radar grafiği."""
    if summary_df.empty:
        return
    
    # Sadece Scenario 1 verilerini kullan (adil karşılaştırma)
    s1_data = summary_df[summary_df['Scenario'] == 1].copy()
    
    if s1_data.empty:
        return
    
    st.subheader("Model Karşılaştırma Radar Grafiği")
    
    # Metrikleri normalize et (0-100 arası)
    metrics = ['Avg Score', 'Avg Semantic', 'Avg BERT', 'Avg ROUGE']
    
    # Hız için ters skor hesapla (düşük süre = yüksek skor)
    s1_data['Speed Score'] = 100 - (pd.to_numeric(s1_data['Avg Time (s)'], errors='coerce') / 
                                     pd.to_numeric(s1_data['Avg Time (s)'], errors='coerce').max() * 100)
    
    # Maliyet için ters skor (düşük maliyet = yüksek skor, local modeller 100)
    cost_values = pd.to_numeric(s1_data['Total Cost ($)'], errors='coerce')
    max_cost = cost_values.max() if cost_values.max() > 0 else 1
    s1_data['Cost Score'] = 100 - (cost_values / max_cost * 100)
    s1_data.loc[cost_values == 0, 'Cost Score'] = 100  # Local modeller bedava
    
    radar_metrics = ['Avg Score', 'Avg Semantic', 'Avg BERT', 'Speed Score', 'Cost Score']
    radar_labels = ['Genel Skor', 'Semantic', 'BERT', 'Hız', 'Maliyet Etkinliği']
    
    fig = go.Figure()
    
    colors = ['#F5A623', '#4ECDC4', '#2ECC71', '#E74C3C', '#9B59B6', '#3498DB', '#1ABC9C']
    
    for idx, (_, row) in enumerate(s1_data.iterrows()):
        values = []
        for metric in radar_metrics:
            val = pd.to_numeric(row.get(metric, 0), errors='coerce')
            values.append(val if not pd.isna(val) else 0)
        
        # Radar için değerleri kapat (ilk değeri sona ekle)
        values.append(values[0])
        labels = radar_labels + [radar_labels[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=row['Model'],
            line_color=colors[idx % len(colors)],
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        height=500,
        margin=dict(t=30, b=80)
    )
    
    st.plotly_chart(fig, key="radar_chart", use_container_width=True)
    
    # Radar açıklaması
    with st.expander("ℹ️ Radar grafiği nasıl okunur?", expanded=False):
        st.markdown("""
        **Her eksen bir performans metriğini temsil eder:**
        
        | Eksen | Açıklama |
        |-------|----------|
        | **Genel Skor** | Ağırlıklı ortalama (Semantic %60 + BERT %30 + ROUGE %10) |
        | **Semantic** | Anlamsal benzerlik skoru |
        | **BERT** | BERTScore F1 değeri |
        | **Hız** | Yanıt süresi (düşük süre = yüksek skor) |
        | **Maliyet Etkinliği** | Düşük maliyet = yüksek skor (Local modeller 100) |
        
        **Daha geniş alan = daha iyi performans**
        """)


def generate_ai_analysis(summary_df: pd.DataFrame, anthropic_api_key: str) -> str:
    """Claude API ile benchmark sonuçlarını yorumla."""
    if not anthropic_api_key:
        return None
    
    try:
        from langchain_anthropic import ChatAnthropic
        
        # Sadece Scenario 1 verilerini kullan
        s1_data = summary_df[summary_df['Scenario'] == 1].copy()
        
        if s1_data.empty:
            return None
        
        # En iyi ve en kötü modeli bul
        s1_data['Avg Score Num'] = pd.to_numeric(s1_data['Avg Score'], errors='coerce')
        best_model = s1_data.loc[s1_data['Avg Score Num'].idxmax()]
        worst_model = s1_data.loc[s1_data['Avg Score Num'].idxmin()]
        
        # API ve Local ayırımı
        api_models = s1_data[s1_data['Total Cost ($)'].apply(lambda x: pd.to_numeric(x, errors='coerce') > 0)]
        local_models = s1_data[s1_data['Total Cost ($)'].apply(lambda x: pd.to_numeric(x, errors='coerce') == 0)]
        
        # Prompt oluştur
        prompt = f"""Sen bir RAG (Retrieval-Augmented Generation) sistemi danışmanısın. Aşağıdaki benchmark sonuçlarını analiz et ve kullanıcıya hangi modeli seçmesi gerektiği konusunda profesyonel bir öneri sun.

## Benchmark Sonuçları (Scenario 1 - Adil Karşılaştırma)

{s1_data[['Model', 'Avg Score', 'Avg Semantic', 'Avg BERT', 'Avg Time (s)', 'Total Cost ($)']].to_string(index=False)}

## Analiz İstekleri:

1. **En İyi Model:** {best_model['Model']} ({best_model['Avg Score']:.1f} puan)
2. **Dikkat Edilmesi Gereken Model:** {worst_model['Model']} ({worst_model['Avg Score']:.1f} puan)

Lütfen şu başlıklar altında kısa ve öz bir analiz yap (Türkçe):

### 🏆 Genel Değerlendirme
(Hangi model en iyi performansı gösterdi ve neden?)

### 💰 Maliyet-Performans Analizi  
(API vs Local modeller karşılaştırması)

### 🎯 Kullanım Senaryosu Önerileri
(Hangi durumlarda hangi model tercih edilmeli?)

### ⚡ Sonuç ve Tavsiye
(Tek cümlelik net öneri)

Not: Yanıtın 300 kelimeyi geçmesin. Markdown formatında yaz."""

        # Claude'a gönder
        llm = ChatAnthropic(
            model="claude-3-5-haiku-latest",
            api_key=anthropic_api_key,
            temperature=0.3,
            max_tokens=1000
        )
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        print(f"AI analiz hatası: {e}")
        return None


def main():
    """Ana Streamlit uygulaması."""
    
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="RAG Benchmark",
        page_icon="◆",
        layout="wide"
    )
    
    # Session state
    init_session_state()
    
    # Başlık
    st.title("Multi-Model RAG Benchmark")
    st.markdown("""
    API Modelleri (Gemini, GPT, Claude) ve Local Modelleri (Ollama: Llama, Mistral, Phi, Qwen) 
    iki farklı senaryoda karşılaştırın.
    
    **Senaryo 1 — Fair Arena:** Tüm modeller aynı embedding ile test edilir  
    **Senaryo 2 — Real World:** Local modeller kendi optimize edilmiş embedding'leri ile test edilir
    """)
    
    st.divider()
    
    # Sistem durumu
    st.subheader("Sistem Durumu")
    display_system_status()
    
    # API keylerini .env'den oku (UI'da gösterme)
    gemini_api_key = os.getenv("GOOGLE_API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Model seçimi
    st.sidebar.header("Model Seçimi")
    st.sidebar.caption("API Modelleri")
    
    use_gemini = st.sidebar.checkbox(
        "Gemini 2.5 Flash", value=bool(gemini_api_key),
        help="Google'ın Gemini 2.5 Flash modeli. Hızlı ve maliyet etkin.",
        disabled=not bool(gemini_api_key)
    )
    use_gpt = st.sidebar.checkbox(
        "GPT-3.5 Turbo", value=False,
        help="OpenAI'ın GPT-3.5 Turbo modeli. Güvenilir ve yaygın kullanılan.",
        disabled=not bool(openai_api_key)
    )
    use_claude = st.sidebar.checkbox(
        "Claude 3.5 Haiku", value=bool(anthropic_api_key),
        help="Anthropic'in en hızlı ve ucuz Claude modeli.",
        disabled=not bool(anthropic_api_key)
    )
    
    st.sidebar.divider()
    st.sidebar.caption("🦙 Ollama Modelleri (Local)")
    
    use_llama = st.sidebar.checkbox(
        "Llama 3.1 (8B)", value=True,
        help="Meta'nın güçlü açık kaynak modeli. Genel amaçlı, dengeli performans."
    )
    use_mistral = st.sidebar.checkbox(
        "Mistral (7B)", value=False,
        help="Hızlı ve verimli Fransız modeli. Kod ve mantık işlemlerinde güçlü."
    )
    use_phi = st.sidebar.checkbox(
        "Phi-3 (3.8B)", value=False,
        help="Microsoft'un küçük ama güçlü modeli. Düşük RAM kullanımı."
    )
    use_qwen = st.sidebar.checkbox(
        "Qwen 2.5 (7B)", value=False,
        help="Alibaba'nın çok dilli modeli. Matematik ve kodlamada başarılı."
    )
    
    # Senaryo seçimi
    st.sidebar.header("Senaryo Seçimi")
    
    with st.sidebar.expander("ℹ️ Senaryolar hakkında", expanded=False):
        st.markdown("""
        **Scenario 1 (Shared — Fair Arena):**
        - Tüm modeller aynı embedding modeli kullanır (MiniLM-L6-v2)
        - Adil karşılaştırma: Aynı context ile sadece LLM yetenekleri test edilir
        - API ve Local modeller eşit şartlarda yarışır
        
        **Scenario 2 (Own — Real World):**
        - Her local model kendi optimize embedding'ini kullanır
        - Llama → BGE-Large, Mistral → MPNet, Phi → Multilingual, Qwen → BGE-Base
        - Gerçek dünya performansını simüle eder
        - Sadece local modeller test edilir
        """)
    
    run_scenario_1 = st.sidebar.checkbox(
        "Scenario 1 (Shared)", 
        value=True,
        help="Tüm modeller aynı MiniLM-L6-v2 embedding ile test edilir"
    )
    run_scenario_2 = st.sidebar.checkbox(
        "Scenario 2 (Own)", 
        value=False,
        help="Her local model kendi embedding modeli ile test edilir"
    )
    
    # Ayarlar
    st.sidebar.header("⚙ Ayarlar")
    
    with st.sidebar.expander("ℹ️ Parametreler hakkında", expanded=False):
        st.markdown("""
        **Chunk Size:** Corpus'un kaç karakterlik parçalara bölüneceği
        - Kısa metinler → 500-800
        - Uzun metinler → 1000-1500
        - Paragraf bazlı → 1500-2000
        
        **Chunk Overlap:** Parçalar arası örtüşme miktarı
        - Bağlam kaybını önler
        - Genelde Chunk Size'ın %10-20'si
        
        **Retriever K:** Her soru için kaç doküman getirilecek
        - K=1-2 → Hızlı, dar odak
        - K=3-5 → Dengeli (önerilen)
        - K=6+ → Geniş context, yavaş
        """)
    
    chunk_size = st.sidebar.slider(
        "Chunk Size", 500, 2000, 1000, 100,
        help="Corpus'un kaç karakterlik parçalara bölüneceği. Kısa metinler için düşük, uzun metinler için yüksek değer seçin."
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap", 0, 500, 200, 50,
        help="Parçalar arası örtüşme. Bağlam kaybını önler. Chunk Size'ın %10-20'si önerilir."
    )
    retriever_k = st.sidebar.slider(
        "Retriever K", 1, 10, 3, 1,
        help="Her soru için kaç doküman getirilecek. 3-5 arası dengeli sonuç verir."
    )
    
    st.sidebar.divider()
    st.sidebar.header("📊 Test Ayarları")
    
    max_questions = st.sidebar.slider(
        "Test Soru Sayısı", 8, 64, 16, 8,
        help="Benchmark için kullanılacak soru sayısı. Daha fazla soru = daha güvenilir sonuç, ama daha uzun süre."
    )
    
    st.divider()
    
    # Dosya yükleyiciler
    st.subheader("Dosya Yükleme")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Corpus CSV**")
        corpus_file = st.file_uploader(
            "Kütüphane/Corpus CSV dosyasını yükleyin",
            type=["csv"],
            key="corpus",
            label_visibility="collapsed"
        )
        if corpus_file:
            st.success(f"{corpus_file.name} yüklendi")
    
    with col2:
        st.markdown("**Test CSV**")
        test_file = st.file_uploader(
            "Test soruları CSV dosyasını yükleyin",
            type=["csv"],
            key="test",
            label_visibility="collapsed"
        )
        if test_file:
            st.success(f"{test_file.name} yüklendi")
    
    # Ollama durumu
    st.subheader("Ollama Durumu")
    
    model_manager = ModelManager()
    display_ollama_status(model_manager)
    
    st.divider()
    
    # Test butonu
    if st.button("Benchmark Başlat", type="primary", use_container_width=True):
        # Validasyonlar
        if not corpus_file:
            st.error("Lütfen corpus CSV dosyası yükleyin.")
            return
        
        if not test_file:
            st.error("Lütfen test CSV dosyası yükleyin.")
            return
        
        if not run_scenario_1 and not run_scenario_2:
            st.error("En az bir senaryo seçin.")
            return
        
        # Seçilen modeller
        selected_ollama = []
        if use_llama:
            selected_ollama.append("Llama")
        if use_mistral:
            selected_ollama.append("Mistral")
        if use_phi:
            selected_ollama.append("Phi")
        if use_qwen:
            selected_ollama.append("Qwen")
        
        if not use_gemini and not use_gpt and not use_claude and not selected_ollama:
            st.error("En az bir model seçin.")
            return
        
        # Benchmark çalıştır
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # CSV İşleme
            status_text.text("Corpus işleniyor...")
            csv_processor = CSVProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            documents = csv_processor.load_and_chunk(corpus_file)
            st.info(f"{len(documents)} chunk oluşturuldu")
            progress_bar.progress(10)
            
            # Test CSV
            status_text.text("Test dosyası yükleniyor...")
            test_df_full = csv_processor.load_test_csv(test_file)
            
            # Kullanıcının seçtiği soru sayısı kadar rastgele seç
            if len(test_df_full) > max_questions:
                test_df = test_df_full.sample(n=max_questions, random_state=None).reset_index(drop=True)
                st.info(f"{len(test_df_full)} sorudan rastgele {max_questions} tanesi seçildi")
            else:
                test_df = test_df_full
                st.info(f"{len(test_df)} test sorusu yüklendi")
            progress_bar.progress(15)
            
            # RAG Pipeline
            status_text.text("RAG Pipeline kuruluyor...")
            rag_pipeline = RAGPipeline(documents)
            
            # Shared embedding (Scenario 1 için)
            if run_scenario_1:
                rag_pipeline.setup_shared_embedding()
            progress_bar.progress(25)
            
            # Model Manager
            status_text.text("Modeller yükleniyor...")
            model_manager.setup_models(
                gemini_api_key=gemini_api_key if use_gemini else None,
                openai_api_key=openai_api_key if use_gpt else None,
                anthropic_api_key=anthropic_api_key if use_claude else None,
                use_ollama=bool(selected_ollama),
                ollama_models=selected_ollama if selected_ollama else None
            )
            st.info(f"{len(model_manager.get_available_models())} model yüklendi")
            progress_bar.progress(35)
            
            # Evaluator
            status_text.text("Değerlendirme modülleri hazırlanıyor...")
            evaluator = UnifiedEvaluator()
            progress_bar.progress(45)
            
            # Benchmark Runner
            benchmark_runner = BenchmarkRunner(
                rag_pipeline=rag_pipeline,
                model_manager=model_manager,
                evaluator=evaluator,
                hw_monitor=st.session_state.hw_monitor
            )
            
            # Senaryoları belirle
            scenarios = []
            if run_scenario_1:
                scenarios.append(1)
            if run_scenario_2:
                scenarios.append(2)
            
            # Progress callback
            def update_progress(pct):
                progress_bar.progress(int(45 + pct * 50))
            
            # Benchmark çalıştır
            status_text.text("Benchmark çalışıyor...")
            
            results = benchmark_runner.run_full_benchmark(
                test_df=test_df,
                scenarios=scenarios,
                k=retriever_k,
                progress_callback=update_progress
            )
            
            progress_bar.progress(95)
            
            # Özet oluştur
            status_text.text("Sonuçlar hazırlanıyor...")
            summary_df = benchmark_runner.generate_summary(results)
            comparison_df = benchmark_runner.generate_comparison(results)
            
            progress_bar.progress(100)
            status_text.text("Benchmark tamamlandı.")
            
            # Sonuçları session state'e kaydet
            st.session_state.benchmark_results = {
                'results': results,
                'summary': summary_df,
                'comparison': comparison_df
            }
            st.session_state.test_completed = True
            
            st.success("Benchmark başarıyla tamamlandı.")
            
        except Exception as e:
            st.error(f"Hata oluştu: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Sonuçları göster
    if st.session_state.test_completed and st.session_state.benchmark_results:
        st.divider()
        st.header("Benchmark Sonuçları")
        
        results = st.session_state.benchmark_results
        
        # Özet tablo
        st.subheader("Model Performans Özeti")
        display_results_table(results['summary'])
        
        # Karşılaştırma
        if not results['comparison'].empty:
            display_comparison_table(results['comparison'])
        
        # Grafikler
        display_charts(results['summary'])
        
        # Radar Chart
        display_radar_chart(results['summary'])
        
        # AI Analizi (Claude ile)
        if anthropic_api_key:
            st.subheader("🤖 AI Destekli Analiz")
            
            with st.spinner("Claude analiz yapıyor..."):
                ai_analysis = generate_ai_analysis(results['summary'], anthropic_api_key)
                
            if ai_analysis:
                st.markdown(ai_analysis)
            else:
                st.info("AI analizi oluşturulamadı.")
        
        # Ham veriler
        with st.expander("Ham Veriler"):
            if 'scenario_1' in results['results']:
                st.markdown("**Scenario 1 Detayları**")
                s1_data = [r.to_dict() for r in results['results']['scenario_1']]
                st.dataframe(pd.DataFrame(s1_data))
            
            if 'scenario_2' in results['results']:
                st.markdown("**Scenario 2 Detayları**")
                s2_data = [r.to_dict() for r in results['results']['scenario_2']]
                st.dataframe(pd.DataFrame(s2_data))
        
        # CSV İndirme
        st.subheader("Sonuçları İndir")
        
        col1, col2 = st.columns(2)
        
        with col1:
            summary_csv = results['summary'].to_csv(index=False)
            st.download_button(
                "Özet Tabloyu İndir (CSV)",
                summary_csv,
                "benchmark_summary.csv",
                "text/csv"
            )
        
        with col2:
            if not results['comparison'].empty:
                comparison_csv = results['comparison'].to_csv(index=False)
                st.download_button(
                    "Karşılaştırmayı İndir (CSV)",
                    comparison_csv,
                    "benchmark_comparison.csv",
                    "text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.caption("Enhanced RAG Benchmark System v2.0")


if __name__ == "__main__":
    main()
