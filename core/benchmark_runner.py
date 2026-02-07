"""
Benchmark Runner - Test çalıştırma ve detaylı loglama.

İki senaryo destekler:
1. Shared Embedding (Fair Arena) - Tüm modeller aynı embedding
2. Model-Specific Embedding (Real World) - Local modeller kendi embedding'leri

SEQUENTIAL EXECUTION - Modeller sırayla çalışır (güvenlik için)
"""

import time
import gc
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .hardware_monitor import HardwareMonitor
from .model_manager import ModelManager, ModelMetrics
from .rag_pipeline import RAGPipeline
from .evaluation.unified_evaluator import UnifiedEvaluator, EvaluationResult

# Config import
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import MODEL_CONFIG, HARDWARE_CONFIG


@dataclass
class TestResult:
    """Tek bir test sonucu."""
    question: str
    ideal_answer: str
    model_name: str
    scenario: int
    answer: str
    
    # Timing
    retrieval_time: float = 0.0
    llm_time: float = 0.0
    total_time: float = 0.0
    
    # Tokens & Cost
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    
    # Retrieval
    retrieved_doc_ids: list = field(default_factory=list)
    retrieved_docs_count: int = 0
    
    # Evaluation
    final_score: float = 0.0
    semantic_score: float = 0.0
    bert_f1: float = 0.0
    rouge_l: float = 0.0
    keyword_f1: float = 0.0
    
    # Hardware
    ram_percent: float = 0.0
    cpu_percent: float = 0.0
    
    # Error
    error: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir."""
        return {
            'question': self.question,
            'ideal_answer': self.ideal_answer,
            'model_name': self.model_name,
            'scenario': self.scenario,
            'answer': self.answer,
            'retrieval_time': self.retrieval_time,
            'llm_time': self.llm_time,
            'total_time': self.total_time,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'estimated_cost': self.estimated_cost,
            'retrieved_docs_count': self.retrieved_docs_count,
            'final_score': self.final_score,
            'semantic_score': self.semantic_score,
            'bert_f1': self.bert_f1,
            'rouge_l': self.rouge_l,
            'keyword_f1': self.keyword_f1,
            'ram_percent': self.ram_percent,
            'cpu_percent': self.cpu_percent,
            'error': self.error
        }


@dataclass
class BenchmarkSummary:
    """Benchmark özet istatistikleri."""
    total_questions: int = 0
    total_duration: float = 0.0
    models_tested: List[str] = field(default_factory=list)
    scenarios_tested: List[int] = field(default_factory=list)
    
    # Model bazlı istatistikler
    model_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class BenchmarkRunner:
    """
    Test çalıştırma ve detaylı loglama sınıfı.
    
    Özellikler:
    - Sequential execution (sıralı çalışma)
    - Hardware monitoring
    - Multi-metric evaluation
    - Detaylı loglama
    """
    
    # RAG Prompt Template
    PROMPT_TEMPLATE = """Aşağıdaki bağlamı kullanarak soruyu cevapla. 
Bilgi bağlamda yoksa 'Bilgi bulunamadı' de.

Bağlam: {context}

Soru: {question}"""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        model_manager: ModelManager,
        evaluator: Optional[UnifiedEvaluator] = None,
        hw_monitor: Optional[HardwareMonitor] = None
    ):
        """
        Args:
            rag_pipeline: RAG pipeline objesi
            model_manager: Model manager objesi
            evaluator: Evaluation modülü (None ise oluşturulur)
            hw_monitor: Hardware monitor (None ise oluşturulur)
        """
        self.rag_pipeline = rag_pipeline
        self.model_manager = model_manager
        
        # Evaluator
        if evaluator is None:
            self.evaluator = UnifiedEvaluator()
        else:
            self.evaluator = evaluator
        
        # Hardware monitor
        if hw_monitor is None:
            self.hw_monitor = HardwareMonitor(
                ram_warning_threshold=HARDWARE_CONFIG.get("ram_warning_threshold", 85),
                ram_critical_threshold=HARDWARE_CONFIG.get("ram_critical_threshold", 90),
                check_interval=HARDWARE_CONFIG.get("check_interval", 5)
            )
        else:
            self.hw_monitor = hw_monitor
        
        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        
        # Results storage
        self.results: List[TestResult] = []
        self.summary: BenchmarkSummary = BenchmarkSummary()
    
    def run_scenario_1(
        self,
        test_df: pd.DataFrame,
        models: List[str] = None,
        k: int = 3,
        progress_callback = None
    ) -> List[TestResult]:
        """
        Scenario 1: Shared Embedding - Tüm modeller aynı embedding ile test.
        
        Args:
            test_df: Test DataFrame (soru, ideal_cevap sütunları)
            models: Test edilecek modeller (None = tüm mevcut modeller)
            k: Retriever için top-k
            progress_callback: İlerleme callback fonksiyonu
            
        Returns:
            Test sonuçları listesi
        """
        print(f"\n{'='*70}")
        print(f"🏟️  SCENARIO 1: SHARED EMBEDDING (Fair Arena)")
        print(f"{'='*70}")
        
        if models is None:
            models = self.model_manager.get_available_models()
        
        print(f"Test edilecek modeller: {', '.join(models)}")
        print(f"Toplam soru: {len(test_df)}")
        print(f"{'='*70}\n")
        
        results = []
        total_tests = len(test_df) * len(models)
        current_test = 0
        
        # Her model için SEQUENTIAL çalıştır
        for model_name in models:
            print(f"\n{'─'*70}")
            print(f"🧪 MODEL: {model_name} (Shared Embedding)")
            print(f"{'─'*70}")
            
            # Hardware check
            hw_status = self.hw_monitor.check_resources(force=True)
            if hw_status.get("critical"):
                print("🔴 RAM kritik seviyede! Test durduruluyor.")
                break
            
            model = self.model_manager.get_model(model_name)
            if model is None:
                print(f"⚠️ {model_name} bulunamadı, atlanıyor...")
                continue
            
            # Her soru için test
            for idx, row in test_df.iterrows():
                current_test += 1
                question = row["soru"]
                ideal_answer = row["ideal_cevap"]
                
                result = self._run_single_test(
                    question=question,
                    ideal_answer=ideal_answer,
                    model_name=model_name,
                    model=model,
                    scenario=1,
                    k=k
                )
                
                results.append(result)
                
                # Progress
                if progress_callback:
                    progress_callback(current_test / total_tests)
                
                # Log
                self._log_test_result(result, idx + 1, len(test_df))
            
            # Model sonrası temizlik
            self._cleanup_after_model(model_name)
        
        return results
    
    def run_scenario_2(
        self,
        test_df: pd.DataFrame,
        models: List[str] = None,
        k: int = 3,
        progress_callback = None
    ) -> List[TestResult]:
        """
        Scenario 2: Model-Specific Embedding - Local modeller kendi embedding'leri ile.
        
        Args:
            test_df: Test DataFrame
            models: Test edilecek local modeller (None = tüm local modeller)
            k: Retriever için top-k
            progress_callback: İlerleme callback fonksiyonu
            
        Returns:
            Test sonuçları listesi
        """
        print(f"\n{'='*70}")
        print(f"🌍 SCENARIO 2: MODEL-SPECIFIC EMBEDDING (Real World)")
        print(f"{'='*70}")
        
        # Sadece local modeller
        if models is None:
            models = self.model_manager.get_local_models()
        else:
            # Sadece local olanları filtrele
            models = [m for m in models if m in self.model_manager.get_local_models()]
        
        if not models:
            print("⚠️ Scenario 2 için local model bulunamadı!")
            return []
        
        print(f"Test edilecek modeller: {', '.join(models)}")
        print(f"Toplam soru: {len(test_df)}")
        print(f"{'='*70}\n")
        
        results = []
        total_tests = len(test_df) * len(models)
        current_test = 0
        
        # Her model için SEQUENTIAL çalıştır
        for model_idx, model_name in enumerate(models, 1):
            print(f"\n{'─'*70}")
            print(f"🔄 MODEL {model_idx}/{len(models)}: {model_name} (Own Embedding)")
            print(f"{'─'*70}")
            
            # Hardware check
            hw_status = self.hw_monitor.check_resources(force=True)
            if hw_status.get("critical"):
                print("🔴 RAM kritik seviyede! Test durduruluyor.")
                break
            
            # Model-specific embedding kur (yoksa)
            self.rag_pipeline.setup_model_specific_embedding(model_name)
            
            model = self.model_manager.get_model(model_name)
            if model is None:
                print(f"⚠️ {model_name} bulunamadı, atlanıyor...")
                continue
            
            # Her soru için test
            for idx, row in test_df.iterrows():
                current_test += 1
                question = row["soru"]
                ideal_answer = row["ideal_cevap"]
                
                result = self._run_single_test(
                    question=question,
                    ideal_answer=ideal_answer,
                    model_name=model_name,
                    model=model,
                    scenario=2,
                    k=k
                )
                
                results.append(result)
                
                # Progress
                if progress_callback:
                    progress_callback(current_test / total_tests)
                
                # Log
                self._log_test_result(result, idx + 1, len(test_df))
            
            # Model sonrası temizlik
            self._cleanup_after_model(model_name)
        
        return results
    
    def _run_single_test(
        self,
        question: str,
        ideal_answer: str,
        model_name: str,
        model: Any,
        scenario: int,
        k: int = 3
    ) -> TestResult:
        """
        Tek bir test çalıştır.
        
        Returns:
            TestResult objesi
        """
        result = TestResult(
            question=question,
            ideal_answer=ideal_answer,
            model_name=model_name,
            scenario=scenario,
            answer=""  # Başlangıçta boş, sonra doldurulacak
        )
        
        try:
            # Hardware stats
            hw_stats = self.hw_monitor.get_system_stats()
            result.ram_percent = hw_stats.ram_percent
            result.cpu_percent = hw_stats.cpu_percent
            
            # Retrieve
            docs, retrieval_time = self.rag_pipeline.retrieve(
                query=question,
                model_name=model_name if scenario == 2 else None,
                scenario=scenario,
                k=k
            )
            result.retrieval_time = retrieval_time
            result.retrieved_docs_count = len(docs)
            result.retrieved_doc_ids = self.rag_pipeline.get_doc_ids(docs)
            
            # Context oluştur
            context = self.rag_pipeline.get_context(docs)
            
            # Prompt oluştur
            prompt = self.PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )
            
            # LLM çağır
            llm_start = time.time()
            response = model.invoke(prompt)
            result.llm_time = time.time() - llm_start
            
            # Response text
            result.answer = response.content if hasattr(response, 'content') else str(response)
            
            # Total time
            result.total_time = result.retrieval_time + result.llm_time
            
            # Token hesaplama
            result.input_tokens = ModelManager.estimate_tokens(prompt)
            result.output_tokens = ModelManager.estimate_tokens(result.answer)
            result.total_tokens = result.input_tokens + result.output_tokens
            
            # Maliyet hesaplama
            result.estimated_cost = ModelManager.calculate_cost(
                model_name,
                result.input_tokens,
                result.output_tokens
            )
            
            # Evaluation
            eval_result = self.evaluator.evaluate(result.answer, ideal_answer)
            result.final_score = eval_result.final_score
            result.semantic_score = eval_result.semantic_score
            result.bert_f1 = eval_result.bert_f1
            result.rouge_l = eval_result.rouge_l
            result.keyword_f1 = eval_result.keyword_f1
            
        except Exception as e:
            result.error = str(e)
            result.answer = f"HATA: {str(e)}"
            print(f"⚠️ Test hatası: {e}")
        
        return result
    
    def _log_test_result(self, result: TestResult, question_num: int, total: int):
        """Test sonucunu logla."""
        scenario_str = "Shared" if result.scenario == 1 else "Own"
        
        print(f"  📝 Soru {question_num}/{total}: {result.question[:50]}...")
        print(f"     ⏱️  Retrieval: {result.retrieval_time:.2f}s | LLM: {result.llm_time:.2f}s")
        print(f"     🎯 Score: {result.final_score:.1f}/100 (Semantic: {result.semantic_score:.1f})")
        print(f"     💰 Token: {result.total_tokens} | Cost: ${result.estimated_cost:.6f}")
        
        if result.error:
            print(f"     ❌ Error: {result.error}")
    
    def _cleanup_after_model(self, model_name: str):
        """Model sonrası temizlik yap."""
        cooldown = HARDWARE_CONFIG.get("cooldown_time", 3)
        
        print(f"\n⏳ {model_name} testi tamamlandı. Temizlik yapılıyor...")
        
        gc.collect()
        time.sleep(cooldown)
        
        self.hw_monitor.log_stats(f"{model_name} sonrası")
    
    def run_full_benchmark(
        self,
        test_df: pd.DataFrame,
        scenarios: List[int] = None,
        models: List[str] = None,
        k: int = 3,
        progress_callback = None
    ) -> Dict[str, List[TestResult]]:
        """
        Tam benchmark çalıştır (her iki senaryo).
        
        Args:
            test_df: Test DataFrame
            scenarios: Çalıştırılacak senaryolar [1, 2]
            models: Test edilecek modeller
            k: Retriever top-k
            progress_callback: İlerleme callback
            
        Returns:
            {scenario_1: results, scenario_2: results}
        """
        if scenarios is None:
            scenarios = [1, 2]
        
        all_results = {}
        
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"🚀 FULL BENCHMARK BAŞLIYOR")
        print(f"{'='*70}")
        print(f"Senaryolar: {scenarios}")
        print(f"Toplam soru: {len(test_df)}")
        self.hw_monitor.log_stats("Başlangıç")
        print(f"{'='*70}\n")
        
        # Scenario 1
        if 1 in scenarios:
            results_1 = self.run_scenario_1(test_df, models, k, progress_callback)
            all_results["scenario_1"] = results_1
        
        # Scenario 2
        if 2 in scenarios:
            results_2 = self.run_scenario_2(test_df, models, k, progress_callback)
            all_results["scenario_2"] = results_2
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✅ BENCHMARK TAMAMLANDI")
        print(f"{'='*70}")
        print(f"Toplam süre: {total_time:.1f} saniye ({total_time/60:.1f} dakika)")
        self.hw_monitor.log_stats("Bitiş")
        print(f"{'='*70}\n")
        
        return all_results
    
    def generate_summary(
        self, 
        results: Dict[str, List[TestResult]]
    ) -> pd.DataFrame:
        """
        Sonuçlardan özet tablo oluştur.
        
        Returns:
            Özet DataFrame
        """
        summary_data = []
        
        for scenario_name, scenario_results in results.items():
            scenario_num = 1 if "1" in scenario_name else 2
            
            # Model bazında grupla
            model_groups = {}
            for r in scenario_results:
                if r.model_name not in model_groups:
                    model_groups[r.model_name] = []
                model_groups[r.model_name].append(r)
            
            # Her model için özet
            for model_name, model_results in model_groups.items():
                n = len(model_results)
                if n == 0:
                    continue
                
                summary_data.append({
                    'Model': model_name,
                    'Scenario': scenario_num,
                    'Questions': n,
                    'Avg Score': sum(r.final_score for r in model_results) / n,
                    'Avg Semantic': sum(r.semantic_score for r in model_results) / n,
                    'Avg BERT': sum(r.bert_f1 for r in model_results) / n,
                    'Avg ROUGE': sum(r.rouge_l for r in model_results) / n,
                    'Avg Time (s)': sum(r.total_time for r in model_results) / n,
                    'Total Tokens': sum(r.total_tokens for r in model_results),
                    'Total Cost ($)': sum(r.estimated_cost for r in model_results),
                    'Avg RAM (%)': sum(r.ram_percent for r in model_results) / n,
                    'Errors': sum(1 for r in model_results if r.error)
                })
        
        return pd.DataFrame(summary_data)
    
    def generate_comparison(
        self, 
        results: Dict[str, List[TestResult]]
    ) -> pd.DataFrame:
        """
        Scenario 1 vs Scenario 2 karşılaştırması oluştur.
        
        Returns:
            Delta analizi DataFrame
        """
        if "scenario_1" not in results or "scenario_2" not in results:
            return pd.DataFrame()
        
        # Model bazında grupla
        s1_groups = {}
        for r in results["scenario_1"]:
            if r.model_name not in s1_groups:
                s1_groups[r.model_name] = []
            s1_groups[r.model_name].append(r)
        
        s2_groups = {}
        for r in results["scenario_2"]:
            if r.model_name not in s2_groups:
                s2_groups[r.model_name] = []
            s2_groups[r.model_name].append(r)
        
        comparison_data = []
        
        # Sadece her iki senaryoda da olan modeller
        common_models = set(s1_groups.keys()) & set(s2_groups.keys())
        
        for model_name in common_models:
            s1 = s1_groups[model_name]
            s2 = s2_groups[model_name]
            
            n1, n2 = len(s1), len(s2)
            if n1 == 0 or n2 == 0:
                continue
            
            s1_score = sum(r.final_score for r in s1) / n1
            s2_score = sum(r.final_score for r in s2) / n2
            
            s1_time = sum(r.total_time for r in s1) / n1
            s2_time = sum(r.total_time for r in s2) / n2
            
            comparison_data.append({
                'Model': model_name,
                'Shared Score': s1_score,
                'Own Score': s2_score,
                'Score Delta': s2_score - s1_score,
                'Shared Time': s1_time,
                'Own Time': s2_time,
                'Time Delta': s2_time - s1_time,
                'Improvement': 'Yes' if s2_score > s1_score else 'No'
            })
        
        return pd.DataFrame(comparison_data)
