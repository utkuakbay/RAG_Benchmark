"""
Model yapılandırmaları ve embedding eşleştirmeleri.
"""

from typing import Dict, Any, Optional

# Shared embedding modeli (Scenario 1 için tüm modeller bu embedding'i kullanır)
SHARED_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"

# Model-specific embedding eşleştirmeleri (Scenario 2 için)
EMBEDDING_MODELS = {
    "Llama": "BAAI/bge-large-en-v1.5",
    "Mistral": "sentence-transformers/all-mpnet-base-v2",
    "Phi": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Qwen": "BAAI/bge-base-en-v1.5",
}

# Model yapılandırmaları
MODEL_CONFIG = {
    # API Modeller
    "Gemini": {
        "type": "api",
        "provider": "google",
        "model_name": "gemini-2.5-flash",  # Stabil versiyon, daha yüksek kota
        "temperature": 0.0,
        "scenario_1_embedding": SHARED_EMBEDDING,
        "scenario_2_embedding": None,  # API modeller için own scenario yok
        "pricing": {
            "input": 0.075 / 1_000_000,   # $0.075 per 1M input tokens
            "output": 0.30 / 1_000_000,   # $0.30 per 1M output tokens
        }
    },
    "GPT": {
        "type": "api",
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.0,
        "scenario_1_embedding": SHARED_EMBEDDING,
        "scenario_2_embedding": None,
        "pricing": {
            "input": 0.50 / 1_000_000,    # $0.50 per 1M input tokens
            "output": 1.50 / 1_000_000,   # $1.50 per 1M output tokens
        }
    },
    "Claude": {
        "type": "api",
        "provider": "anthropic",
        "model_name": "claude-3-5-haiku-latest",  # En hızlı ve ucuz Claude
        "temperature": 0.0,
        "scenario_1_embedding": SHARED_EMBEDDING,
        "scenario_2_embedding": None,
        "pricing": {
            "input": 0.25 / 1_000_000,    # $0.25 per 1M input tokens
            "output": 1.25 / 1_000_000,   # $1.25 per 1M output tokens
        }
    },
    
    # Local Modeller (Ollama)
    "Llama": {
        "type": "local",
        "provider": "ollama",
        "model_name": "llama3.1:8b",
        "temperature": 0.0,
        "scenario_1_embedding": SHARED_EMBEDDING,
        "scenario_2_embedding": EMBEDDING_MODELS["Llama"],
        "model_size": "8B",
        "pricing": {
            "input": 0.0,
            "output": 0.0,
        }
    },
    "Mistral": {
        "type": "local",
        "provider": "ollama",
        "model_name": "mistral:7b",
        "temperature": 0.0,
        "scenario_1_embedding": SHARED_EMBEDDING,
        "scenario_2_embedding": EMBEDDING_MODELS["Mistral"],
        "model_size": "7B",
        "pricing": {
            "input": 0.0,
            "output": 0.0,
        }
    },
    "Phi": {
        "type": "local",
        "provider": "ollama",
        "model_name": "phi3:mini",
        "temperature": 0.0,
        "scenario_1_embedding": SHARED_EMBEDDING,
        "scenario_2_embedding": EMBEDDING_MODELS["Phi"],
        "model_size": "3.8B",
        "pricing": {
            "input": 0.0,
            "output": 0.0,
        }
    },
    "Qwen": {
        "type": "local",
        "provider": "ollama",
        "model_name": "qwen2:7b",
        "temperature": 0.0,
        "scenario_1_embedding": SHARED_EMBEDDING,
        "scenario_2_embedding": EMBEDDING_MODELS["Qwen"],
        "model_size": "7B",
        "pricing": {
            "input": 0.0,
            "output": 0.0,
        }
    },
}

# LLM Judge yapılandırması
LLM_JUDGE_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "sample_rate": 0.20,  # %20 sample ile cross-validation
    "pricing": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    }
}

# Hardware monitoring ayarları
HARDWARE_CONFIG = {
    "ram_warning_threshold": 85,    # %85'te uyarı ver
    "ram_critical_threshold": 90,   # %90'da testi durdur
    "check_interval": 5,            # Her 5 saniyede bir kontrol
    "cooldown_time": 3,             # Model arası bekleme süresi (saniye)
}

# Evaluation ağırlıkları
EVALUATION_WEIGHTS = {
    "semantic": 0.60,   # Semantic Similarity - PRIMARY
    "bert": 0.30,       # BERTScore - SECONDARY
    "rouge": 0.10,      # ROUGE-L - TERTIARY
}


def get_model_embedding(model_name: str, scenario: int = 1) -> Optional[str]:
    """
    Model için embedding modelini döndür.
    
    Args:
        model_name: Model adı (Gemini, GPT, Llama, Mistral, Phi, Qwen)
        scenario: Senaryo numarası (1 = shared, 2 = own)
        
    Returns:
        Embedding model adı veya None
    """
    if model_name not in MODEL_CONFIG:
        return None
    
    config = MODEL_CONFIG[model_name]
    
    if scenario == 1:
        return config.get("scenario_1_embedding")
    elif scenario == 2:
        return config.get("scenario_2_embedding")
    
    return None


def get_local_models() -> Dict[str, Any]:
    """
    Sadece local (Ollama) modelleri döndür.
    
    Returns:
        Local model yapılandırmaları
    """
    return {
        name: config 
        for name, config in MODEL_CONFIG.items() 
        if config.get("type") == "local"
    }


def get_api_models() -> Dict[str, Any]:
    """
    Sadece API modellerini döndür.
    
    Returns:
        API model yapılandırmaları
    """
    return {
        name: config 
        for name, config in MODEL_CONFIG.items() 
        if config.get("type") == "api"
    }


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Model için tahmini maliyeti hesapla.
    
    Args:
        model_name: Model adı
        input_tokens: Input token sayısı
        output_tokens: Output token sayısı
        
    Returns:
        Tahmini maliyet (USD)
    """
    if model_name not in MODEL_CONFIG:
        return 0.0
    
    pricing = MODEL_CONFIG[model_name].get("pricing", {})
    input_cost = input_tokens * pricing.get("input", 0)
    output_cost = output_tokens * pricing.get("output", 0)
    
    return input_cost + output_cost
