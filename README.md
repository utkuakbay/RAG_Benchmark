# RAG Model Selector
<p align="center">
  <img src="https://img.shields.io/badge/RAG-Model_Selector-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMiAxNWwtNS01IDEuNDEtMS40MUwxMCAxNC4xN2w3LjU5LTcuNTlMMTkgOGwtOSA5eiIvPjwvc3ZnPg==" alt="RAG Model Selector"/>
</p>

<h3 align="center">🎯 Find the Perfect LLM for Your RAG System</h3>

<p align="center">
  <strong>Enterprise-Grade Benchmarking Tool for AI Model Selection</strong><br>
  <em>Help individuals and enterprises discover which AI model performs best with their specific data</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square&logo=chainlink&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/FAISS-Vector_DB-00ADD8?style=flat-square&logo=meta&logoColor=white" alt="FAISS"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/API-Gemini_|_GPT_|_Claude-8B5CF6?style=flat-square" alt="API Models"/>
  <img src="https://img.shields.io/badge/Local-Llama_|_Mistral_|_Phi_|_Qwen-F97316?style=flat-square" alt="Local Models"/>
  <img src="https://img.shields.io/badge/Metrics-Semantic_|_BERT_|_ROUGE-10B981?style=flat-square" alt="Metrics"/>
</p>

---

## 📋 Contents

- [About the Project](#-about-the-project)
- [Why RAG Model Selector?](#-why-rag-model-selector)
- [Features](#-features)
- [Technologies](#-technologies)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Evaluation Metrics](#-evaluation-metrics)
- [Sample Results](#-sample-results)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About the Project

**RAG Model Selector** is an AI-powered benchmarking platform that modernizes the LLM selection process for RAG (Retrieval-Augmented Generation) systems. By leveraging scientific evaluation metrics, it helps users identify the optimal model for their specific use case.

Building a RAG system requires choosing the right LLM, but:
- **Which model works best with YOUR specific data?**
- **Should you use expensive API models or free local alternatives?**
- **How do models compare on accuracy vs. speed vs. cost?**

This tool answers these questions with data-driven insights.

---

## 🌟 Why RAG Model Selector?

<table>
<tr>
<td width="50%">

### 🔬 Scientific Evaluation
Three-metric weighted scoring system ensures comprehensive and unbiased model assessment using Semantic Similarity, BERTScore, and ROUGE metrics.

</td>
<td width="50%">

### 🤖 Multi-Model Support
Test 7 different LLMs simultaneously - including Google Gemini, OpenAI GPT, Anthropic Claude, and local models via Ollama.

</td>
</tr>
<tr>
<td width="50%">

### 📊 Interactive Visualizations
Beautiful charts including bar graphs, radar charts, and delta analysis to visualize model performance across multiple dimensions.

</td>
<td width="50%">

### 🧠 AI-Powered Insights
Claude automatically analyzes benchmark results and provides personalized recommendations for your use case.

</td>
</tr>
<tr>
<td width="50%">

### ⚡ Performance Optimized
GPU acceleration, vectorstore caching, and sequential execution ensure fast benchmarks without system overload.

</td>
<td width="50%">

### 💰 Cost-Aware Analysis
Track token usage and API costs in real-time, helping you make budget-conscious decisions.

</td>
</tr>
</table>

---

## 💡 Features

### 👤 For Individual Developers

| Feature | Description |
|---------|-------------|
| **Model Comparison** | Compare API and local models on your own data |
| **Cost Optimization** | Find free local alternatives to expensive API models |
| **Quick Testing** | Test with 8-64 questions for fast results |
| **Export Results** | Download benchmark results as CSV for further analysis |

### 💼 For Enterprises

| Feature | Description |
|---------|-------------|
| **Data Privacy** | Test with local models - your data never leaves your server |
| **Scalable Testing** | Benchmark multiple models systematically |
| **Compliance Ready** | Document model selection decisions with scientific metrics |
| **Team Collaboration** | Share reproducible benchmark results |

### 🔬 Evaluation Capabilities

| Capability | Description |
|------------|-------------|
| **Semantic Similarity** | Embedding-based meaning comparison (60% weight) |
| **BERTScore** | Context-aware token matching - catches synonyms (30% weight) |
| **ROUGE-L** | N-gram overlap for factual accuracy (10% weight) |
| **Hardware Monitoring** | Real-time RAM/CPU tracking with safety thresholds |

---

## 🛠 Technologies

<table>
<tr>
<td align="center" width="96">
  <img src="https://skillicons.dev/icons?i=python" width="48" height="48" alt="Python" />
  <br>Python
</td>
<td align="center" width="96">
  <img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" width="48" height="48" alt="Streamlit" />
  <br>Streamlit
</td>
<td align="center" width="96">
  <img src="https://skillicons.dev/icons?i=pytorch" width="48" height="48" alt="PyTorch" />
  <br>PyTorch
</td>
<td align="center" width="96">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="48" height="48" alt="HuggingFace" />
  <br>HuggingFace
</td>
<td align="center" width="96">
  <img src="https://skillicons.dev/icons?i=github" width="48" height="48" alt="GitHub" />
  <br>GitHub
</td>
</tr>
</table>

### Backend & Core

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **LangChain 0.3+** | RAG framework and model orchestration |
| **FAISS** | Vector similarity search |
| **Sentence Transformers** | Text embeddings |
| **psutil** | Hardware monitoring |

### Frontend & Visualization

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Interactive web UI |
| **Plotly** | Interactive charts (bar, radar) |
| **Pandas** | Data manipulation and export |

### LLM Providers

| Provider | Models | Type |
|----------|--------|------|
| **Google** | Gemini 2.5 Flash | API |
| **OpenAI** | GPT-3.5 Turbo | API |
| **Anthropic** | Claude 3.5 Haiku | API |
| **Ollama** | Llama 3.1, Mistral, Phi-3, Qwen 2 | Local |

### Evaluation Models

| Model | Purpose |
|-------|---------|
| **all-MiniLM-L6-v2** | Shared embedding (Scenario 1) |
| **BGE-Large-EN** | Llama-specific embedding |
| **MPNet-Base** | Mistral-specific embedding |
| **Multilingual-MiniLM** | Phi-specific embedding |
| **BGE-Base-EN** | Qwen-specific embedding |
| **bert-base-multilingual-cased** | BERTScore evaluation |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG MODEL SELECTOR                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │  Corpus CSV │───▶│  Chunking   │───▶│  FAISS Vectorstore      │ │
│  │  (Your Data)│    │  (1000 chr) │    │  (GPU Accelerated)      │ │
│  └─────────────┘    └─────────────┘    └───────────┬─────────────┘ │
│                                                     │               │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────▼─────────────┐ │
│  │  Test CSV   │───▶│  Questions  │───▶│  Retriever (Top-K)      │ │
│  │  (Q&A Pairs)│    │  Iteration  │    │  Context Generation     │ │
│  └─────────────┘    └─────────────┘    └───────────┬─────────────┘ │
│                                                     │               │
│  ┌──────────────────────────────────────────────────▼─────────────┐│
│  │                    LLM GENERATION LAYER                        ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐││
│  │  │ Gemini  │  │   GPT   │  │ Claude  │  │  Ollama (Local)     │││
│  │  │  API    │  │   API   │  │   API   │  │ Llama|Mistral|Phi|Q │││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘││
│  └────────────────────────────┬───────────────────────────────────┘│
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────────┐│
│  │                   UNIFIED EVALUATOR                            ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          ││
│  │  │  Semantic    │  │  BERTScore   │  │   ROUGE-L    │          ││
│  │  │  (60%)       │  │  (30%)       │  │   (10%)      │          ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘          ││
│  └────────────────────────────┬───────────────────────────────────┘│
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────────┐│
│  │                 RESULTS & VISUALIZATION                        ││
│  │  📊 Performance Tables  📈 Interactive Charts  🤖 AI Analysis  ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Data → Chunking → Embedding → FAISS Index → Retrieval → LLM → Evaluation → Results
```

---

## 📂 Project Structure

```
RAG-Model-Selector/
├── 📁 config/
│   ├── __init__.py              # Configuration exports
│   └── model_config.py          # Model configs, pricing, embeddings
│
├── 📁 core/
│   ├── __init__.py              # Core module exports
│   ├── benchmark_runner.py      # Test orchestration engine
│   ├── csv_processor.py         # Data ingestion & chunking
│   ├── hardware_monitor.py      # RAM/CPU monitoring & safety
│   ├── model_manager.py         # LLM loading & invocation
│   ├── rag_pipeline.py          # FAISS vectorstore management
│   │
│   └── 📁 evaluation/
│       ├── __init__.py          # Evaluation exports
│       ├── unified_evaluator.py # Weighted score aggregation
│       ├── semantic_scorer.py   # Sentence Transformers similarity
│       ├── bert_scorer.py       # BERTScore (P/R/F1)
│       ├── rouge_scorer.py      # ROUGE-1, ROUGE-2, ROUGE-L
│       └── keyword_scorer.py    # Keyword extraction & matching
│
├── 📁 cache/
│   └── 📁 vectorstores/         # Cached FAISS indices (gitignored)
│
├── 📄 enhanced_app.py           # Streamlit UI application
├── 📄 requirements.txt          # Python dependencies
├── 📄 env.example               # API key template
├── 📄 .gitignore                # Git ignore rules
├── 📄 LICENSE                   # MIT License
└── 📄 README.md                 # This file
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.8+**
- **Ollama** (for local models) - [Download](https://ollama.ai/)
- **CUDA-compatible GPU** (optional, but recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/rag-model-selector.git
cd rag-model-selector
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

```bash
# Copy example file
cp env.example .env

# Edit .env with your API keys
# GOOGLE_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

### Step 5: Download Local Models (Optional)

```bash
# Start Ollama service
ollama serve

# Download models (in a new terminal)
ollama pull llama3.1:8b      # Meta Llama 3.1
ollama pull mistral:7b       # Mistral 7B
ollama pull phi3:mini        # Microsoft Phi-3
ollama pull qwen2:7b         # Alibaba Qwen 2
```

### Step 6: Run the Application

```bash
streamlit run enhanced_app.py
```

The application will open at `http://localhost:8501`

---

## 📘 Usage Guide

### Step 1: Prepare Your Data

**Corpus CSV** (your knowledge base):
```csv
title,content,category
Product Overview,Our software provides real-time analytics...,Documentation
Installation Guide,To install the application follow these steps...,Technical
FAQ,Common questions include how to reset password...,Support
```

**Test CSV** (evaluation questions with ideal answers):
```csv
soru,ideal_cevap
What does the software do?,The software provides real-time analytics and reporting.
How do I install it?,Follow the installation guide in the documentation.
```

### Step 2: Configure Benchmark

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Test Questions** | 8-64 | 16 | Number of evaluation questions |
| **Chunk Size** | 500-2000 | 1000 | Characters per document chunk |
| **Chunk Overlap** | 0-500 | 200 | Overlap between chunks |
| **Retriever K** | 1-10 | 3 | Documents retrieved per query |

### Step 3: Select Models & Scenarios

**Scenario 1 (Fair Arena):**
- All models use the same embedding (MiniLM-L6-v2)
- Fair comparison of LLM capabilities only

**Scenario 2 (Real World):**
- Each local model uses its optimized embedding
- Simulates production performance

### Step 4: Run & Analyze

1. Click **"Benchmark Başlat"** (Start Benchmark)
2. Monitor progress in real-time
3. Review performance tables
4. Explore interactive charts
5. Read AI-generated recommendations
6. Export results as CSV

---

## 📊 Evaluation Metrics

### Scoring Formula

```
Final Score = (Semantic × 0.60) + (BERT × 0.30) + (ROUGE × 0.10)
```

### Metric Details

| Metric | Weight | What It Measures | Example |
|--------|--------|------------------|---------|
| **Semantic Similarity** | 60% | Meaning-based comparison via embeddings | "car" ≈ "automobile" |
| **BERTScore** | 30% | Context-aware token matching | "doctor" ≈ "physician" |
| **ROUGE-L** | 10% | Longest common subsequence | Exact phrase matching |

### Why These Weights?

- **Semantic (60%)**: Primary metric - captures meaning even with different words
- **BERT (30%)**: Secondary - adds contextual understanding
- **ROUGE (10%)**: Tertiary - rewards factual accuracy but penalizes paraphrasing

---

## 📈 Sample Results

### Performance Summary

| Model | Score | Semantic | BERT | Time | Cost |
|-------|-------|----------|------|------|------|
| Claude 3.5 Haiku | **53.7** | 50.8 | 66.0 | 1.69s | $0.0026 |
| Qwen 2 (7B) | 51.0 | 47.4 | 65.7 | 1.96s | $0.0000 |
| Llama 3.1 (8B) | 50.7 | 43.3 | 65.4 | 1.60s | $0.0000 |
| Gemini 2.5 Flash | 49.8 | 45.5 | 61.5 | 3.79s | $0.0006 |
| Mistral (7B) | 49.9 | 46.7 | 63.3 | 7.68s | $0.0000 |
| Phi-3 (3.8B) | 30.4 | 21.1 | 54.1 | 37.08s | $0.0000 |

### AI Analysis Example

> **🏆 Best Model: Claude 3.5 Haiku**
> 
> Claude achieved the highest overall score (53.7) with excellent semantic similarity (50.8) and BERTScore (66.0). Response time is fast (1.69s).
>
> **💰 Cost-Performance Winner: Qwen 2**
> 
> For cost-conscious deployments, Qwen offers 95% of Claude's performance at zero cost.
>
> **⚠️ Not Recommended: Phi-3**
> 
> Phi-3 shows significantly lower performance (30.4) with very slow response times (37s).

---

## 💰 Cost Estimation

| Model | Input Price | Output Price | ~Cost per 100 Queries |
|-------|-------------|--------------|----------------------|
| Gemini 2.5 Flash | $0.075/1M | $0.30/1M | ~$0.02 |
| GPT-3.5 Turbo | $0.50/1M | $1.50/1M | ~$0.10 |
| Claude 3.5 Haiku | $0.25/1M | $1.25/1M | ~$0.08 |
| **Local Models** | **Free** | **Free** | **$0.00** |

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Optional | For Gemini models |
| `OPENAI_API_KEY` | Optional | For GPT models |
| `ANTHROPIC_API_KEY` | Optional | For Claude models |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16+ GB |
| **GPU** | None | CUDA-compatible |
| **Storage** | 5 GB | 10+ GB (for local models) |

---

## 🤝 Contributing

Contributions are welcome! Follow these steps:

1. **Fork** the repository
2. **Create** a feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update README for significant changes

---

## 👥 Author

| Name | Role | GitHub |
|------|------|--------|
| **Mustafa Utku Akbay** | Developer | [utkuakbay](https://github.com/utkuakbay) |

---

## 📄 License

```
MIT License

Copyright (c) 2026 Mustafa Utku Akbay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - RAG framework
- [Ollama](https://ollama.ai/) - Local model serving
- [Hugging Face](https://huggingface.co/) - Embedding models
- [Streamlit](https://streamlit.io/) - UI framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Plotly](https://plotly.com/) - Interactive visualizations

---

<p align="center">
  <strong>🎯 Stop guessing. Start benchmarking.</strong><br>
  <em>Find the perfect LLM for your RAG system.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge" alt="Made with love"/>
</p>
