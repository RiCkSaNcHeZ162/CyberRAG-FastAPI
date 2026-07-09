# 🚀 CyberRAG - GenAI Powered Cyber Security Audit Intelligence

A multimodal Retrieval-Augmented Generation (RAG) assistant designed to answer questions across leading cybersecurity and regulatory frameworks, including DORA, NIS2 Directive, RBI IT Outsourcing Guidelines, RBI Operational Resilience Guidance Note, ISO/IEC 27000, and ISO/IEC 27002. The system enables grounded question answering over text, tables, and images extracted from regulatory documents, leveraging multimodal retrieval to provide accurate, context-aware responses with references to the original content.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CyberRag System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ PDF Upload   │───►│  Image/Table │───►│  Smart           │   │
│  │ (pymupdf4llm)│    │  Summary Gen │    │  Chunking        │   │
│  └──────────────┘    └──────────────┘    └─────── ┬─────────┘   │
│                                                   │             │
│                                          ┌────────▼─────────┐   │
│                                          │   Embedding      │   │
│                                          │   (BGE Model)    │   │
│                                          └────────┬─────────┘   │
│                                                   │             │
│                                          ┌────────▼─────────┐   │
│                                          │   FAISS Vector   │   │
│                                          │   Store + BM25   │   │
│                                          └────────┬─────────┘   │
│                                                   │             │
│  ┌──────────────┐    ┌──────────────┐    ┌────────▼─────────┐   │
│  │  User Query  │───►│  Query       │───►│  Hybrid          │   │
│  │              │    │  Rewriter    │    │  Retrieval       │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                   │             │
│                      ┌──────────────┐    ┌────────▼─────────┐   │
│                      │  LLM Answer  │◄───│  Image and table │   │
│                      │  Generation  │    │  query ingestion │   │
│                      └──────────────┘    └──────────────────┘   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Conversation│    │  MULTIMODAL  │    │  Streaming       │   │
│  │  Memory      │    │  RAG         │    │  Responses       │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘                                                                
```

---

## ⭐ Features (6 Advanced RAG Components)

| # | Component | Description |
|---|-----------|-------------|
| 1 | **Smart Chunking** | Semantic, Recursive, and Context-Aware splitting |
| 2 | **Summary Generation**| LLM-powered summary generation for images, tables extracted to store in vectordb for faster retrieval|
| 3 | **BGE Embeddings** | BAAI/bge-small-en-v1.5 (free, local, ~130MB) |
| 4 | **FAISS Vector Store** | High-performance local similarity search with persistence |
| 5 | **Hybrid Retrieval** | Vector + BM25 keyword search with Reciprocal Rank Fusion |
| 6 | **Query Rewriting** | LLM-powered expansion, reformulation, and HyDE |

---

## ⭐ Chunking strategy for visual content: Context-Aware Retrieval for Images and Tables
To enable effective retrieval of visual content, the pipeline processes every extracted image and table from the source PDF using a multimodal LLM to generate a concise semantic summary describing its contents. Instead of embedding the raw visual data, these summaries are converted into vector embeddings and indexed in the vector store, while the original images and tables are stored separately with unique document identifiers. During retrieval, the similarity search operates on the embedded summaries to identify the most relevant visual content. The associated document identifier is then used to fetch the original image or table, allowing the retrieved context to include both the generated summary and the corresponding visual artifact. This approach combines the semantic search capabilities of text embeddings with the fidelity of the original visual content, enabling the downstream multimodal LLM to reason over both textual descriptions and the actual images or tables when generating responses.

---
## 📁 Project Structure

```
PDFGPT/
├── app/
│   ├── main.py                          # FastAPI application
│   ├── config.py                        # Pydantic settings
│   ├── api/
│   │   ├── schemas.py                   # Request/response models
│   │   ├── deps.py                      # Dependency injection
│   │   └── routes/
│   │       ├── documents.py             # PDF upload/delete/list
│   │       ├── query.py                 # Query/stream/agent/evaluate
│   │       └── health.py               # Health check
│   ├── core/
│   │   ├── chunking/
│   │   │   ├── semantic_chunker.py      # Embedding-based boundary detection
│   │   │   ├── recursive_chunker.py     # Hierarchical splitting with overlap
│   │   │   ├── summary_chunker.py       # returns summary of images and tables with metadata
│   │   │   └── context_aware.py         # Structure-preserving chunking
│   │   ├── embeddings/
│   │   │   └── embedding_manager.py     # BGE embedding singleton
│   │   ├── vectorstore/
│   │   │   └── faiss_store.py           # FAISS with metadata & persistence
│   │   ├── media/
│   │   │   └── media_handler.py         # saves and extracts images and tables stored on disk
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py      # Vector + BM25 + RRF
│   │   │   └── query_rewriter.py        # Query expansion + HyDE
│   │   ├── summary/
│   │   │   └── summary.py               # LLM-based summary generation
│   │   └── llm/
│   │       └── llm_manager.py           # Groq/Ollama LLM wrapper
│   └── pipeline/
│       ├── ingestion.py                 # PDF → chunks → vectors
│       └── query_pipeline.py            # query → retrieve → answer
├── data/
│   ├── uploads/                         # Uploaded PDFs
│   └── vectorstores/                    # Persisted FAISS indexes
├── .env.example                         # Environment template
├── requirements.txt                     # Dependencies
├── run.py                               # Entry point
└── README.md                            # You are here
```

---
## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Free Groq API Key** from [console.groq.com](https://console.groq.com)

### 1. Clone & Install

```bash
cd "d:\ML Projects\PDFGPT"

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the template
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# Edit .env and add your Groq API key
# GROQ_API_KEY=gsk_your_key_here
```

### 3. Run the Server

```bash
python run.py
```

Server starts at: **http://localhost:8000**

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 4. First-Time Model Download

On first startup, the system downloads:
- **BGE Embedding Model** (~130MB) — runs locally
- **Cross-Encoder Reranker** (~80MB) — runs locally

This only happens once. Subsequent startups are fast.

---

## 📡 API Reference

### Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/upload` | Upload & process a PDF |
| `GET` | `/api/documents` | List all documents |
| `DELETE` | `/api/documents/{doc_id}` | Delete a document |

### Querying

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Standard RAG query (full pipeline) |
| `POST` | `/api/query/stream` | Streaming response (SSE) |

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/query/sessions` | List conversation sessions |
| `GET` | `/api/query/sessions/{id}/history` | Get session history |
| `DELETE` | `/api/query/sessions/{id}` | Clear session |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |

---

## ⚙️ Configuration Reference

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | LLM provider: `groq` or `ollama` |
| `GROQ_API_KEY` | — | Free API key from console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `CHUNK_SIZE` | `512` | Target chunk size (characters) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `CHUNKING_STRATEGY` | `semantic` | Default: `semantic`, `recursive`, `context_aware` |
| `TOP_K_RETRIEVAL` | `20` | Candidates to retrieve |
| `TOP_K_RERANK` | `5` | Top results after re-ranking |
| `HYBRID_ALPHA` | `0.5` | Vector vs keyword weight (0=keyword, 1=vector) |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `CORS_ORIGINS` | `["localhost:3000","localhost:5173"]` | Allowed frontend origins |

---

## 🔧 Using Ollama (Fully Local, No API Key)

To run everything locally without any API key:

1. Install Ollama: https://ollama.ai
2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Update `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```

---

## 📊 How the Pipeline Works

### Ingestion Pipeline

```
PDF ──► pymupdf4llm (text + tables + images) ──► summary generation ──► Smart Chunking ──► BGE Embeddings ──► FAISS Store
```

### Query Pipeline (8 stages)

```
1. Context Resolution   │ Resolve pronouns using conversation history
2. Query Rewriting      │ LLM improves vague queries
3. Hybrid Retrieval     │ FAISS (semantic) + BM25 (keyword) + RRF merge
```
---

## 🆓 All Free Resources Used

| Component | Resource | Cost |
|-----------|----------|------|
| LLM | Groq API (llama-3.3-70b) | **Free** (rate limited) |
| Embeddings | BAAI/bge-small-en-v1.5 | **Free** (local) |
| Vector DB | FAISS | **Free** (local) |
| BM25 | rank-bm25 | **Free** (local) |
| PDF Parse | pymupdf4llm | **Free** (local) |

---

## 📝 License

MIT License - feel free to use for any purpose.
