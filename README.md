# Perfect RAG — Context-Augmented Voice AI & Vector Chunking Engine

A high-performance, deterministic & LLM-assisted document chunking engine designed for Voice AI pipelines and Vector Embeddings (Cloudflare Vectorize, Qdrant, Pinecone).

## 🚀 Key Features

- **2-Stage Semantic Architecture**:
  - **Stage 1 (LLM Boundary Tagging)**: Powered by Google Gemini (`gemini-3.8-flash`) to identify semantic entity boundaries across complex, messy, and multilingual documents (English, Tamil, Hindi, etc.) without hallucinations.
  - **Stage 2 (Deterministic Chunking & Token Ceilings)**: Enforces hard ceilings (e.g. Cloudflare's 512-token limit), protects child attributes from artificial splits, and formats clean text payloads.
- **Pure Text Payloads**: Zero markdown or bracket pollution in embedding text payloads (`c.text`) for maximum vector retrieval accuracy.
- **Structured Vector DB Metadata**: Stores `branch_name`, `chunk_type`, `char_count`, `token_count`, `parent_entity`, and `hard_ceiling` strictly in the metadata payload for Qdrant / Vectorize.
- **Context Header Anchoring**: Oversized listings that exceed the ceiling automatically inject contextual parent entity tags (`[Entity (Part X/Y)]`) so semantic relevance is never lost.
- **Specialized Chunking Engines**:
  - `PricingTableChunker`: Table preservation with column header retention.
  - `LocationEntityChunker`: Indivisible entity card chunking for branch directories, clinics, and real estate listings.
  - `AtomicFAQChunker`: 1:1 Q&A atomic pair chunker.
  - `SemanticSentenceChunker`: Rolling semantic sentence window with overlap.
  - `HybridCompositeChunker`: Automatic multi-zone classifier and dispatcher for mixed documents.

## 🛠️ Quick Start

### 1. Prerequisites
- Python 3.10+
- (Optional) Google Gemini API Key

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/prathyush1019/Perfect_RAG.git
cd Perfect_RAG

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your GEMINI_API_KEY in .env if using LLM normalization
```

### 3. Launching the Web Studio
- **Clean HTML Studio (Single-page app)**:
  ```bash
  .\run_html.bat
  ```
  Open `http://localhost:8080` in your browser.

- **Streamlit Version**:
  ```bash
  .\run.bat
  ```

### 4. Running Tests
```bash
python chunking_studio/tests/test_studio.py
```
