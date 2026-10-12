"""
Configuration & Model Registry for Cloudflare Workers AI Embeddings & Qdrant Vector Space.
"""

import os
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chunking_studio", ".env"))

# Cloudflare Credentials
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")

# Local Qdrant Storage (Embedded on disk, zero Docker needed)
DEFAULT_QDRANT_STORAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "qdrant_storage"
)

# Ingestion & Benchmark Defaults
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "512"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "16"))


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Specification for a Cloudflare Workers AI embedding model."""
    id: str
    display_name: str
    dimensions: int
    languages: str
    max_tokens: int
    collection_name: str
    description: str
    badge_color: str
    is_multilingual: bool = True
    provider: str = "Cloudflare Workers AI"


# Full catalog of ALL Cloudflare Workers AI text embedding models
ALL_CLOUDFLARE_MODELS: Dict[str, EmbeddingModelConfig] = {
    "@cf/baai/bge-m3": EmbeddingModelConfig(
        id="@cf/baai/bge-m3",
        display_name="BGE-M3 (Flagship Multilingual)",
        dimensions=1024,
        languages="100+ Languages (Universal)",
        max_tokens=8192,
        collection_name="cf_baai_bge_m3",
        description="Multi-functional, multi-lingual, and multi-granularity flagship model from BAAI. Premier dense multilingual retrieval.",
        badge_color="#F38020",
        is_multilingual=True
    ),
    "@cf/google/embeddinggemma-300m": EmbeddingModelConfig(
        id="@cf/google/embeddinggemma-300m",
        display_name="EmbeddingGemma-300M (Google)",
        dimensions=768,
        languages="100+ Languages (Google Gemma 3)",
        max_tokens=2048,
        collection_name="cf_google_embeddinggemma_300m",
        description="High-efficiency 300M parameter model built from Google's Gemma 3 architecture, fine-tuned for multilingual search.",
        badge_color="#4285F4",
        is_multilingual=True
    ),
    "@cf/qwen/qwen3-embedding-0.6b": EmbeddingModelConfig(
        id="@cf/qwen/qwen3-embedding-0.6b",
        display_name="Qwen3-Embedding-0.6B (Alibaba)",
        dimensions=1024,
        languages="100+ Languages (Cross-Lingual)",
        max_tokens=8192,
        collection_name="cf_qwen_qwen3_embedding_0_6b",
        description="Lightweight 0.6B model from Alibaba Qwen series optimized for high-throughput multilingual ranking and semantic similarity.",
        badge_color="#7B1FA2",
        is_multilingual=True
    ),
    "@cf/pfnet/plamo-embedding-1b": EmbeddingModelConfig(
        id="@cf/pfnet/plamo-embedding-1b",
        display_name="PLaMo-Embedding-1B (PFN Multilingual)",
        dimensions=1024,
        languages="Japanese, English & Multilingual",
        max_tokens=2048,
        collection_name="cf_pfnet_plamo_embedding_1b",
        description="Preferred Networks 1B parameter embedding model specializing in Japanese & multilingual nuance and cross-lingual retrieval.",
        badge_color="#00897B",
        is_multilingual=True
    ),
    "@cf/baai/bge-large-en-v1.5": EmbeddingModelConfig(
        id="@cf/baai/bge-large-en-v1.5",
        display_name="BGE-Large-EN-v1.5 (BAAI High-Capacity)",
        dimensions=1024,
        languages="English (High-Capacity)",
        max_tokens=512,
        collection_name="cf_baai_bge_large_en_v1_5",
        description="High-capacity 1024-dimensional English dense embedding model with superior semantic separation.",
        badge_color="#E65100",
        is_multilingual=False
    ),
    "@cf/baai/bge-base-en-v1.5": EmbeddingModelConfig(
        id="@cf/baai/bge-base-en-v1.5",
        display_name="BGE-Base-EN-v1.5 (BAAI Balanced)",
        dimensions=768,
        languages="English (Balanced)",
        max_tokens=512,
        collection_name="cf_baai_bge_base_en_v1_5",
        description="Standard 768-dimensional English embedding model balancing inference speed and retrieval precision.",
        badge_color="#FB8C00",
        is_multilingual=False
    ),
    "@cf/baai/bge-small-en-v1.5": EmbeddingModelConfig(
        id="@cf/baai/bge-small-en-v1.5",
        display_name="BGE-Small-EN-v1.5 (BAAI Ultra-Fast)",
        dimensions=384,
        languages="English (Lightweight)",
        max_tokens=512,
        collection_name="cf_baai_bge_small_en_v1_5",
        description="Ultra-fast 384-dimensional English embedding model designed for low-latency edge deployment.",
        badge_color="#FFA726",
        is_multilingual=False
    ),
}

AVAILABLE_MODEL_IDS: List[str] = list(ALL_CLOUDFLARE_MODELS.keys())
MULTILINGUAL_MODEL_IDS: List[str] = [m for m, cfg in ALL_CLOUDFLARE_MODELS.items() if cfg.is_multilingual]
