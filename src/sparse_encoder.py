"""
High-Performance Lexical Sparse Vector Generator & Asymmetric Task Instruction Formatter.
Implements native BM25/lexical term hashing for Qdrant RRF hybrid search and handles
BGE/Gemma asymmetric instruction prefixes.
"""

import re
import zlib
from typing import Tuple, List, Dict
from qdrant_client.http import models as qmodels

# Common stop words to reduce noise while preserving all numbers and entities
STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "am", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "what", "when", "where", "who", "which", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "can", "will", "just", "should", "now"
}

# Regex to capture words, prices with currencies ($49, ₹6,000, €99), numbers, and alphanumeric IDs
TOKEN_PATTERN = re.compile(r'[\$₹€£]?\d+(?:[\.,]\d+)?%?|[a-zA-Z0-9_\-]+')


def compute_sparse_vector(text: str) -> qmodels.SparseVector:
    """
    Computes a deterministic lexical sparse vector for Qdrant Hybrid Search.
    Features heavy weighting (3.0x) on exact prices, numbers, and proper entities.
    """
    if not text:
        return qmodels.SparseVector(indices=[0], values=[1.0])

    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    if not raw_tokens:
        return qmodels.SparseVector(indices=[0], values=[1.0])

    tf: Dict[str, float] = {}
    for t in raw_tokens:
        clean_t = t.strip()
        if not clean_t:
            continue
        # Skip generic stop words unless they contain digits/currencies
        if clean_t in STOP_WORDS and not re.search(r'\d', clean_t):
            continue

        weight = 1.0
        # Boost numbers, percentages, and currencies by 3.0x
        if re.search(r'\d', clean_t) or any(c in clean_t for c in "$₹€£%"):
            weight = 3.0
        elif len(clean_t) > 6:
            weight = 1.4  # Slightly boost specific domain terms

        tf[clean_t] = tf.get(clean_t, 0.0) + weight

    indices: List[int] = []
    values: List[float] = []

    for token, weight in tf.items():
        # Deterministic 31-bit CRC32 hash into positive integer space
        idx = zlib.crc32(token.encode("utf-8")) % 2147483647
        indices.append(idx)
        values.append(weight)

    # Sort and deduplicate by index (required by Qdrant sparse vectors)
    pairs = sorted(zip(indices, values), key=lambda x: x[0])
    dedup_idx: List[int] = []
    dedup_val: List[float] = []

    for i, v in pairs:
        if dedup_idx and dedup_idx[-1] == i:
            dedup_val[-1] += v
        else:
            dedup_idx.append(i)
            dedup_val.append(round(v, 3))

    if not dedup_idx:
        return qmodels.SparseVector(indices=[0], values=[1.0])

    return qmodels.SparseVector(indices=dedup_idx, values=dedup_val)


def format_asymmetric_query(query: str, model_id: str) -> Tuple[str, bool]:
    """
    Applies model-specific asymmetric task instruction prefixes.
    Returns:
        Tuple of (formatted_query, was_prefixed)
    """
    clean_query = query.strip()
    mid = model_id.lower()

    if "bge" in mid:
        # BAAI standard asymmetric search query instruction
        prefix = "Represent this sentence for searching relevant passages: "
        if not clean_query.startswith(prefix):
            return f"{prefix}{clean_query}", True
        return clean_query, False

    elif "gemma" in mid:
        # Google EmbeddingGemma task instruction
        prefix = "task: search query | query: "
        if not clean_query.startswith(prefix):
            return f"{prefix}{clean_query}", True
        return clean_query, False

    return clean_query, False


def format_asymmetric_document(doc_text: str, model_id: str) -> str:
    """
    Formats document text for asymmetric models.
    BGE models require raw unprefixed document chunks.
    Gemma models require 'task: search document | document: '.
    """
    clean_doc = doc_text.strip()
    mid = model_id.lower()

    if "gemma" in mid:
        prefix = "task: search document | document: "
        if not clean_doc.startswith(prefix):
            return f"{prefix}{clean_doc}"

    # BGE and standard models embed documents raw
    return clean_doc
