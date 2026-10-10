import re
from typing import Dict, Any, Tuple
from .base import BaseChunker, ChunkResult
from .context_ag import (
    build_searchable_augmented_text,
    clean_for_speech,
    calculate_speech_seconds,
    format_context_header
)
from .pricing_table import PricingTableChunker
from .location_entity import LocationEntityChunker
from .atomic_faq import AtomicFAQChunker
from .semantic_text import SemanticSentenceChunker
from .hybrid_composite import HybridCompositeChunker


def detect_document_strategy(text: str) -> Tuple[str, str]:
    """
    Analyzes document text structure and returns:
    (strategy_name, reason_explanation)
    """
    if not text or not text.strip():
        return "semantic", "Empty or minimal text; using semantic chunker."

    lower = text.lower()

    # Feature counts
    has_table = bool(re.search(r"^[ \t]*\|[^\n]+\|[ \t]*\n[ \t]*\|(?:\s*:?-+:?\s*\|)+", text, re.MULTILINE))
    faq_count = len(re.findall(r"(?:^|\n)(?:#{1,4}\s*(?:Question|Q\b)|(?:Question|Q)\s*(?:#?\d+)?\s*[:\.\?\-]|\d+\.\s*(?:Q|Question)\s*[:\.\?\-])", text, re.IGNORECASE))
    # Entity directory pattern (Real estate properties, Clinic branches, Store locations)
    child_keys = r"(?:Location|Address|Directions?|Landmarks?|Operating Hours|Hours|Timings?|Phone|Mobile|Contact|Services?|Facilities|Approval|Nearby locations?|Construction Details?|Main Content|Guideline Value|Price|Pricing|Rate|Cost)"
    branch_count = len(re.findall(
        r"(?:^|\n)(?:"
        r"#{1,4}\s*(?:Branch|Location|Clinic|Centre|Center|Property|Project|Site|Plot)\b[^\n]*|"
        r"(?:Branch|Property|Project|Site|Plot|Clinic)\s*#?\d+\s*[:\.\-]?[^\n]*|"
        r"\d+[\.\)]\s*(?!" + child_keys + r"\s*:)[^\n]+"
        r")",
        text,
        re.IGNORECASE | re.UNICODE
    ))
    has_directory_features = bool(re.search(
        r"(?:location|address|facilities|landmarks?|operating hours|phone|guideline value|bhk|sqft|square feet|approval)",
        lower
    ))
    section_headers = len(re.findall(r"(?:^|\n)#{1,3}\s+[^\n]+", text))

    # Count independent content types
    types_present = 0
    if has_table:
        types_present += 1
    if faq_count >= 2:
        types_present += 1
    if branch_count >= 2 and has_directory_features:
        types_present += 1

    # Check if there are distinct non-branch, non-faq prose headers
    prose_headers = max(0, section_headers - (branch_count + faq_count))
    if prose_headers >= 2:
        types_present += 1

    if types_present >= 2:
        return "hybrid_composite", "Mixed document detected (multiple content zones found: tables, directories, FAQs, or prose)."

    if has_table and faq_count == 0 and branch_count == 0:
        return "pricing_table", "Dominant Markdown/ASCII pricing table structure detected."

    if (branch_count >= 2 and has_directory_features) or ("branch" in lower and "hours" in lower and "phone" in lower):
        return "location", f"Detected {max(branch_count, 1)} discrete branch / property directory cards."

    if faq_count >= 2:
        return "faq", f"Detected {faq_count} discrete Question & Answer pairs."

    return "semantic", "Prose/policy narrative detected; applying semantic sentence chunking."


def get_chunker(strategy: str, **kwargs) -> BaseChunker:
    """Factory helper to instantiate chunker by name."""
    s = strategy.lower()
    unit = kwargs.get("unit", "characters")
    chunk_size = kwargs.get("chunk_size")
    overlap = kwargs.get("overlap")

    if "price" in s or "table" in s:
        return PricingTableChunker(rows_per_chunk=kwargs.get("rows_per_chunk", 1))
    elif "loc" in s or "branch" in s:
        return LocationEntityChunker()
    elif "faq" in s or "atomic" in s:
        return AtomicFAQChunker()
    elif "hybrid" in s or "composite" in s:
        return HybridCompositeChunker(
            unit=unit,
            chunk_size=chunk_size,
            overlap=overlap,
            target_words=kwargs.get("target_words", 150),
            overlap_words=kwargs.get("overlap_words", 25),
            rows_per_chunk=kwargs.get("rows_per_chunk", 1)
        )
    else:
        return SemanticSentenceChunker(
            unit=unit,
            chunk_size=chunk_size,
            overlap=overlap,
            target_words=kwargs.get("target_words", 150),
            overlap_words=kwargs.get("overlap_words", 25)
        )


__all__ = [
    "BaseChunker",
    "ChunkResult",
    "PricingTableChunker",
    "LocationEntityChunker",
    "AtomicFAQChunker",
    "SemanticSentenceChunker",
    "HybridCompositeChunker",
    "detect_document_strategy",
    "get_chunker",
    "build_searchable_augmented_text",
    "clean_for_speech",
    "calculate_speech_seconds",
    "format_context_header"
]
