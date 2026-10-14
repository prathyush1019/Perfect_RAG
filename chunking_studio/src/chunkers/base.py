import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


def count_tokens(text: str) -> int:
    """Estimates token count (~4 characters per token or word boundary heuristic)."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def count_units(text: str, unit: str = "tokens") -> int:
    """Returns metric according to unit ('characters' or 'tokens')."""
    if not text:
        return 0
    if unit in ["characters", "chars"]:
        return len(text)
    return count_tokens(text)


def split_text_to_ceiling(text: str, max_limit: int, unit: str = "tokens") -> List[str]:
    """
    Enforces a strict hard ceiling on text length.
    If text exceeds max_limit (tokens or chars), safely partitions it
    along sentence or newline boundaries so NO sub-chunk exceeds max_limit.
    """
    if count_units(text, unit) <= max_limit:
        return [text.strip()]

    # Split into sentences or lines
    parts = re.split(r"(\n\n+|\n|(?<=[.?!।؟])\s+)", text)
    sub_chunks = []
    current_acc = ""

    for part in parts:
        candidate = current_acc + part if current_acc else part
        if count_units(candidate, unit) <= max_limit:
            current_acc = candidate
        else:
            if current_acc.strip():
                sub_chunks.append(current_acc.strip())
            # If a single atomic part exceeds max_limit, hard slice by characters
            if count_units(part, unit) > max_limit:
                char_limit = max_limit if unit in ["characters", "chars"] else max_limit * 4
                for i in range(0, len(part), char_limit):
                    sub = part[i : i + char_limit].strip()
                    if sub:
                        sub_chunks.append(sub)
                current_acc = ""
            else:
                current_acc = part

    if current_acc.strip():
        sub_chunks.append(current_acc.strip())

    return [s for s in sub_chunks if s]


@dataclass
class ChunkResult:
    """
    Universal Chunk Schema for Cloudflare & Vector Embeddings.
    - 'text': Pure text to send to the embedding model (within hard ceiling, no bracket pollution).
    - 'metadata': Auxiliary metadata stored in the vector database payload (NOT embedded).
    """
    chunk_id: int                           # Sequential chunk index
    start_char: int                         # Start character offset
    end_char: int                           # End character offset
    text: str = ""                          # EXACT clean text payload for embedding model
    chunk_type: str = "semantic"            # "pricing_table" | "location" | "faq" | "semantic"
    raw_text: str = ""                      # Verbatim slice (ungrounded)
    searchable_augmented_text: str = ""     # Searchable representation
    spoken_clean_text: str = ""             # (Legacy compatibility)
    context_grounding: str = ""             # Anthropic-style Contextual Retrieval Anchor [Context: ...]
    word_count: int = 0
    speech_seconds: float = 0.0
    warnings: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.text:
            self.text = self.searchable_augmented_text or self.raw_text
        if not self.raw_text:
            self.raw_text = self.text
        if not self.searchable_augmented_text:
            self.searchable_augmented_text = self.text
        if not self.context_grounding and self.metadata.get("context_grounding"):
            self.context_grounding = self.metadata["context_grounding"]
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export schema for Cloudflare Embeddings & Vector DBs:
        - 'text': Pure grounded text to embed.
        - 'raw_text': Verbatim ungrounded slice.
        - 'context_grounding': The contextual anchor breadcrumb.
        - 'metadata': Context tags stored in vector payload.
        """
        grounding = self.context_grounding or self.metadata.get("context_grounding", "")
        combined_metadata = {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "context_grounding": grounding,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            **self.metadata
        }
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "raw_text": self.raw_text,
            "context_grounding": grounding,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": combined_metadata,
        }



class BaseChunker(ABC):
    """Abstract base class for all specialized chunkers."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[ChunkResult]:
        """Partition text into chunks strictly within hard ceiling limit."""
        pass
