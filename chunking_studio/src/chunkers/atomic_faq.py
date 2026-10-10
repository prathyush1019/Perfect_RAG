import re
from typing import List, Dict, Any, Optional
from .base import BaseChunker, ChunkResult
from .context_ag import (
    build_searchable_augmented_text,
    clean_for_speech,
    calculate_speech_seconds,
)


class AtomicFAQChunker(BaseChunker):
    """
    Atomic FAQ Chunker.
    Ensures that 1 Chunk = Exactly 1 Complete Q&A Pair.
    Guarantees an answer is never severed from its question.
    """

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[ChunkResult]:
        if not text or not text.strip():
            return []

        chunks: List[ChunkResult] = []
        meta = metadata or {}

        # Regex matching the start of a Question
        # e.g.: "### Question:", "### Q:", "Question 1:", "Q1:", "Q:", "**Q:**", "**Question 1:**", "1. Q:"
        q_pattern = re.compile(
            r"(?=(?:^|\n)(?:#{1,4}\s*(?:Question|Q\b|\d+\.\s*Question)|(?:\*{1,2})?(?:Question|Q)\s*(?:#?\d+)?(?:\*{1,2})?\s*[:\.\?\-]|(?:\d+\.\s*(?:Q|Question)\s*[:\.\?\-])))",
            re.IGNORECASE
        )

        matches = list(q_pattern.finditer(text))

        # Fallback if no explicit Q: matches found (e.g. numbered questions like "1. How do I book an appointment?")
        if len(matches) <= 1:
            numbered_q_pattern = re.compile(
                r"(?=(?:^|\n)(?:\d+\.\s+[A-Z][^\n\?]+\?))",
                re.MULTILINE
            )
            numbered_matches = list(numbered_q_pattern.finditer(text))
            if len(numbered_matches) > 1:
                matches = numbered_matches

        # Fallback 2: Markdown headers with question marks "### What is...?"
        if len(matches) <= 1:
            header_q_pattern = re.compile(
                r"(?=(?:^|\n)#{1,4}\s+[^\n]+\?)",
                re.MULTILINE
            )
            header_matches = list(header_q_pattern.finditer(text))
            if len(header_matches) > 1:
                matches = header_matches

        # If still no matches or only 1, check paragraph blocks with Question / Answer
        if not matches:
            # Whole text as one FAQ
            return [self._create_faq_chunk(1, 0, len(text), text.strip(), meta)]

        # Extract start offsets
        cut_points = [m.start() if m.start() == 0 else m.start() + 1 for m in matches]
        # Clean duplicates and sort
        cut_points = sorted(list(set(cut_points)))

        # Only insert 0 if the preamble before the first question actually contains a Q&A
        if cut_points and cut_points[0] != 0:
            preamble = text[0:cut_points[0]].strip()
            if "?" in preamble or "answer:" in preamble.lower():
                cut_points.insert(0, 0)
        elif not cut_points:
            cut_points = [0]

        cut_points.append(len(text))

        chunk_id = 1
        for i in range(len(cut_points) - 1):
            s_idx = cut_points[i]
            e_idx = cut_points[i + 1]
            pair_text = text[s_idx:e_idx].strip()

            if not pair_text:
                continue

            chunk = self._create_faq_chunk(chunk_id, s_idx, e_idx, pair_text, meta)
            chunks.append(chunk)
            chunk_id += 1

        return chunks

    def _create_faq_chunk(
        self,
        chunk_id: int,
        start_char: int,
        end_char: int,
        raw_text: str,
        metadata: Dict[str, Any]
    ) -> ChunkResult:
        # Extract question line for context
        first_line = raw_text.splitlines()[0].strip()
        q_cleaned = re.sub(r"^#{1,4}\s*", "", first_line)

        searchable = build_searchable_augmented_text(
            raw_text,
            metadata,
            section_context=f"FAQ Pair: {q_cleaned}"
        )
        spoken = clean_for_speech(raw_text, metadata)

        return ChunkResult(
            chunk_id=chunk_id,
            start_char=start_char,
            end_char=end_char,
            text=raw_text,
            raw_text=raw_text,
            searchable_augmented_text=raw_text,
            chunk_type="faq",
            warnings=[],
            metadata={
                **metadata,
                "question_summary": q_cleaned,
            }
        )
