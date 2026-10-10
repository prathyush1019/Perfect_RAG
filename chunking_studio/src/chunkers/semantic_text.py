import re
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseChunker, ChunkResult
from .context_ag import (
    build_searchable_augmented_text,
    clean_for_speech,
    calculate_speech_seconds,
)


class SemanticSentenceChunker(BaseChunker):
    """
    Sentence-Boundary Chunker with Overlap for Narrative & Policy Prose.
    Supports chunking by 'characters', 'tokens', or 'words'.
    Splits along natural sentence terminations (. ? ! । ؟).
    Never cuts mid-word or mid-sentence.
    """

    def __init__(
        self,
        unit: str = "characters",
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        target_words: int = 150,
        overlap_words: int = 25
    ):
        self.unit = unit.lower() if unit else "characters"

        # Determine target and overlap according to unit
        if chunk_size is not None:
            self.target_size = max(10, chunk_size)
            self.overlap_size = max(0, min(overlap if overlap is not None else 0, self.target_size // 2))
        elif self.unit == "characters":
            self.target_size = 600
            self.overlap_size = 100
        elif self.unit == "tokens":
            self.target_size = 150
            self.overlap_size = 25
        else:  # words
            self.target_size = max(20, target_words)
            self.overlap_size = max(0, min(overlap_words, self.target_size // 2))

    def _get_sentence_unit_metric(self, s_text: str, w_count: int) -> int:
        """Computes metric for sentence according to configured unit."""
        if self.unit == "characters" or self.unit == "chars":
            return len(s_text)
        elif self.unit == "tokens":
            return max(1, round(len(s_text) / 4))
        else:  # words
            return w_count

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[ChunkResult]:
        if not text or not text.strip():
            return []

        chunks: List[ChunkResult] = []
        meta = metadata or {}

        # 1. Split text into sentences while tracking exact character spans
        sentence_regex = re.compile(
            r"([^.?!।؟\n]+[.?!।؟]+[\s\n]*|.+?(?:\n\s*\n+|$))",
            re.DOTALL
        )

        sentences: List[Tuple[str, int, int, int, int]] = []  # (text, start, end, word_count, metric)
        for match in sentence_regex.finditer(text):
            s_text = match.group(0)
            if not s_text.strip():
                continue
            s_start = match.start()
            s_end = match.end()
            w_count = len(s_text.split())
            metric = self._get_sentence_unit_metric(s_text, w_count)
            sentences.append((s_text, s_start, s_end, w_count, metric))

        if not sentences:
            w_count = len(text.split())
            return [
                ChunkResult(
                    chunk_id=1,
                    start_char=0,
                    end_char=len(text),
                    raw_text=text.strip(),
                    searchable_augmented_text=build_searchable_augmented_text(text, meta),
                    spoken_clean_text=clean_for_speech(text, meta),
                    chunk_type="semantic",
                    word_count=w_count,
                    speech_seconds=calculate_speech_seconds(text),
                    warnings=[],
                    metadata=meta,
                )
            ]

        # 2. Window sentences according to target_size and overlap_size
        chunk_id = 1
        curr_start_idx = 0

        while curr_start_idx < len(sentences):
            accum_metric = 0
            curr_end_idx = curr_start_idx

            while curr_end_idx < len(sentences):
                s_text, _, _, w_count, metric = sentences[curr_end_idx]
                if accum_metric + metric > self.target_size and accum_metric > 0:
                    break
                accum_metric += metric
                curr_end_idx += 1

            # Build chunk from sentences[curr_start_idx : curr_end_idx]
            selected_sentences = sentences[curr_start_idx:curr_end_idx]
            chunk_start_char = selected_sentences[0][1]
            chunk_end_char = selected_sentences[-1][2]
            raw_chunk_text = text[chunk_start_char:chunk_end_char].strip()

            searchable = build_searchable_augmented_text(
                raw_chunk_text,
                meta,
                section_context="Policy & General Prose"
            )
            spoken = clean_for_speech(raw_chunk_text, meta)
            chunks.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    start_char=chunk_start_char,
                    end_char=chunk_end_char,
                    text=raw_chunk_text,
                    raw_text=raw_chunk_text,
                    searchable_augmented_text=raw_chunk_text,
                    chunk_type="semantic",
                    warnings=[],
                    metadata={
                        **meta,
                        "unit": self.unit,
                        "sentence_count": len(selected_sentences),
                    }
                )
            )
            chunk_id += 1

            if curr_end_idx >= len(sentences):
                break

            # Calculate overlap: rewind curr_end_idx by overlap_size
            if self.overlap_size > 0:
                overlap_accum = 0
                rewind_idx = curr_end_idx - 1
                while rewind_idx > curr_start_idx:
                    overlap_accum += sentences[rewind_idx][4]
                    if overlap_accum >= self.overlap_size:
                        break
                    rewind_idx -= 1
                curr_start_idx = max(curr_start_idx + 1, rewind_idx)
            else:
                curr_start_idx = curr_end_idx

        return chunks
