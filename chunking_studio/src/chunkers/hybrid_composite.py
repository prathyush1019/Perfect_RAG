import re
from typing import List, Dict, Any, Optional
from .base import BaseChunker, ChunkResult
from .pricing_table import PricingTableChunker
from .location_entity import LocationEntityChunker
from .atomic_faq import AtomicFAQChunker
from .semantic_text import SemanticSentenceChunker


class HybridCompositeChunker(BaseChunker):
    """
    Hybrid Composite Chunker for All-in-One Mixed Documents.
    Scans and partitions the master document into typed zones:
    - Tables -> PricingTableChunker
    - Branch directories -> LocationEntityChunker
    - Q&A sections -> AtomicFAQChunker
    - Policy/Narrative -> SemanticSentenceChunker

    Preserves global character offsets and consecutively re-indexes chunks.
    """

    def __init__(
        self,
        unit: str = "characters",
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        target_words: int = 150,
        overlap_words: int = 25,
        rows_per_chunk: int = 1
    ):
        self.unit = unit
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.target_words = target_words
        self.overlap_words = overlap_words
        self.rows_per_chunk = rows_per_chunk

        self.table_chunker = PricingTableChunker(rows_per_chunk=rows_per_chunk)
        self.location_chunker = LocationEntityChunker()
        self.faq_chunker = AtomicFAQChunker()
        self.semantic_chunker = SemanticSentenceChunker(
            unit=unit,
            chunk_size=chunk_size,
            overlap=overlap,
            target_words=target_words,
            overlap_words=overlap_words
        )

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[ChunkResult]:
        if not text or not text.strip():
            return []

        meta = metadata or {}
        zones = self._partition_into_zones(text)
        all_chunks: List[ChunkResult] = []
        global_chunk_id = 1

        for zone in zones:
            zone_type = zone["type"]
            zone_text = zone["text"]
            zone_start = zone["start"]

            # Dispatch to specialized chunker
            if zone_type == "pricing_table":
                zone_chunks = self.table_chunker.chunk(zone_text, meta)
            elif zone_type == "location":
                zone_chunks = self.location_chunker.chunk(zone_text, meta)
            elif zone_type == "faq":
                zone_chunks = self.faq_chunker.chunk(zone_text, meta)
            else:
                zone_chunks = self.semantic_chunker.chunk(zone_text, meta)

            # Adjust offsets to master document and re-index sequentially
            for c in zone_chunks:
                if not (c.text or c.raw_text).strip():
                    continue
                c.chunk_id = global_chunk_id
                c.start_char = zone_start + c.start_char
                c.end_char = zone_start + c.end_char
                all_chunks.append(c)
                global_chunk_id += 1

        return all_chunks

    def _partition_into_zones(self, text: str) -> List[Dict[str, Any]]:
        """
        Partitions text into typed zones based on headers and content signatures.
        """
        # Split text on primary markdown headers # or ## while preserving character offsets.
        # If no ## exist, fallback to ###.
        has_h2 = bool(re.search(r"(?:^|\n)#{1,2}\s+[^\n]+", text))
        header_regex_str = r"(?=(?:^|\n)#{1,2}\s+[^\n]+)" if has_h2 else r"(?=(?:^|\n)#{1,3}\s+[^\n]+)"
        header_pattern = re.compile(header_regex_str, re.MULTILINE)
        split_positions = [m.start() if m.start() == 0 else m.start() + 1 for m in header_pattern.finditer(text)]

        if not split_positions or split_positions[0] != 0:
            split_positions = [0] + split_positions
        if split_positions[-1] != len(text):
            split_positions.append(len(text))

        # Deduplicate and sort
        split_positions = sorted(list(set(split_positions)))

        raw_sections = []
        for i in range(len(split_positions) - 1):
            s = split_positions[i]
            e = split_positions[i + 1]
            sec_text = text[s:e]
            if sec_text.strip():
                raw_sections.append({
                    "start": s,
                    "end": e,
                    "text": sec_text
                })

        # If no sections could be formed from headers, treat entire text as one section
        if not raw_sections:
            raw_sections.append({"start": 0, "end": len(text), "text": text})

        # Classify each section
        classified_zones = []
        for sec in raw_sections:
            stext = sec["text"]
            ztype = self._classify_section(stext)
            classified_zones.append({
                "type": ztype,
                "start": sec["start"],
                "end": sec["end"],
                "text": stext
            })

        return classified_zones

    def _classify_section(self, text: str) -> str:
        """Determines the dominant content type of a section."""
        lower = text.lower()

        # Check for Pricing Table
        table_lines = [l for l in text.splitlines() if "|" in l and l.strip().startswith("|")]
        if len(table_lines) >= 2 and any(re.match(r"^\|(\s*:?-+:?\s*\|)+$", l.strip()) for l in table_lines):
            return "pricing_table"

        # Check for FAQ
        # Looks for multiple Q/A indicators
        faq_matches = re.findall(
            r"(?:^|\n)(?:#{1,4}\s*(?:Question|Q\b)|(?:Question|Q)\s*(?:#?\d+)?\s*[:\.\?\-]|\d+\.\s*(?:Q|Question)\s*[:\.\?\-])",
            text,
            re.IGNORECASE
        )
        if len(faq_matches) >= 2 or ("faq" in lower and len(faq_matches) >= 1) or ("frequently asked" in lower):
            return "faq"

        # Check for Location / Branch / Property Directory
        child_keys = r"(?:Location|Address|Directions?|Landmarks?|Operating Hours|Hours|Timings?|Phone|Mobile|Contact|Services?|Facilities|Approval|Nearby locations?|Construction Details?|Main Content|Guideline Value|Price|Pricing|Rate|Cost)"
        location_matches = re.findall(
            r"(?:^|\n)(?:"
            r"#{1,4}\s*(?:Branch|Location|Clinic|Centre|Center|Property|Project|Site|Plot)\b[^\n]*|"
            r"(?:Branch|Property|Project|Site|Plot|Clinic)\s*#?\d+\s*[:\.\-]?[^\n]*|"
            r"\d+[\.\)]\s*(?!" + child_keys + r"\s*:)[^\n]+"
            r")",
            text,
            re.IGNORECASE | re.UNICODE
        )
        has_dir_kw = bool(re.search(r"(?:location|address|facilities|phone|hours|guideline value|bhk)", lower))
        if (len(location_matches) >= 2 and has_dir_kw) or ("branch" in lower and "phone" in lower and ("address" in lower or "hours" in lower)):
            return "location"

        # Default to semantic text
        return "semantic"
