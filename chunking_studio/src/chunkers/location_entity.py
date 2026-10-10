import re
from typing import List, Dict, Any, Optional
from .base import BaseChunker, ChunkResult
from .context_ag import (
    build_searchable_augmented_text,
    clean_for_speech,
    calculate_speech_seconds,
)


class LocationEntityChunker(BaseChunker):
    """
    Indivisible Entity-Card Chunker for Branch & Location Directories.
    Bundles branch name, address, landmarks, operating hours, and phone numbers into a single indivisible card.
    Guarantee: If a document has 4 branch clinics, outputs exactly 4 discrete chunks.
    """

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[ChunkResult]:
        if not text or not text.strip():
            return []

        chunks: List[ChunkResult] = []
        meta = metadata or {}

        # Disallowed field keys that should never trigger an entity boundary split when appearing as child attributes
        child_keys = r"(?:Location|Address|Directions?|Landmarks?|Operating Hours|Hours|Timings?|Phone|Mobile|Contact|Services?|Facilities|Approval|Nearby locations?|Construction Details?|Main Content|Guideline Value|Price|Pricing|Rate|Cost)"

        # Patterns that indicate a top-level root entity (Branch, Clinic, Property, Project, Plot, Site)
        # Supports Markdown headers, keyword-numbered titles, multilingual numbered lists (e.g. 1. ஸ்ரீராம்கார்டன்), and divider lines (---)
        boundary_pattern = re.compile(
            r"(?=(?:^|\n)(?:"
            r"#{1,4}\s*(?:(?:Branch|Location|Clinic|Centre|Center|Property|Project|Site|Plot)\b[^\n]*|[^\n]+)|"
            r"(?:Branch|Property|Project|Site|Plot|Clinic)\s*#?\d+\s*[:\.\-]?[^\n]*|"
            r"\d+[\.\)]\s*(?!" + child_keys + r"\s*:)[^\n]+|"
            r"---+|===+"
            r"))",
            re.IGNORECASE | re.UNICODE
        )

        # Split text into candidate blocks while capturing positions
        split_positions = [m.start() if m.start() == 0 else m.start() + 1 for m in boundary_pattern.finditer(text)]

        # If no explicit branch markers found with the primary regex, fallback to markdown headers
        if len(split_positions) <= 1:
            header_pattern = re.compile(r"(?=(?:^|\n)#{1,3}\s+[^\n]+)", re.MULTILINE)
            split_positions = [m.start() if m.start() == 0 else m.start() + 1 for m in header_pattern.finditer(text)]

        # Fallback 2: Paragraph blocks separated by 2+ newlines only if no headers or numbered cards exist
        if len(split_positions) <= 1:
            matches = list(re.finditer(r"\n\s*\n+", text))
            split_positions = [0] + [m.end() for m in matches]

        # Clean and sort unique positions
        pos_list = sorted(list(set([p for p in split_positions if 0 <= p <= len(text)])))
        if not pos_list or pos_list[0] != 0:
            pos_list.insert(0, 0)
        if pos_list[-1] != len(text):
            pos_list.append(len(text))

        chunk_id = 1
        for i in range(len(pos_list) - 1):
            s_idx = pos_list[i]
            e_idx = pos_list[i + 1]
            block_text = text[s_idx:e_idx].strip()

            # Ignore standalone divider markers or empty blocks
            if not block_text or re.match(r"^(?:---+|===+)$", block_text):
                continue

            # Strip leading or trailing divider rules from the block
            block_text = re.sub(r"^(?:---+|===+)\s*\n+", "", block_text)
            block_text = re.sub(r"\n+\s*(?:---+|===+)$", "", block_text).strip()
            if not block_text:
                continue

            # Extract the branch / property anchor name from the first non-empty line
            first_line = block_text.splitlines()[0].strip()
            branch_match = re.sub(r"^#{1,4}\s*", "", first_line)

            searchable = build_searchable_augmented_text(
                block_text,
                metadata,
                section_context=f"Location Card: {branch_match}"
            )
            spoken = clean_for_speech(block_text, metadata)

            chunks.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    start_char=s_idx,
                    end_char=e_idx,
                    text=block_text,
                    raw_text=block_text,
                    searchable_augmented_text=block_text,
                    chunk_type="location",
                    warnings=[],
                    metadata={
                        **meta,
                        "branch_name": branch_match,
                    }
                )
            )
            chunk_id += 1

        return chunks
