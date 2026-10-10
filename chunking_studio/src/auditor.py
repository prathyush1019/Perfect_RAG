import re
from typing import List, Dict, Any
from .chunkers.base import ChunkResult


class VoiceChunkAuditor:
    """
    Automated Voice Readiness Quality Linting Engine.
    Detects acoustic telephone agent hazards:
    1. Acoustic Read-Aloud Time (> 15.0s causes caller drop-off)
    2. Orphaned / Floating Price rows without column header anchors
    3. Severed Location Cards missing phone or operating hours
    4. Splintered FAQ Answers prematurely cut off
    5. Unresolved Pronoun Openings without subject anchor
    """

    MAX_SPEECH_SECONDS = 15.0

    HEADER_KEYWORDS = [
        "plan", "tier", "price", "treatment", "fee", "rate",
        "warranty", "service", "cost", "concurrency", "procedure",
        "description", "item", "package", "duration", "base price"
    ]

    PRONOUN_STARTS = [
        r"^it\s+is\b", r"^they\s+are\b", r"^this\s+rule\b",
        r"^these\s+are\b", r"^it\s+will\b", r"^he\s+is\b",
        r"^she\s+is\b", r"^it\s+has\b", r"^we\s+do\b"
    ]

    def audit_chunk(self, chunk: ChunkResult) -> List[Dict[str, str]]:
        """Runs all voice lints on an individual chunk."""
        warnings: List[Dict[str, str]] = []

        # 1. Acoustic Read-Aloud Time
        if chunk.speech_seconds > self.MAX_SPEECH_SECONDS:
            warnings.append({
                "rule": "speech_duration",
                "severity": "critical",
                "badge": f"🚨 Too Long: {chunk.speech_seconds:.1f}s",
                "message": f"TTS duration exceeds 15s threshold ({chunk.speech_seconds:.1f}s). Callers may disconnect."
            })

        # 2. Orphaned / Floating Price Check
        # Triggered if raw text or searchable text contains currency indicators but lacks header anchors
        text_for_price = chunk.raw_text.lower()
        has_currency = any(c in text_for_price for c in ["₹", "$", "€", "£", "/mo", "inr", "usd", "eur"])
        if has_currency:
            searchable_lower = chunk.searchable_augmented_text.lower()
            has_header = any(hk in searchable_lower for hk in self.HEADER_KEYWORDS)
            if not has_header:
                warnings.append({
                    "rule": "orphaned_price",
                    "severity": "critical",
                    "badge": "🚨 Critical: Severed Price Row",
                    "message": "Chunk contains pricing or rates without table header anchors or column definitions."
                })

        # 3. Severed Location Card Check
        if chunk.chunk_type == "location" or "branch" in chunk.raw_text.lower() or "clinic" in chunk.raw_text.lower():
            lower_raw = chunk.raw_text.lower()
            has_phone = bool(re.search(r"(?:phone|ph|tel|call|contact|mobile|\+?\d{2,4}[-\s]?\d{6,10})", lower_raw))
            has_hours = bool(re.search(r"(?:hours|timing|time|open|mon|am|pm|\b\d{1,2}:\d{2}\b)", lower_raw))
            if not (has_phone and has_hours):
                missing = []
                if not has_phone:
                    missing.append("phone number")
                if not has_hours:
                    missing.append("operating hours")
                warnings.append({
                    "rule": "severed_location",
                    "severity": "warning",
                    "badge": "⚠️ Warning: Incomplete Location Card",
                    "message": f"Location card is missing {' and '.join(missing)}."
                })

        # 4. Splintered FAQ Answer Check
        if chunk.chunk_type == "faq":
            stripped_spoken = chunk.spoken_clean_text.strip()
            # Premature termination with colon or trailing phrases without bullets
            if stripped_spoken.endswith(":") or re.search(r"(?:the following|such as|includes):\s*$", stripped_spoken, re.IGNORECASE):
                warnings.append({
                    "rule": "splintered_faq",
                    "severity": "warning",
                    "badge": "⚠️ Warning: Splintered FAQ Answer",
                    "message": "FAQ answer terminates with a dangling colon or introductory phrase."
                })

        # 5. Unresolved Pronoun Opening Check
        first_segment = chunk.spoken_clean_text[:35].strip().lower()
        for p_pattern in self.PRONOUN_STARTS:
            if re.match(p_pattern, first_segment):
                warnings.append({
                    "rule": "vague_pronoun",
                    "severity": "warning",
                    "badge": "⚠️ Warning: Vague Pronoun Start",
                    "message": f"Chunk starts with ambiguous pronoun ('{first_segment.split()[0]} {first_segment.split()[1]}') without explicit subject context."
                })
                break

        return warnings

    def audit_chunks(self, chunks: List[ChunkResult]) -> Dict[str, Any]:
        """
        Runs audit on all chunks, assigns warnings, and computes the overall Voice Quality Score.
        Returns audit summary dictionary.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "passing_chunks": 0,
                "failing_chunks": 0,
                "quality_score": 100.0,
                "total_warnings": 0,
                "warnings_by_type": {}
            }

        passing_count = 0
        total_warnings = 0
        warning_counts: Dict[str, int] = {}

        for c in chunks:
            c.warnings = self.audit_chunk(c)
            if not c.warnings:
                passing_count += 1
            else:
                total_warnings += len(c.warnings)
                for w in c.warnings:
                    rule = w["rule"]
                    warning_counts[rule] = warning_counts.get(rule, 0) + 1

        quality_score = (passing_count / len(chunks)) * 100.0

        return {
            "total_chunks": len(chunks),
            "passing_chunks": passing_count,
            "failing_chunks": len(chunks) - passing_count,
            "quality_score": round(quality_score, 1),
            "total_warnings": total_warnings,
            "warnings_by_type": warning_counts
        }
