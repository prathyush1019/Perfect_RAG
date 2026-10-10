import re
from typing import List, Dict, Any, Optional
from .base import BaseChunker, ChunkResult
from .context_ag import (
    build_searchable_augmented_text,
    clean_for_speech,
    calculate_speech_seconds,
)


class PricingTableChunker(BaseChunker):
    """
    Row-Preserved Markdown Table Chunker.
    Ensures that pricing rows are never severed from table column headers or currency anchors.
    """

    def __init__(self, rows_per_chunk: int = 1):
        self.rows_per_chunk = max(1, rows_per_chunk)

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[ChunkResult]:
        if not text or not text.strip():
            return []

        if not text.endswith("\n"):
            text = text + "\n"

        chunks: List[ChunkResult] = []
        meta = metadata or {}

        # Regular expression to find Markdown tables
        # Matches lines with pipes and a delimiter line like |---|---|
        table_pattern = re.compile(
            r"((?:^[ \t]*\|[^\n]+\|[ \t]*\n)+)",
            re.MULTILINE
        )

        matches = list(table_pattern.finditer(text))

        # If no explicit tables matched via regex, try line-by-line fallback
        if not matches:
            lines = text.splitlines(keepends=True)
            tbl_lines = [l for l in lines if "|" in l]
            if len(tbl_lines) >= 2:
                # Treat whole text as a potential table
                start_c = 0
                end_c = len(text)
                return self._process_single_table(text, start_c, end_c, meta, 1)
            return []

        chunk_id = 1
        for match in matches:
            table_text = match.group(1)
            table_start = match.start(1)
            table_end = match.end(1)

            table_chunks = self._process_single_table(
                table_text, table_start, table_end, meta, chunk_id
            )
            chunks.extend(table_chunks)
            chunk_id += len(table_chunks)

        return chunks

    def _process_single_table(
        self,
        table_text: str,
        global_start: int,
        global_end: int,
        metadata: Dict[str, Any],
        start_chunk_id: int
    ) -> List[ChunkResult]:
        chunks: List[ChunkResult] = []
        lines_with_pos = []
        
        # Track line positions within table_text
        curr_offset = 0
        for line in table_text.splitlines(keepends=True):
            line_len = len(line)
            lines_with_pos.append((line, curr_offset, curr_offset + line_len))
            curr_offset += line_len

        # Separate headers and data rows
        header_lines = []
        data_rows = []
        is_past_separator = False

        for line_str, s_pos, e_pos in lines_with_pos:
            stripped = line_str.strip()
            if not stripped.startswith("|") or not stripped.endswith("|"):
                continue

            # Check if this is a separator line like |---|---|
            if re.match(r"^\|(\s*:?-+:?\s*\|)+$", stripped):
                is_past_separator = True
                continue

            if not is_past_separator and len(header_lines) == 0:
                header_lines.append((line_str, s_pos, e_pos))
            else:
                data_rows.append((line_str, s_pos, e_pos))

        if not header_lines and data_rows:
            # Fallback: treat first row as header
            header_lines.append(data_rows.pop(0))

        header_text = "".join([h[0] for h in header_lines]).strip()

        # Group data rows according to rows_per_chunk
        chunk_idx = start_chunk_id
        for i in range(0, len(data_rows), self.rows_per_chunk):
            batch = data_rows[i : i + self.rows_per_chunk]
            batch_text = "".join([b[0] for b in batch]).strip()

            # Global offsets
            chunk_start = global_start + batch[0][1]
            chunk_end = global_start + batch[-1][2]

            # Verbatim raw text slice
            raw_slice = text_slice = table_text[batch[0][1] : batch[-1][2]].strip()

            # The atomic chunk injects the column headers + the row(s)
            atomic_table_chunk = f"{header_text}\n{batch_text}"

            chunks.append(
                ChunkResult(
                    chunk_id=chunk_idx,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    text=atomic_table_chunk,
                    raw_text=raw_slice,
                    searchable_augmented_text=atomic_table_chunk,
                    chunk_type="pricing_table",
                    warnings=[],
                    metadata={
                        **metadata,
                        "header_injected": True,
                        "row_count": len(batch),
                    }
                )
            )
            chunk_idx += 1

        return chunks
