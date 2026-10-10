# 🎙️ SPECIFICATION PROMPT: Voice AI Context-AG Chunking Engine & Visual Studio

> **Instructions for the AI Agent receiving this document:**
> You are tasked with building a standalone, production-grade **Voice AI Context-Augmented Generation (Context-AG) Chunking Engine & Visual Studio**.
> Your application ingests client documents (`.pdf`, `.docx`, `.txt`, `.md`, unstructured notes, and multilingual text) and partitions them into **voice-first, zero-hallucination discrete chunks** optimized for Voice AI telephone agents and real-time Text-to-Speech (TTS) synthesis.

---

## 🚨 1. Critical System Constraints & Negative Rules

1. **NO Vector Database Required**: The application processes documents 100% in-memory. Do not require Qdrant, Pinecone, Milvus, or external vector stores.
2. **NO Conversational Chatbot**: The application does not answer user questions; its output is the **discrete, verified vector chunks**.
3. **NEVER Sever Context**: 
   - A pricing table row must NEVER be split from its column headers or currency.
   - A business branch address must NEVER be split from its operating hours or phone number.
   - An FAQ question must NEVER be severed from its answer.
4. **MANDATORY Dual-Text Contract**: Every generated chunk must output two distinct text representations:
   - `searchable_augmented_text` (For vector search indexing with bracketed metadata context).
   - `spoken_clean_text` (For TTS speech synthesis, completely free of brackets, markdown pipes, or symbols).
5. **LIVE CHUNK BOUNDARY VISUALIZATION**: The application must visibly draw and map chunk partitions directly across the source text using alternating color bands and explicit cutline markers (`✂️ Boundary #1 [Chars 0-350]`).

---

## 🏛️ 2. The Two-Stage Hybrid Architecture

To handle messy, unstructured, and multilingual documents (e.g., scanned PDFs, Hinglish, mixed brochures), the system implements the **Two-Stage Hybrid Architecture**:

```
[Raw Messy / Multilingual / Unstructured Document]
                      │
                      ▼
[STAGE 1: LLM Normalizer & Structural Transcriber (Optional Pre-Processor)]
   • Reconstructs scrambled tables into standard Markdown (| Col | Col |).
   • Unifies scattered branch info into "### Branch: [Name]" entity cards.
   • Unifies scattered Q&As into "### Question: ... Answer: ...".
   • ZERO-REWRITE CONSTRAINT: Verbatim extraction with zero price/number rounding.
                      │
                      ▼
[Clean Intermediate Document (.md)]
                      │
                      ▼
[STAGE 2: Deterministic Context-AG Chunking Engine (100% Deterministic Python)]
   • Analyzes structure & dispatches to specialized voice chunkers.
   • Preserves exact character offsets ([Chars start-end]) for boundary drawing.
   • Generates Searchable Augmented vs Spoken Clean Dual-Text.
   • Runs Automated Voice Readiness Linter (TTS read-aloud time < 15s, orphaned rates).
                      │
                      ▼
[Final Clean Chunks Output (JSON / CSV / Interactive UI Cards)]
```

---

## 📂 3. Target Project Directory Structure

```
chunking_studio/
├── app.py                          # Streamlit interactive workbench & visualizer
├── requirements.txt                # streamlit, pypdf, python-docx, markdown, pandas
├── run.bat                         # 1-click Windows launcher
├── src/
│   ├── __init__.py
│   ├── document_parsers.py         # PDF, DOCX, Markdown, Text extractors
│   ├── chunkers/
│   │   ├── __init__.py             # Chunker registry & content auto-detection
│   │   ├── base.py                 # Abstract Base Chunker & ChunkResult dataclass
│   │   ├── context_ag.py           # Dual-Text Engine (Searchable vs Spoken)
│   │   ├── pricing_table.py        # Row-Preserved Markdown Table Chunker
│   │   ├── location_entity.py      # Indivisible Entity-Card Chunker
│   │   ├── atomic_faq.py           # 1:1 Q&A Pair Chunker
│   │   ├── semantic_text.py        # Sentence-Boundary Chunker with Overlap
│   │   └── hybrid_composite.py     # Multi-Section Chunker for Mixed All-in-One Docs
│   ├── auditor.py                  # Voice Readiness Quality Linting Engine
│   └── ui/
│       ├── styles.py               # Dark glassmorphic CSS & alternating band colors
│       └── views.py                # Drawing canvas, KPI widgets, side-by-side cards
├── tests/
│   └── test_studio.py              # Automated verification test suite
└── sample_data/                    # Ready-to-load sample client documents
    ├── sample_pricing.md           # Tiered pricing table
    ├── sample_locations.txt        # Multi-branch clinic directory
    ├── sample_faq.txt              # 10 discrete Q&A pairs
    ├── sample_policy.txt           # General cancellation prose
    └── sample_comprehensive_guide.md # Mixed document (all types in one)
```

---

## 🧩 4. Core Data Contracts & Specifications

### A. The Universal Chunk Schema (`src/chunkers/base.py`)
Every chunker must produce instances of `ChunkResult`:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ChunkResult:
    chunk_id: int                           # Sequential chunk index (1, 2, 3...)
    start_char: int                         # Global character start offset in source text
    end_char: int                           # Global character end offset in source text
    raw_text: str                           # Verbatim slice from the original document
    searchable_augmented_text: str          # Vector text prepended with [Context Brackets]
    spoken_clean_text: str                  # Natural text-to-speech text (no pipes, no hashes)
    chunk_type: str                         # "pricing_table" | "location" | "faq" | "semantic"
    word_count: int                         # Total words in chunk
    speech_seconds: float                   # Estimated TTS duration at 150 words/min
    warnings: List[Dict[str, str]] = field(default_factory=list)  # Voice lint flags
    metadata: Dict[str, Any] = field(default_factory=dict)        # Document context tags
```

### B. Input Document Metadata Contract
```python
doc_metadata = {
    "doc_title": "Apex Dental Guide 2026",
    "entity_name": "Apex Dental Care",
    "category": "Pricing & Operating Policies",
    "location": "Chennai & Bangalore",
    "currency": "INR",
    "language": "en-US" # or "hi-IN", "es-ES", "hinglish"
}
```

---

## ⚙️ 5. Algorithmic Modules & Chunking Strategies

### Module 1: The Context-AG Dual-Text Engine (`src/chunkers/context_ag.py`)
A voice telephone agent must **never** read raw bracketed tags or markdown syntax aloud.

1. **`searchable_augmented_text`** (Indexed in Vector Store):
   ```text
   [Document: Apex Dental Guide 2026 | Entity: Apex Dental Care | Category: Pricing | Currency: INR]
   | Treatment | Base Price | Warranty |
   | Zirconia Crown | ₹6,000 | 15 Years Replacement Warranty |
   ```
2. **`spoken_clean_text`** (Read Aloud by TTS):
   ```text
   Treatment: Zirconia Crown, Base Price: 6,000 rupees, Warranty: 15 Years Replacement Warranty.
   ```
   - Strips all `[...]` brackets and markdown hashes (`#`, `##`).
   - Converts markdown table pipes into natural conversational phrasing.
   - Strips raw URLs and cleans excessive bullet points.

---

### Module 2: Specialized Voice Chunkers

#### 1. Pricing Table Chunker (`src/chunkers/pricing_table.py`)
- Detects Markdown / ASCII tables (`| col | col |` with `|---|---|`).
- **Atomic Rule**: Never split mid-row.
- **Header Injection Rule**: Every individual row chunk must retain the full column header line and currency anchor.
- Configurable slider: `rows_per_chunk` (default = 1 row per chunk).

#### 2. Location Entity Chunker (`src/chunkers/location_entity.py`)
- Identifies multi-branch sections (`Branch 1:`, `Location 2:`, `Clinic:`, or address blocks).
- **Indivisible Card Rule**: Address, landmarks, operating hours, and phone numbers are bundled into a single entity card.
- **Integrity Rule**: If a document lists 4 branch clinics, the output must be **exactly 4 discrete chunks**, regardless of character count.

#### 3. Atomic FAQ Chunker (`src/chunkers/atomic_faq.py`)
- Detects Q/A delimiters (`Q:`, `Question:`, `### Question`, `**Q:**`).
- **1:1 Rule**: 1 Chunk = Exactly 1 complete Q&A pair. Never chop an answer halfway through.

#### 4. Semantic Sentence Chunker (`src/chunkers/semantic_text.py`)
- For policies, guidelines, and general narrative prose.
- Splits cleanly along sentence terminations (`. `, `? `, `! `, `\n\n`) while respecting multilingual stops (`।`, `؟`).
- Supports configurable sliders: `target_words` (e.g. 150 words) and `overlap_words` (e.g. 25 words).
- Snaps cleanly to sentence boundaries (never cuts mid-word).

#### 5. Hybrid Composite Chunker (`src/chunkers/hybrid_composite.py`)
- Solves the **all-in-one mixed document problem** (e.g., an operational handbook containing prose, pricing tables, branch directories, and FAQs in a single file).
- Scans and partitions the master document into typed zones (`table`, `location`, `faq`, `prose`).
- Dispatches each zone to its dedicated voice chunker.
- Preserves master character offsets and re-indexes chunks consecutively (Chunk #1, #2, #3...).

---

## 🛡️ 6. Automated Voice Readiness Quality Auditor (`src/auditor.py`)

Every generated chunk must run through automated voice linting rules that catch common telephone agent hazards:

| Lint Rule | Detection Logic | Warning Badge |
| :--- | :--- | :--- |
| **Acoustic Read-Aloud Time** | $\text{seconds} = \frac{\text{words}}{150} \times 60$. Flag if $> 15.0\text{s}$. (Caller will disconnect if bot speaks too long). | `🚨 Too Long: 22s` vs `⏱️ Speech Time: ~5.4s (Optimal)` |
| **Orphaned / Floating Price** | Contains `$`, `₹`, `€`, `£`, `/mo` or rates, but lacks table headers (`plan`, `tier`, `price`, `concurrency`). | `🚨 Critical: Severed Price Row` |
| **Severed Location Card** | Chunk has address/branch name, but is missing operating hours or phone number. | `⚠️ Warning: Incomplete Location Card` |
| **Splintered FAQ Answer** | Terminated prematurely with a colon `:` or phrases like *"the following:"* without list bullets. | `⚠️ Warning: Splintered FAQ Answer` |
| **Unresolved Pronoun Opening** | First 25 chars start with *"It is"*, *"They are"*, *"This rule"* without clear subject anchor. | `⚠️ Warning: Vague Pronoun Start` |

- **Quality Score Formula**:
  $$\text{Quality Score (\%)} = \frac{\text{Chunks Passing All Lints}}{\text{Total Chunks}} \times 100$$

---

## 🖥️ 7. User Interface (UI) Requirements (`app.py`)

1. **Aesthetics**: Modern dark glassmorphic UI (`#0b0f19` background, `#111827` cards, glowing badges, Inter & Outfit typography).
2. **Sidebar Controls**:
   - Live File Uploader (`.pdf`, `.docx`, `.txt`, `.md`).
   - Quick Dataset Buttons (`📊 Pricing Sheet`, `🏢 Branch Cards`, `❓ Atomic FAQ`, `📜 Policy Prose`, `🌟 Master Guide (Mixed)`).
   - Context-AG Metadata Form (`Title`, `Entity`, `Category`, `Location`, `Currency`).
   - Strategy Selector (`Auto-Detect Strategy (Recommended)`, `Hybrid Multi-Section`, `Pricing Table`, `Atomic FAQ`, `Location Entity`, `Semantic Sentence`).
   - Sliders (`Target Chunk Size`, `Chunk Overlap`, `Table Rows per Chunk`).
3. **Top KPI Dashboard**:
   - `Document Volume` (Total words).
   - `Chunks Produced` (Count).
   - `Average Chunk Size` (Words).
   - `Voice Readiness` (% quality score).
4. **Live Chunk Drawing Canvas ("Draw Chunks" Visual Map)**:
   - Renders the complete document with alternating highlight bands:
     - `Chunk #1`: Amber tint (`rgba(243, 128, 32, 0.12)`)
     - `Chunk #2`: Cyan tint (`rgba(0, 210, 180, 0.12)`)
     - `Chunk #3`: Purple tint (`rgba(168, 85, 247, 0.12)`)
     - `Chunk #4`: Emerald tint (`rgba(16, 185, 129, 0.12)`)
   - Inserts explicit dashed cutline markers between chunks:  
     `✂️ Boundary #1 | End of Chunk 1 ➔ Start of Chunk 2 [Character Offset: 342]`
5. **Final Chunks Inspector**:
   - Card feed displaying each parsed chunk.
   - Badges: `Chunk #`, `Type`, `Word Count`, `Speech Time`, `Linter Status`.
   - Side-by-Side Dual-Text Comparison:
     - **Left Box (Yellow Border)**: `searchable_augmented_text` (Vector search target).
     - **Right Box (Green Border)**: `spoken_clean_text` (TTS speech target).
6. **Export Buttons**:
   - `Download Clean Chunks (JSON)`.
   - `Download Clean Chunks (CSV)`.

---

## 🧪 8. Acceptance Criteria & Test Suite (`tests/test_studio.py`)

Before declaring the implementation complete, verify that:

1. **Table Integrity**: Ingesting `sample_pricing.md` produces row chunks where every single chunk retains the column headers and currency anchor.
2. **Location Integrity**: Ingesting a document with 4 clinic branches produces **exactly 4 discrete location card chunks**.
3. **FAQ Integrity**: Ingesting a 10-question FAQ document produces **exactly 10 discrete Q&A chunks**.
4. **Hybrid Detection**: Ingesting a mixed document (policy + table + branches + FAQ) triggers `hybrid_composite` auto-detection and yields chunks across all 4 types.
5. **Auditor Validation**: A floating table row without headers is flagged with `🚨 Critical: Severed Price Row`.
6. **Acoustic Timing**: 150 words calculates to exactly `60.0s` speech time.
7. **Negative Constraints**: Zero calls to external vector databases, zero Docker dependencies, and zero LLM conversational chat endpoints.
8. **Launch Script**: Runs via `run.bat` or `streamlit run app.py --server.port 8505 --theme.base dark`.

---

## 🚀 9. Starter Code for Testing (`tests/test_studio.py`)

```python
import os, sys
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.chunkers.pricing_table import PricingTableChunker
from src.chunkers.location_entity import LocationEntityChunker
from src.chunkers.atomic_faq import AtomicFAQChunker
from src.chunkers.hybrid_composite import HybridCompositeChunker
from src.auditor import VoiceChunkAuditor

def run_tests():
    print("Testing Voice AI CAG Chunking Engine...")
    # Verify Table Chunker
    # Verify Location Chunker
    # Verify FAQ Chunker
    # Verify Hybrid Chunker
    # Verify Auditor
    print("All tests passed with 100% compliance!")

if __name__ == "__main__":
    run_tests()
```
