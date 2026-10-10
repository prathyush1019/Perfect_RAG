import os
import sys

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.chunkers.pricing_table import PricingTableChunker
from src.chunkers.location_entity import LocationEntityChunker
from src.chunkers.atomic_faq import AtomicFAQChunker
from src.chunkers.semantic_text import SemanticSentenceChunker
from src.chunkers.hybrid_composite import HybridCompositeChunker
from src.chunkers import detect_document_strategy
from src.chunkers.context_ag import calculate_speech_seconds
from src.auditor import VoiceChunkAuditor
from src.chunkers.base import ChunkResult


def run_tests():
    print("=" * 60)
    print("🎙️ Starting Voice AI Context-AG Chunking Engine Test Suite")
    print("=" * 60)

    sample_dir = os.path.join(CURRENT_DIR, "sample_data")

    # 1. Test Table Integrity
    print("\n[Test 1] Testing Pricing Table Integrity & Header Injection...")
    pricing_file = os.path.join(sample_dir, "sample_pricing.md")
    with open(pricing_file, "r", encoding="utf-8") as f:
        pricing_text = f.read()

    table_chunker = PricingTableChunker(rows_per_chunk=1)
    meta = {"doc_title": "Apex Dental Guide", "currency": "INR"}
    table_chunks = table_chunker.chunk(pricing_text, meta)
    assert len(table_chunks) == 10, f"Expected 10 table rows, got {len(table_chunks)}"
    for c in table_chunks:
        assert "Treatment" in c.searchable_augmented_text, "Missing Treatment header"
        assert "Base Price" in c.searchable_augmented_text, "Missing Base Price header"
        assert "Warranty" in c.searchable_augmented_text, "Missing Warranty header"
        assert c.chunk_type == "pricing_table"
        assert c.start_char >= 0 and c.end_char > c.start_char
    print(f"  ✅ Passed: 10/10 table row chunks retained headers & currency anchor.")

    # 2. Test Location Integrity
    print("\n[Test 2] Testing Location Entity Indivisible Cards...")
    loc_file = os.path.join(sample_dir, "sample_locations.txt")
    with open(loc_file, "r", encoding="utf-8") as f:
        loc_text = f.read()

    loc_chunker = LocationEntityChunker()
    loc_chunks = loc_chunker.chunk(loc_text, meta)
    assert len(loc_chunks) == 4, f"Expected exactly 4 location chunks, got {len(loc_chunks)}"
    for c in loc_chunks:
        assert c.chunk_type == "location"
        assert "Phone" in c.raw_text or "Call" in c.raw_text
        assert "Hours" in c.raw_text or "Timing" in c.raw_text
    print(f"  ✅ Passed: Exactly 4 discrete location card chunks created.")

    # 3. Test FAQ Integrity
    print("\n[Test 3] Testing Atomic FAQ 1:1 Q&A Pair Rule...")
    faq_file = os.path.join(sample_dir, "sample_faq.txt")
    with open(faq_file, "r", encoding="utf-8") as f:
        faq_text = f.read()

    faq_chunker = AtomicFAQChunker()
    faq_chunks = faq_chunker.chunk(faq_text, meta)
    assert len(faq_chunks) == 10, f"Expected exactly 10 FAQ chunks, got {len(faq_chunks)}"
    for c in faq_chunks:
        assert c.chunk_type == "faq"
        assert "Question" in c.raw_text
        assert "Answer" in c.raw_text
    print(f"  ✅ Passed: Exactly 10 discrete Q&A chunks created.")

    # 4. Test Hybrid Detection & Multi-Zone Chunking
    print("\n[Test 4] Testing Hybrid Composite Detection & Multi-Zone Dispatch...")
    comp_file = os.path.join(sample_dir, "sample_comprehensive_guide.md")
    with open(comp_file, "r", encoding="utf-8") as f:
        comp_text = f.read()

    detected_strat, reason = detect_document_strategy(comp_text)
    assert detected_strat == "hybrid_composite", f"Expected hybrid_composite, got {detected_strat}"
    
    hybrid_chunker = HybridCompositeChunker()
    comp_chunks = hybrid_chunker.chunk(comp_text, meta)
    types_found = set(c.chunk_type for c in comp_chunks)
    assert "pricing_table" in types_found, "Missing pricing_table in hybrid chunks"
    assert "location" in types_found, "Missing location in hybrid chunks"
    assert "faq" in types_found, "Missing faq in hybrid chunks"
    assert "semantic" in types_found, "Missing semantic in hybrid chunks"
    print(f"  ✅ Passed: Hybrid detection succeeded ({reason}); produced {len(comp_chunks)} chunks across {types_found}.")

    # 5. Test Auditor Validation (Orphaned Price Detection)
    print("\n[Test 5] Testing Voice Readiness Auditor Linting Rules...")
    auditor = VoiceChunkAuditor()
    # Bad chunk: floating price with no headers
    bad_chunk = ChunkResult(
        chunk_id=99,
        start_char=0,
        end_char=50,
        raw_text="Zirconia Crown ₹6,000 /mo 15 yrs",
        searchable_augmented_text="Zirconia Crown ₹6,000 /mo 15 yrs",
        spoken_clean_text="Zirconia Crown 6,000 rupees per month 15 yrs.",
        chunk_type="pricing_table",
        word_count=8,
        speech_seconds=3.2
    )
    bad_warnings = auditor.audit_chunk(bad_chunk)
    assert any(w["badge"] == "🚨 Critical: Severed Price Row" for w in bad_warnings), "Failed to detect severed price row"
    print(f"  ✅ Passed: Orphaned price row correctly flagged with '🚨 Critical: Severed Price Row'.")

    # 6. Test Acoustic Timing
    print("\n[Test 6] Testing Acoustic Timing Calculation...")
    words_150 = "word " * 150
    seconds = calculate_speech_seconds(words_150)
    assert abs(seconds - 60.0) < 0.001, f"Expected 60.0s for 150 words, got {seconds}"
    print(f"  ✅ Passed: Exactly 150 words calculates to {seconds:.1f}s speech time.")

    # 7. Audit Complete Test
    audit_summary = auditor.audit_chunks(table_chunks)
    assert audit_summary["quality_score"] == 100.0, f"Expected 100% quality score, got {audit_summary['quality_score']}%"
    print(f"  ✅ Passed: Auditor quality score is {audit_summary['quality_score']}% for properly structured table chunks.")

    print("\n" + "=" * 60)
    print("🏆 ALL ACCEPTANCE TESTS PASSED WITH 100% COMPLIANCE!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
