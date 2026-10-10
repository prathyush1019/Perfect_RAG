import os
import sys
import argparse
import json

# Add chunking_studio to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIO_DIR = os.path.join(BASE_DIR, "chunking_studio")
if STUDIO_DIR not in sys.path:
    sys.path.insert(0, STUDIO_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from chunking_studio.src.chunkers import detect_document_strategy, get_chunker
from chunking_studio.src.auditor import VoiceChunkAuditor
from chunking_studio.src.normalizer import GeminiDocumentNormalizer
from chunking_studio.src.document_parsers import parse_uploaded_file


def main():
    parser = argparse.ArgumentParser(description="Voice AI CAG Chunking Engine CLI Runner")
    parser.add_argument(
        "--file",
        "-f",
        default=os.path.join(STUDIO_DIR, "sample_data", "sample_pricing.md"),
        help="Path to input document (.md, .txt, .pdf, .docx)"
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Enable Stage 1 Gemini API Normalizer"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API Key"
    )
    parser.add_argument(
        "--strategy",
        default="auto",
        help="Strategy: auto, pricing_table, location, faq, semantic, hybrid_composite"
    )
    parser.add_argument(
        "--output-json",
        "-o",
        default=None,
        help="Path to write JSON output chunks"
    )
    args = parser.parse_args()

    print("=" * 75)
    print("🎙️ VOICE AI CONTEXT-AG CHUNKING ENGINE")
    print("=" * 75)
    print(f"📄 Input Document: {args.file}")

    if not os.path.exists(args.file):
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)

    with open(args.file, "rb") as f:
        text, file_ext = parse_uploaded_file(f, os.path.basename(args.file))

    print(f"📊 Document Length: {len(text)} characters, {len(text.split())} words")

    # Stage 1: Gemini Normalizer if requested
    if args.use_gemini:
        print("\n[STAGE 1] Running Gemini API Structural Normalizer...")
        normalizer = GeminiDocumentNormalizer(api_key=args.api_key)
        text, meta = normalizer.normalize(text)
        print(f"  Stage 1 Result: {meta.get('status')} ({meta.get('message', 'Completed')})")

    # Strategy detection
    if args.strategy == "auto":
        strat, reason = detect_document_strategy(text)
        print(f"\n[STAGE 2] Strategy Auto-Detected: {strat.upper()} ({reason})")
    else:
        strat = args.strategy
        print(f"\n[STAGE 2] Using Specified Strategy: {strat.upper()}")

    metadata = {
        "doc_title": os.path.basename(args.file),
        "entity_name": "Apex Dental Care",
        "category": "Pricing & Operating Policies",
        "currency": "INR",
        "language": "en-US"
    }

    chunker = get_chunker(strat)
    chunks = chunker.chunk(text, metadata=metadata)

    # Auditor
    auditor = VoiceChunkAuditor()
    audit_summary = auditor.audit_chunks(chunks)

    print(f"✅ Generated {len(chunks)} discrete chunks. Voice Readiness Score: {audit_summary['quality_score']}%\n")

    print("-" * 75)
    print("📦 CHUNKS OUTPUT:")
    print("-" * 75)

    for c in chunks:
        print(f"\n🔹 CHUNK #{c.chunk_id} [{c.chunk_type.upper()}] | {c.char_count} chars | ~{c.token_count} tokens | Offset: {c.start_char}-{c.end_char}")
        print("  ┌── CHUNK CONTENT (Ready for Embedding):")
        chunk_body = c.text or c.searchable_augmented_text or c.raw_text
        for line in chunk_body.splitlines():
            print(f"  │  {line}")
        print("  └──")

    print("\n" + "=" * 75)
    print(f"🎉 Complete! Total {len(chunks)} discrete chunks produced.")
    print("=" * 75)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)
        print(f"💾 Saved chunks to {args.output_json}")


if __name__ == "__main__":
    main()
