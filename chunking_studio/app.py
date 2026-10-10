import os
import sys
import json
import streamlit as st

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dotenv import load_dotenv
load_dotenv()

from src.document_parsers import parse_uploaded_file
from src.chunkers import detect_document_strategy, get_chunker
from src.auditor import VoiceChunkAuditor
from src.normalizer import GeminiDocumentNormalizer

st.set_page_config(
    page_title="Voice AI Chunking App",
    page_icon="🎙️",
    layout="centered"
)

# Minimal clean styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.chunk-box {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
    color: #f3f4f6;
}
.chunk-title {
    font-size: 16px;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
}
.sub-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 10px;
    margin-bottom: 4px;
}
.text-searchable {
    color: #fbbf24;
}
.text-spoken {
    color: #34d399;
}
pre {
    background: #1f2937;
    border-radius: 6px;
    padding: 10px;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 13px;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Voice AI Document Chunker")
st.write("Upload or paste a document, select settings, and click **Chunk Document**.")

# 1. Document Input
st.subheader("1. Document")
uploaded_file = st.file_uploader("Upload File (.txt, .md, .pdf, .docx)", type=["txt", "md", "pdf", "docx"])

default_sample = ""
sample_path = os.path.join(CURRENT_DIR, "sample_data", "sample_pricing.md")
if os.path.exists(sample_path):
    with open(sample_path, "r", encoding="utf-8") as f:
        default_sample = f.read()

doc_text = ""
if uploaded_file is not None:
    doc_text, _ = parse_uploaded_file(uploaded_file, uploaded_file.name)
    st.success(f"Loaded {uploaded_file.name}")
else:
    doc_text = st.text_area("Or Paste Document Text Here:", value=default_sample, height=220)

# 2. Parameters (Unit, Chunk size, overlap, model)
st.subheader("2. Settings")
col0, col1, col2, col3 = st.columns(4)

with col0:
    unit_choice = st.selectbox("Unit", options=["Characters", "Tokens"], index=0)

default_chunk = 500 if unit_choice == "Characters" else 150
default_overlap = 100 if unit_choice == "Characters" else 25

with col1:
    chunk_size = st.number_input(f"Chunk Size ({unit_choice})", min_value=10, max_value=5000, value=default_chunk, step=10)

with col2:
    overlap = st.number_input(f"Overlap ({unit_choice})", min_value=0, max_value=1000, value=default_overlap, step=5)

with col3:
    model_choice = st.selectbox(
        "Model",
        options=["gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash", "None (Deterministic Engine Only)"],
        index=0
    )

# 3. Action Button
st.write("")
if st.button("🚀 Chunk Document", type="primary", use_container_width=True):
    if not doc_text or not doc_text.strip():
        st.error("Please provide document text or upload a file.")
    else:
        with st.spinner("Processing chunks..."):
            active_text = doc_text.strip()
            
            # Run Gemini if requested
            if "gemini" in model_choice:
                api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
                normalizer = GeminiDocumentNormalizer(api_key=api_key, model_name=model_choice)
                norm_text, norm_meta = normalizer.normalize(active_text)
                if norm_meta.get("status") == "success":
                    active_text = norm_text
                    st.caption(f"✨ Document structured by {model_choice}")

            # Chunk document
            strategy, _ = detect_document_strategy(active_text)
            chunker = get_chunker(
                strategy,
                unit=unit_choice.lower(),
                chunk_size=int(chunk_size),
                overlap=int(overlap),
                target_words=int(chunk_size),
                overlap_words=int(overlap)
            )
            metadata = {"doc_title": uploaded_file.name if uploaded_file else "Document"}
            chunks = chunker.chunk(active_text, metadata=metadata)

            # Audit
            auditor = VoiceChunkAuditor()
            audit_summary = auditor.audit_chunks(chunks)

        st.success(f"Generated {len(chunks)} Chunks • Quality Score: {audit_summary['quality_score']}%")

        # Display Chunks
        st.subheader("3. Output Chunks")
        
        # Download button
        chunks_json = json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download Chunks (JSON)",
            data=chunks_json,
            file_name="output_chunks.json",
            mime="application/json"
        )

        for c in chunks:
            lint_text = "🛡️ Passed" if not c.warnings else f"⚠️ {c.warnings[0]['badge']}"
            
            st.markdown(f"""
            <div class="chunk-box">
                <div class="chunk-title">
                    <span>Chunk #{c.chunk_id} ({c.chunk_type.upper()})</span>
                    <span style="font-size:12px; font-weight:normal; color:#9ca3af;">{c.char_count} chars • ~{c.token_count} tokens</span>
                </div>
                <div class="sub-label" style="color: #38bdf8;">📄 Chunk Content (Ready for Embedding):</div>
                <pre>{c.text or c.searchable_augmented_text or c.raw_text}</pre>
            </div>
            """, unsafe_allow_html=True)
