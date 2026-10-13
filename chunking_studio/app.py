import os
import sys
import json
import time
import uuid
import streamlit as st
import concurrent.futures

# Setup paths
STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(STUDIO_DIR)
if STUDIO_DIR not in sys.path:
    sys.path.insert(0, STUDIO_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv()

from src.document_parsers import parse_uploaded_file
from src.chunkers import detect_document_strategy, get_chunker
from src.chunkers.base import count_units, split_text_to_ceiling, ChunkResult
from src.auditor import VoiceChunkAuditor
from src.normalizer import GeminiDocumentNormalizer

from src.cloudflare_config import ALL_CLOUDFLARE_MODELS, AVAILABLE_MODEL_IDS
from src.cloudflare_embedder import CloudflareEmbedder
from src.qdrant_manager import LocalQdrantManager

@st.cache_resource
def get_embedder():
    return CloudflareEmbedder()

@st.cache_resource
def get_qdrant():
    return LocalQdrantManager()

embedder = get_embedder()
qdrant = get_qdrant()

st.set_page_config(
    page_title="Perfect RAG — Cloudflare & Qdrant Studio",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Perfect RAG — Cloudflare & Qdrant Studio")
st.caption("Context-Augmented Chunking • 7 Cloudflare Embedding Models • Multi-Document Filtered Search")

tab1, tab2, tab3 = st.tabs([
    "📄 1. Chunk & Index Document",
    "🔍 2. Question & Multi-Model Search",
    "📁 3. Qdrant Collections & Documents"
])

# ----------------- TAB 1: CHUNK & INDEX -----------------
with tab1:
    st.subheader("1. Document Input & Chunking")

    default_sample = """1.ஸ்ரீராம்கார்டன்

Guideline Value:nine two five

Main Content:
காரமடை ஜங்ஷன்ல இருந்து மேட்டுப்பாளையம் ரோட்ல டீச்சர்ஸ் காலனில நம்ம site இருக்குங்க. மேட்டுப்பாளையம் ரோட்ல இருந்து just Seven Hundred metersல T G Pudur Road மேல site இருக்குங்க. அங்க one Acreல Twentyfive plots இருக்குங்க. அங்க பாத்தீங்கனா full gated community site தாங்க. உங்களுக்கு full facilities ஓட நம்மகிட்ட sites இருக்குங்க. Minimum two centல இருந்து five cent வரைக்கும் நம்மகிட்ட site availableஆ இருக்குங்க. Rate பாத்தீங்கனா eight lakhs twenty five Thousand தாங்க.

Location:
காரமடை ஜங்ஷன்ல இருந்து மேட்டுப்பாளையம் ரோட்ல வந்தா டீச்சர்ஸ் காலனில T G Pudur Roadல seven hundred meters வந்தீங்கனா ரோட்லயே site இருக்குங்க. மடோனா ஸ்கூல் Road அல்லது தெக்கம் பட்டி போற ரோட்ல நம்ம site இருக்குங்க.

Facilities:
இந்த projectல gated community, solar street lights, EB supply, thirty three feet tar road, individual water supply, drainage facility மாதிரி எல்லா Facilities இருக்குங்க.

Approval:DEE TEE CEE PEE and RE RAH approval இருக்குதுங்க 

Nearby locations:
இந்த projectலிருந்து காரமடை just three minutes distanceல, two point five kilometersல இருக்குங்க.
மேட்டுப்பாளையம் Bus Stand eight minutes distanceல, five point five kilometersல இருக்குங்க.
மடோனா Matriculation Higher Secondary School three minutes distanceல, one point five kilometersல இருக்குங்க.
Savidha Hospital two minutes distanceல, one point five kilometersல இருக்குங்க.

Construction Details:
One BHK option two cents land area siteல five hundred square feet builtup area இருக்கும். Price thirty four lakhsல ஆரம்பிக்குதுங்க.
Two BHK option two cents land area siteல eight hundred square feet builtup area இருக்கும். Price thirty eight lakhsல ஆரம்பிக்குதுங்க.
Three BHK option three cents land area siteல one thousand square feet builtup area இருக்கும். Price forty seven lakhsல ஆரம்பிக்குதுங்க.

---

2.பிரம்மா பார்க்

Guideline Value: two five zero

Main Content:
அன்னூர் to சிறுமுகை ரோட்ல கைகட்டி ஊர்ல, அன்னூர் ரோடு மேலயே நம்ம site இருக்குங்க. அங்க three Acreல seventy plots போட்டுருக்காங்க. அங்க பாத்தீங்கனா full gated community site தாங்க. உங்களுக்கு full facilities ஓட நம்மகிட்ட sites இருக்குங்க. Minimum two centல இருந்து five point five cent வரைக்கும் நம்மகிட்ட site availableஆ இருக்குங்க. Rate பாத்தீங்கனா three lakhs ninety Thousandல இருந்து four lakhs twenty five Thousand வரை போயிட்டு இருக்குங்க."""

    doc_name = st.text_input("Document Name / Title", value="Real Estate Guide - Coimbatore")

    uploaded_file = st.file_uploader("Upload File (PDF, DOCX, TXT, MD)", type=["txt", "md", "pdf", "docx"])
    if uploaded_file is not None:
        doc_text, _ = parse_uploaded_file(uploaded_file, uploaded_file.name)
        st.success(f"Loaded {uploaded_file.name}")
    else:
        doc_text = st.text_area("Or Paste Document Text Here:", value=default_sample, height=200)

    col0, col1, col2, col3 = st.columns(4)
    with col0:
        unit_choice = st.selectbox("Unit", options=["Tokens", "Characters"], index=0)
    default_chunk = 512 if unit_choice == "Tokens" else 2000
    default_overlap = 50 if unit_choice == "Tokens" else 150
    with col1:
        chunk_size = st.number_input(f"Max Ceiling ({unit_choice})", min_value=50, max_value=8192, value=default_chunk, step=10)
    with col2:
        overlap = st.number_input(f"Overlap ({unit_choice})", min_value=0, max_value=1000, value=default_overlap, step=5)
    with col3:
        model_choice = st.selectbox(
            "LLM Normalizer",
            options=["None (Deterministic Fast)", "gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
            index=0
        )

    if st.button("⚡ Chunk Document", type="primary"):
        if not doc_text or not doc_text.strip():
            st.error("Please provide document text.")
        else:
            with st.spinner("Chunking document..."):
                active_text = doc_text.strip()
                if "gemini" in model_choice:
                    api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
                    normalizer = GeminiDocumentNormalizer(api_key=api_key, model_name=model_choice)
                    norm_text, norm_meta = normalizer.normalize(active_text)
                    if norm_meta.get("status") == "success":
                        active_text = norm_text

                strat, _ = detect_document_strategy(active_text)
                chunker_inst = get_chunker(
                    strat,
                    unit=unit_choice.lower(),
                    chunk_size=int(chunk_size),
                    overlap=int(overlap),
                    target_words=int(chunk_size),
                    overlap_words=int(overlap)
                )
                raw_chunks = chunker_inst.chunk(active_text, metadata={"doc_title": doc_name})

                # Hard ceiling enforcement
                final_chunks = []
                curr_id = 1
                for c in raw_chunks:
                    c_text = c.text or c.searchable_augmented_text or c.raw_text
                    if count_units(c_text, unit_choice.lower()) > chunk_size:
                        entity_title = c.metadata.get("branch_name") or (c_text.splitlines()[0].strip() if c_text.splitlines() else "Entity")
                        sub_pieces = split_text_to_ceiling(c_text, chunk_size, unit_choice.lower())
                        for sub_idx, sub in enumerate(sub_pieces, 1):
                            anchored_sub = sub
                            if sub_idx > 1 and entity_title and entity_title not in sub[:120]:
                                anchored_sub = f"[{entity_title} (Part {sub_idx}/{len(sub_pieces)})]\n" + sub
                            final_chunks.append(
                                ChunkResult(
                                    chunk_id=curr_id,
                                    start_char=c.start_char,
                                    end_char=c.end_char,
                                    text=anchored_sub,
                                    raw_text=anchored_sub,
                                    searchable_augmented_text=anchored_sub,
                                    chunk_type=c.chunk_type,
                                    metadata={**c.metadata, "parent_entity": entity_title, "part": sub_idx, "total_parts": len(sub_pieces)}
                                )
                            )
                            curr_id += 1
                    else:
                        c.chunk_id = curr_id
                        final_chunks.append(c)
                        curr_id += 1

                st.session_state["active_chunks"] = [c.to_dict() for c in final_chunks]
                st.session_state["active_doc_name"] = doc_name

    if "active_chunks" in st.session_state and st.session_state["active_chunks"]:
        chunks = st.session_state["active_chunks"]
        st.success(f"Generated {len(chunks)} Chunks!")

        if st.button("💾 Embed & Save All to Qdrant (All 4 BGE Models)", type="secondary"):
            with st.spinner("Generating embeddings across all BGE models & upserting into Qdrant..."):
                chunk_texts = [c["text"] for c in chunks]
                doc_id = f"doc_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                embeddings_by_model = {}

                def _embed(mid):
                    vecs, _ = embedder.embed_texts(chunk_texts, model_id=mid)
                    return mid, vecs

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                    results = list(ex.map(_embed, AVAILABLE_MODEL_IDS))

                for mid, vecs in results:
                    embeddings_by_model[mid] = vecs

                upsert_res = qdrant.upsert_document_chunks(
                    doc_id=doc_id,
                    doc_name=st.session_state.get("active_doc_name", "Document"),
                    chunks=chunks,
                    embeddings_by_model=embeddings_by_model
                )
                st.success(f"✅ Successfully indexed into Qdrant across all 4 BGE models! Document ID: `{doc_id}`")

        for c in chunks:
            with st.expander(f"Chunk #{c['chunk_id']} [{c.get('chunk_type', 'text').upper()}] ({c.get('char_count', len(c['text']))} chars • ~{c.get('token_count', len(c['text'])//4)} tokens)"):
                st.code(c["text"], language="markdown")


# ----------------- TAB 2: QUESTION & SEARCH -----------------
with tab2:
    st.subheader("2. Question & Multi-Model Semantic Search (Filtered by Document)")

    docs = qdrant.list_documents()
    if not docs:
        st.info("No documents indexed in Qdrant yet. Please index a document in Tab 1 first.")
    else:
        doc_options = {d["doc_id"]: f"{d['doc_name']} ({d['chunk_count']} chunks)" for d in docs}
        selected_ids = st.multiselect(
            "🎯 Select Documents to Search From (Qdrant HNSW Filter):",
            options=list(doc_options.keys()),
            default=list(doc_options.keys()),
            format_func=lambda x: doc_options[x]
        )

        query = st.text_input("Question / Search Query", value="What are the facilities and BHK construction prices in Sriram Garden?")
        top_k = st.slider("Top Chunks per Model", min_value=1, max_value=10, value=3)

        if st.button("🔎 Search Across All BGE Models", type="primary"):
            if not selected_ids:
                st.warning("Please select at least one document to search from.")
            elif not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Querying all BGE models in parallel..."):
                    def _search(mid):
                        m_cfg = ALL_CLOUDFLARE_MODELS[mid]
                        q_vec, api_lat = embedder.embed_single_query(query, model_id=mid)
                        hits, q_lat = qdrant.search(
                            query_vector=q_vec,
                            model_id=mid,
                            doc_ids=selected_ids,
                            top_k=top_k,
                            raw_query=query
                        )
                        return {
                            "mid": mid,
                            "name": m_cfg.display_name,
                            "dims": m_cfg.dimensions,
                            "badge_color": m_cfg.badge_color,
                            "api_lat": round(api_lat, 2),
                            "q_lat": round(q_lat, 2),
                            "total_lat": round(api_lat + q_lat, 2),
                            "hits": hits
                        }

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                        search_results = list(ex.map(_search, AVAILABLE_MODEL_IDS))

                    st.write("### Comparative Multi-Model Results")
                    cols = st.columns(2)
                    for i, res in enumerate(search_results):
                        with cols[i % 2]:
                            st.markdown(f"#### {res['name']}")
                            st.caption(f"**{res['dims']}d** • ⚡ Total Latency: **{res['total_lat']} ms** (API: {res['api_lat']}ms | Qdrant: {res['q_lat']}ms)")
                            if not res["hits"]:
                                st.write("No matching chunks found in selected documents.")
                            for h_idx, h in enumerate(res["hits"], 1):
                                st.markdown(f"**Hit #{h_idx}** • Score: `{h['score']}` • Doc: `{h['doc_name']}` (Chunk #{h['chunk_id']})")
                                st.code(h["text"], language="markdown")


# ----------------- TAB 3: STORAGE STATS -----------------
with tab3:
    st.subheader("3. Qdrant Collections & Indexed Documents")

    all_docs = qdrant.list_documents()
    st.write("#### Indexed Documents:")
    if not all_docs:
        st.write("No documents in Qdrant.")
    else:
        for d in all_docs:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.write(f"📄 **{d['doc_name']}** (ID: `{d['doc_id']}` • {d['chunk_count']} Chunks)")
            with c2:
                if st.button("🗑️ Delete", key=f"del_{d['doc_id']}"):
                    qdrant.delete_document(d["doc_id"])
                    st.rerun()

    st.write("#### Qdrant Collections Stats:")
    stats = qdrant.get_collection_stats()
    grid_cols = st.columns(2)
    for idx, (mid, s) in enumerate(stats.items()):
        with grid_cols[idx % 2]:
            st.info(f"**{s['display_name']}**\n- Collection: `{s['name']}`\n- Dimensions: **{s['dimensions']}d**\n- Stored Vectors: **{s['points_count']}**")
