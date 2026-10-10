import html
from typing import List, Dict, Any
import streamlit as st
from .styles import BAND_CLASSES, BAND_COLORS
from ..chunkers.base import ChunkResult


def render_kpi_dashboard(
    doc_word_count: int,
    chunks_count: int,
    avg_words: float,
    voice_quality_score: float
):
    """Renders the top 4 KPI glassmorphic cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Document Volume</div>
            <div class="kpi-value">{doc_word_count:,}</div>
            <div style="font-size:0.75rem; color:#94a3b8;">Total Raw Words</div>
            <div class="kpi-accent" style="background:#6366f1;"></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Chunks Produced</div>
            <div class="kpi-value">{chunks_count}</div>
            <div style="font-size:0.75rem; color:#94a3b8;">Discrete Voice Units</div>
            <div class="kpi-accent" style="background:#06b6d4;"></div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Chunk Size</div>
            <div class="kpi-value">{avg_words:.1f}</div>
            <div style="font-size:0.75rem; color:#94a3b8;">Words per Chunk</div>
            <div class="kpi-accent" style="background:#a855f7;"></div>
        </div>
        """, unsafe_allow_html=True)

    score_color = "#10b981" if voice_quality_score >= 90 else "#eab308" if voice_quality_score >= 70 else "#ef4444"
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Voice Readiness</div>
            <div class="kpi-value" style="color:{score_color};">{voice_quality_score:.1f}%</div>
            <div style="font-size:0.75rem; color:#94a3b8;">Linter Pass Rate</div>
            <div class="kpi-accent" style="background:{score_color};"></div>
        </div>
        """, unsafe_allow_html=True)


def render_chunk_drawing_canvas(source_text: str, chunks: List[ChunkResult]):
    """
    Renders the live chunk visual canvas.
    Maps alternating highlight color bands across the text with explicit cutline markers.
    """
    if not chunks:
        st.info("No chunks to draw yet. Upload a document or select a sample dataset.")
        return

    st.markdown("### 🗺️ Live Chunk Boundary Visualization (Visual Canvas)")
    st.caption("Direct visual mapping of chunk partitions with alternating color bands and explicit character cutlines.")

    canvas_html_parts = []
    canvas_html_parts.append('<div class="glass-panel" style="max-height: 520px; overflow-y: auto; padding: 1.25rem;">')

    for i, chunk in enumerate(chunks):
        band_class = BAND_CLASSES[i % len(BAND_CLASSES)]
        band_color = BAND_COLORS[i % len(BAND_COLORS)]

        # Cutline marker between chunks
        if i > 0:
            prev_chunk = chunks[i - 1]
            canvas_html_parts.append(f"""
            <div class="cutline-marker">
                <span>✂️ <strong>Boundary #{i}</strong></span>
                <span>| End of Chunk {prev_chunk.chunk_id} ➔ Start of Chunk {chunk.chunk_id}</span>
                <span style="color:#38bdf8; margin-left:auto;">[Offset: {chunk.start_char}]</span>
            </div>
            """)

        escaped_text = html.escape(chunk.raw_text).replace("\n", "<br>")
        canvas_html_parts.append(f"""
        <div class="{band_class}" style="padding: 0.85rem 1rem; border-radius: 6px; margin: 0.35rem 0; font-family:'JetBrains Mono', monospace; font-size: 0.82rem; line-height: 1.6;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem; font-size:0.75rem; font-weight:700; color:{band_color}; text-transform:uppercase;">
                <span>Chunk #{chunk.chunk_id} [{chunk.chunk_type.upper()}]</span>
                <span>[Chars {chunk.start_char} – {chunk.end_char}] • {chunk.word_count} words • ~{chunk.speech_seconds:.1f}s</span>
            </div>
            <div>{escaped_text}</div>
        </div>
        """)

    canvas_html_parts.append('</div>')
    st.markdown("".join(canvas_html_parts), unsafe_allow_html=True)


def render_chunks_inspector(chunks: List[ChunkResult]):
    """
    Renders the interactive Final Chunks Inspector with Side-by-Side Dual-Text comparison.
    """
    if not chunks:
        return

    st.markdown("### 🧩 Final Clean Chunks Output (Dual-Text Inspector)")
    st.caption("Compare vector-indexed searchable text vs natural TTS speech synthesis.")

    for chunk in chunks:
        type_badge_class = {
            "pricing_table": "badge-amber",
            "location": "badge-cyan",
            "faq": "badge-purple",
            "semantic": "badge-emerald"
        }.get(chunk.chunk_type, "badge-cyan")

        # Warnings badge
        warning_badges = []
        if chunk.warnings:
            for w in chunk.warnings:
                bclass = "badge-red" if w["severity"] == "critical" else "badge-yellow"
                warning_badges.append(f'<span class="badge {bclass}">{html.escape(w["badge"])}</span>')
        else:
            warning_badges.append('<span class="badge badge-emerald">🛡️ All Voice Lints Passed</span>')

        warnings_html = " ".join(warning_badges)

        speech_time_badge = f'<span class="badge badge-cyan">⏱️ Speech: ~{chunk.speech_seconds:.1f}s</span>'

        with st.container():
            st.markdown(f"""
            <div class="glass-panel" style="margin-bottom: 1.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.6rem; margin-bottom: 0.75rem;">
                    <div>
                        <span style="font-family:'Outfit', sans-serif; font-size:1.15rem; font-weight:700; color:#f8fafc; margin-right: 0.75rem;">Chunk #{chunk.chunk_id}</span>
                        <span class="badge {type_badge_class}">{chunk.chunk_type.replace('_', ' ').title()}</span>
                        <span class="badge" style="background:rgba(255,255,255,0.08); color:#cbd5e1;">Chars: {chunk.start_char}–{chunk.end_char}</span>
                        <span class="badge" style="background:rgba(255,255,255,0.08); color:#cbd5e1;">{chunk.word_count} words</span>
                        {speech_time_badge}
                    </div>
                    <div>
                        {warnings_html}
                    </div>
                </div>

                <div class="dual-text-container">
                    <!-- Left: Searchable Augmented Text (Vector target) -->
                    <div class="dual-box box-searchable">
                        <div class="box-title box-title-searchable">
                            <span>🔍 Searchable Augmented Text (Vector Target)</span>
                        </div>
                        <div class="code-view">{html.escape(chunk.searchable_augmented_text)}</div>
                    </div>

                    <!-- Right: Spoken Clean Text (TTS speech target) -->
                    <div class="dual-box box-spoken">
                        <div class="box-title box-title-spoken">
                            <span>🎙️ Spoken Clean Text (TTS Target)</span>
                        </div>
                        <div class="code-view" style="color:#a7f3d0;">{html.escape(chunk.spoken_clean_text)}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Web Speech API Audio Preview button
            col_a, col_b = st.columns([1, 4])
            with col_a:
                escaped_speech = html.escape(chunk.spoken_clean_text).replace("'", "\\'").replace('"', '&quot;')
                button_id = f"speak_btn_{chunk.chunk_id}"
                html_snippet = f"""
                <button id="{button_id}" onclick="speakText()" style="background:#10b981; color:#0f172a; border:none; border-radius:6px; padding:6px 14px; font-weight:600; font-size:12px; cursor:pointer; display:flex; align-items:center; gap:6px;">
                    🔊 Listen (TTS)
                </button>
                <script>
                function speakText() {{
                    if ('speechSynthesis' in window) {{
                        window.speechSynthesis.cancel();
                        const utterance = new SpeechSynthesisUtterance('{escaped_speech}');
                        utterance.rate = 1.0;
                        window.speechSynthesis.speak(utterance);
                    }} else {{
                        alert('Browser does not support Speech Synthesis API');
                    }}
                }}
                </script>
                """
                if hasattr(st, "html"):
                    st.html(html_snippet)
                else:
                    st.components.v1.html(html_snippet, height=40)
