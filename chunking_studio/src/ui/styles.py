"""
UI Styling & Theme Tokens for Voice AI Chunking Studio.
Dark glassmorphic aesthetics with neon accents, Inter & Outfit typography.
"""

DARK_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    color: #e2e8f0;
}

h1, h2, h3, h4, .brand-title {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Glassmorphic Cards */
.glass-panel {
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-bottom: 1.25rem;
}

/* KPI Metric Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(31, 41, 55, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.5);
}

.kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.1;
    margin: 0.35rem 0;
}

.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
}

.kpi-accent {
    height: 3px;
    width: 100%;
    position: absolute;
    bottom: 0;
    left: 0;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
}

.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-cyan {
    background: rgba(6, 182, 212, 0.15);
    color: #22d3ee;
    border: 1px solid rgba(6, 182, 212, 0.3);
}

.badge-purple {
    background: rgba(168, 85, 247, 0.15);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.3);
}

.badge-emerald {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-red {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.badge-yellow {
    background: rgba(234, 179, 8, 0.2);
    color: #fde047;
    border: 1px solid rgba(234, 179, 8, 0.4);
}

/* Visual Canvas Band Tints */
.band-amber {
    background-color: rgba(243, 128, 32, 0.14);
    border-left: 3px solid #f97316;
}

.band-cyan {
    background-color: rgba(0, 210, 180, 0.14);
    border-left: 3px solid #14b8a6;
}

.band-purple {
    background-color: rgba(168, 85, 247, 0.14);
    border-left: 3px solid #a855f7;
}

.band-emerald {
    background-color: rgba(16, 185, 129, 0.14);
    border-left: 3px solid #10b981;
}

/* Cutline marker */
.cutline-marker {
    background: rgba(15, 23, 42, 0.9);
    border: 1px dashed #64748b;
    border-radius: 6px;
    padding: 0.35rem 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #cbd5e1;
    margin: 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Side by Side Dual Text Box */
.dual-text-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 0.75rem;
}

.dual-box {
    border-radius: 8px;
    padding: 0.85rem;
    font-size: 0.875rem;
    line-height: 1.5;
    background: rgba(15, 23, 42, 0.6);
}

.box-searchable {
    border: 1px solid rgba(245, 158, 11, 0.5);
    background: rgba(245, 158, 11, 0.04);
}

.box-spoken {
    border: 1px solid rgba(16, 185, 129, 0.5);
    background: rgba(16, 185, 129, 0.04);
}

.box-title {
    font-family: 'Outfit', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.box-title-searchable {
    color: #fbbf24;
}

.box-title-spoken {
    color: #34d399;
}

.code-view {
    font-family: 'JetBrains Mono', monospace;
    white-space: pre-wrap;
    word-break: break-word;
    color: #cbd5e1;
    font-size: 0.82rem;
}
</style>
"""

BAND_CLASSES = ["band-amber", "band-cyan", "band-purple", "band-emerald"]
BAND_COLORS = ["#f97316", "#14b8a6", "#a855f7", "#10b981"]
