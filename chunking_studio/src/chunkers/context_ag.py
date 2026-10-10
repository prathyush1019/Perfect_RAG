import re
from typing import Dict, Any, Optional, List


def format_context_header(metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Builds the Context-AG bracketed header tag for vector indexing.
    Format: [Document: Apex Dental Guide 2026 | Entity: Apex Dental Care | Category: Pricing | ...]
    """
    if not metadata:
        return ""

    parts = []
    keys_order = [
        ("doc_title", "Document"),
        ("entity_name", "Entity"),
        ("category", "Category"),
        ("location", "Location"),
        ("currency", "Currency"),
        ("language", "Language")
    ]

    for key, label in keys_order:
        val = metadata.get(key)
        if val:
            parts.append(f"{label}: {str(val).strip()}")

    # Include any custom metadata keys
    for k, v in metadata.items():
        if k not in [k_ord[0] for k_ord in keys_order] and v:
            clean_k = k.replace("_", " ").title()
            parts.append(f"{clean_k}: {str(v).strip()}")

    if not parts:
        return ""

    return f"[{' | '.join(parts)}]"


def build_searchable_augmented_text(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    section_context: str = ""
) -> str:
    """
    Prepends Context-AG metadata brackets and section context to the text for vector search.
    """
    header = format_context_header(metadata)
    components = []
    if header:
        components.append(header)
    if section_context:
        components.append(f"[Section: {section_context.strip()}]")
    components.append(text.strip())
    return "\n".join(components)


def convert_table_to_spoken(text: str) -> str:
    """
    Converts markdown table row representation into natural conversational phrasing.
    e.g.:
    | Treatment | Base Price | Warranty |
    | Zirconia Crown | ₹6,000 | 15 Years Replacement Warranty |
    --> "Treatment: Zirconia Crown, Base Price: 6,000 rupees, Warranty: 15 Years Replacement Warranty."
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    table_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]

    if len(table_lines) < 2:
        return text

    # Extract headers
    header_cols = [c.strip() for c in table_lines[0].strip("|").split("|")]

    # Find data rows (skip delimiter row |---|---|)
    data_rows = []
    for l in table_lines[1:]:
        if re.match(r"^\|(\s*:?-+:?\s*\|)+$", l):
            continue  # Separator row
        cols = [c.strip() for c in l.strip("|").split("|")]
        data_rows.append(cols)

    if not data_rows:
        return text

    spoken_sentences = []
    for row in data_rows:
        row_phrases = []
        for i, val in enumerate(row):
            if not val or val == "-":
                continue
            hdr = header_cols[i] if i < len(header_cols) and header_cols[i] else f"Column {i+1}"
            row_phrases.append(f"{hdr}: {val}")
        if row_phrases:
            spoken_sentences.append(", ".join(row_phrases) + ".")

    if spoken_sentences:
        return " ".join(spoken_sentences)
    return text


def clean_for_speech(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Generates natural text-to-speech text completely free of brackets, pipes, hashes, or code.
    Replaces currency symbols with verbal equivalents.
    """
    if not text:
        return ""

    s = text

    # If it's a table chunk, convert to conversational sentences first
    if "|" in s and ("\n" in s or s.count("|") >= 4):
        s = convert_table_to_spoken(s)

    # 1. Remove all bracketed metadata tags e.g. [Document: ...], [Section: ...]
    s = re.sub(r"\[.*?\]", " ", s)

    # 2. Verbalize currencies (ensure trailing punctuation like comma is not captured inside number)
    s = re.sub(r"₹\s*(\d(?:[\d,]*\d)?(?:\.\d+)?)", r"\1 rupees", s)
    s = re.sub(r"\$\s*(\d(?:[\d,]*\d)?(?:\.\d+)?)", r"\1 dollars", s)
    s = re.sub(r"€\s*(\d(?:[\d,]*\d)?(?:\.\d+)?)", r"\1 euros", s)
    s = re.sub(r"£\s*(\d(?:[\d,]*\d)?(?:\.\d+)?)", r"\1 pounds", s)

    # Replace standalone currency symbols
    s = s.replace("₹", " rupees ")
    s = s.replace("$", " dollars ")
    s = s.replace("€", " euros ")
    s = s.replace("£", " pounds ")
    s = s.replace("/mo", " per month")
    s = s.replace("/yr", " per year")

    # 3. Strip markdown syntax: headers (#), bold (**), italic (*), code (`)
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", s)
    s = re.sub(r"`{1,3}(.*?)(`{1,3}|$)", r"\1", s)
    s = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", s)

    # 4. Strip markdown links [text](url) -> text, and bare URLs
    s = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", s)
    s = re.sub(r"https?://\S+|www\.\S+", "", s)

    # 5. Clean bullet points (- , * , •, 1. ) to natural speech
    s = re.sub(r"^\s*[-*•]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)

    # 6. Clean markdown table residual pipes
    s = s.replace("|", " ")

    # 7. Normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    s = s.strip()

    # Ensure ending punctuation for TTS cadence
    if s and not s[-1] in ".?!":
        s += "."

    return s


def calculate_speech_seconds(text: str, words_per_minute: float = 150.0) -> float:
    """
    Calculates estimated acoustic read-aloud time at 150 words per minute.
    Formula: (words / 150) * 60.
    150 words = 60.0s.
    """
    if not text:
        return 0.0
    words = len(text.split())
    return (words / words_per_minute) * 60.0
