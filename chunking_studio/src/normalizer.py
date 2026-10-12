import os
import re
import time
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()


class GeminiDocumentNormalizer:
    """
    STAGE 1: LLM Normalizer & Structural Transcriber powered by Google Gemini API.
    
    Transforms raw messy, multilingual, or scrambled text into clean, structured Markdown:
    - Tables -> Standard Markdown tables (| Col | Col |) with headers
    - Clinic/Store Branches -> '### Branch: [Name]' entity cards
    - Q&A pairs -> '### Question: ... \nAnswer: ...'
    - Narrative prose -> Clean paragraph prose
    
    CRITICAL CONSTRAINT: ZERO-REWRITE RULE
    - Strict verbatim extraction.
    - Zero price/number rounding.
    - Zero omitted procedures, phone numbers, or warranty terms.
    """

    SYSTEM_INSTRUCTION = """You are a specialized Voice AI Structural Document Transcriber and Normalizer.
Your sole job is to ingest raw, messy, scanned, multilingual, or scrambled client documents and normalize them into clean, structured Markdown for an automated Voice AI chunking engine.

STRICT RULES & CONSTRAINTS:
1. ZERO-REWRITE CONSTRAINT: You must extract and preserve all numbers, pricing, currency symbols (₹, $, €, £), percentages, phone numbers, addresses, and medical/technical terms VERBATIM. DO NOT round numbers, DO NOT invent facts, and DO NOT summarize or omit pricing/features.
2. RECONSTRUCT SCRAMBLED TABLES: If you detect pricing tables, fees, procedures, or comparisons, format them as standard Markdown tables with explicit column headers and delimiter rows:
| Column 1 | Column 2 | Column 3 |
|---|---|---|
| Value 1 | Value 2 | Value 3 |
3. UNIFY BRANCH DIRECTORIES: If you detect multiple clinic, store, or office locations, format each location as an indivisible card:
### Branch: [Branch Name]
Address: [Full Address]
Landmarks: [Landmarks if present]
Phone: [Phone Number]
Hours: [Operating Hours]
4. UNIFY REAL ESTATE & PROPERTY LISTINGS: If you detect real estate projects, plots, housing layouts, or property listings (e.g. '1.ஸ்ரீராம்கார்டன்', '2.பிரம்மா பார்க்'), format each property as an indivisible card:
### Property: [Property/Project Name]
Guideline Value: [Value]
Main Content: [Description / Overview]
Location: [Directions / Landmarks]
Facilities: [Community Facilities]
Approval: [Approvals]
Nearby Locations: [Distances / Times]
Construction Details: [BHK Options & Pricing]
CRITICAL: All child attributes (Location directions, Facilities, Approvals, BHK pricing) must remain strictly nested inside that property card. Separate multiple properties with '---'.
5. UNIFY FAQS: If you detect questions and answers, format each pair as:
### Question: [Exact Question]
Answer: [Complete, unshortened answer]
6. POLICIES & PROSE: Format general background, policies, or instructions as clean markdown paragraphs with headings.
7. Return ONLY the clean, structured Markdown output. Do not wrap in markdown code blocks like ```markdown ... ``` and do not include commentary or chit-chat."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3-flash-preview"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or "gemini-3-flash-preview"
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initializes the Gemini client if API key is present."""
        if not self.api_key:
            return

        # Attempt to initialize using google-genai or google-generativeai
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._sdk_type = "google-genai"
        except Exception:
            try:
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self._client = gai.GenerativeModel(self.model_name)
                self._sdk_type = "google-generativeai"
            except Exception as e:
                self._client = None
                self._sdk_type = None

    def is_available(self) -> bool:
        """Returns True if Gemini API client is configured."""
        return self._client is not None or bool(self.api_key)

    @staticmethod
    def _is_busy_error(err_str: str) -> bool:
        """Checks if the error indicates temporary server busy / rate limit / 503."""
        indicators = [
            "503", "unavailable", "high demand", "resource_exhausted",
            "429", "rate limit", "overloaded", "server is busy",
            "spikes in demand", "temporarily unavailable"
        ]
        low = err_str.lower()
        return any(ind in low for ind in indicators)

    def normalize(self, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribes and restructures raw document text into clean Markdown.
        Retries up to 5 times with 2-second delay if server busy errors (503/high demand) occur.
        Returns: (normalized_markdown, metadata_stats)
        """
        if not raw_text or not raw_text.strip():
            return "", {"status": "empty", "model": self.model_name}

        if not self.api_key:
            return raw_text, {
                "status": "bypassed_no_key",
                "message": "Gemini API Key not configured; passed verbatim through Stage 1."
            }

        prompt = f"Please normalize and structurally transcribe the following document into clean Markdown adhering to all rules:\n\n{raw_text}"

        candidate_models = [self.model_name]
        for alt in ["gemini-3-flash-preview", "gemini-flash-lite-latest", "gemini-flash-latest"]:
            if alt not in candidate_models:
                candidate_models.append(alt)

        last_error = None
        was_server_busy = False

        for m in candidate_models:
            # Try up to 5 times with 2-second wait when server is busy
            for attempt in range(1, 6):
                try:
                    # 1. New google-genai SDK
                    if hasattr(self, "_sdk_type") and self._sdk_type == "google-genai":
                        from google import genai
                        from google.genai import types

                        client = self._client or genai.Client(api_key=self.api_key)
                        response = client.models.generate_content(
                            model=m,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=self.SYSTEM_INSTRUCTION,
                                temperature=0.1,  # Low temperature for strict adherence & zero hallucination
                            )
                        )
                        output_text = response.text.strip()
                        output_text = self._strip_outer_code_fence(output_text)
                        return output_text, {
                            "status": "success",
                            "model": m,
                            "sdk": "google-genai",
                            "attempt": attempt,
                            "input_len": len(raw_text),
                            "output_len": len(output_text)
                        }

                    # 2. Legacy google-generativeai SDK fallback
                    import google.generativeai as gai
                    gai.configure(api_key=self.api_key)
                    model = gai.GenerativeModel(
                        model_name=m,
                        system_instruction=self.SYSTEM_INSTRUCTION
                    )
                    response = model.generate_content(
                        prompt,
                        generation_config=gai.GenerationConfig(temperature=0.1)
                    )
                    output_text = response.text.strip()
                    output_text = self._strip_outer_code_fence(output_text)
                    return output_text, {
                        "status": "success",
                        "model": m,
                        "sdk": "google-generativeai",
                        "attempt": attempt,
                        "input_len": len(raw_text),
                        "output_len": len(output_text)
                    }

                except Exception as e:
                    err_str = str(e)
                    last_error = err_str

                    if self._is_busy_error(err_str):
                        was_server_busy = True
                        if attempt < 5:
                            print(f"[GeminiNormalizer] Model {m} server busy (Attempt {attempt}/5). Waiting 2s before retry...")
                            time.sleep(2)
                            continue
                        else:
                            print(f"[GeminiNormalizer] Model {m} still busy after 5 attempts.")
                            break
                    else:
                        # Non-busy error (e.g. 404 model not found), switch to candidate model
                        break

        # If server was busy across all attempts
        if was_server_busy:
            return raw_text, {
                "status": "server_busy",
                "error": "Server is busy: LLM endpoint is experiencing high demand (503). Retried 5 times with 2-second intervals. Please try again in a few moments.",
                "message": f"Server busy error after 5 attempts: {last_error}"
            }

        return raw_text, {
            "status": "error",
            "error": str(last_error),
            "message": f"Gemini API normalization failed across candidate models: {str(last_error)}. Proceeding with deterministic stage."
        }

    @staticmethod
    def _strip_outer_code_fence(text: str) -> str:
        """Strips accidental ```markdown ... ``` fences from LLM responses."""
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text
