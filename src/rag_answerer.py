"""
RAG Answer Engine: In-takes retrieved vector/hybrid chunks from Qdrant
and synthesizes precise, grounded, hallucination-free answers using fast LLMs.
Supports Cloudflare Workers AI (Llama 3.1 8B, Llama 3.3 70B, Mistral 7B)
and Google Gemini API.
"""

import os
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()


LLM_MODELS = [
    {
        "id": "@cf/meta/llama-3.1-8b-instruct",
        "name": "Meta Llama 3.1 8B Instruct (Cloudflare AI)",
        "badge": "⚡ Ultra-Fast (~800ms)",
        "provider": "cloudflare",
        "default": True
    },
    {
        "id": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "name": "Meta Llama 3.3 70B Instruct (Cloudflare AI)",
        "badge": "🧠 Deep Reasoning (70B)",
        "provider": "cloudflare",
        "default": False
    },
    {
        "id": "@cf/mistral/mistral-7b-instruct-v0.1",
        "name": "Mistral 7B Instruct (Cloudflare AI)",
        "badge": "🌪️ Mistral 7B",
        "provider": "cloudflare",
        "default": False
    },
    {
        "id": "gemini-3.8-flash",
        "name": "Google Gemini 3.8 Flash",
        "badge": "✨ Google Gemini",
        "provider": "gemini",
        "default": False
    }
]


class RAGAnswerEngine:
    """
    Synthesizes natural language answers to user queries grounded strictly
    in retrieved document chunks from the Qdrant hybrid vector store.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        self.account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.api_token = api_token or os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()

    def format_context_blocks(self, chunks: List[Dict[str, Any]], max_chunks: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Formats top retrieved chunks into structured context blocks with
        verbatim text, source document tags, and section breadcrumbs.
        """
        if not chunks:
            return "", []

        selected = chunks[:max_chunks]
        context_parts = []
        citations = []

        for idx, c in enumerate(selected, 1):
            doc_name = c.get("doc_name") or c.get("metadata", {}).get("doc_title", "Document")
            chunk_id = c.get("chunk_id", idx)
            score = c.get("score", 0.0)
            score_pct = f"{round(score * 100, 1)}%" if score else "N/A"
            grounding = c.get("context_grounding") or c.get("metadata", {}).get("context_grounding", "")
            raw_text = (c.get("text") or c.get("raw_text") or "").strip()

            header = f"[Source #{idx} | Doc: {doc_name} | Chunk #{chunk_id} | Similarity: {score_pct}]"
            if grounding:
                header += f"\n{grounding}"

            context_parts.append(f"{header}\n{raw_text}")
            citations.append({
                "source_index": idx,
                "doc_name": doc_name,
                "chunk_id": chunk_id,
                "score": score,
                "context_grounding": grounding,
                "preview": raw_text[:120] + "..." if len(raw_text) > 120 else raw_text
            })

        formatted_context = "\n\n---\n\n".join(context_parts)
        return formatted_context, citations

    def answer_question(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model_id: str = "@cf/meta/llama-3.1-8b-instruct",
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generates a direct, grounded answer to the query using the provided chunks.
        """
        query = (query or "").strip()
        if not query:
            return {
                "status": "error",
                "error": "Query cannot be empty",
                "answer": "Please provide a valid question to answer."
            }

        if not chunks:
            return {
                "status": "empty_context",
                "answer": "No relevant document chunks were found in the knowledge base to answer this question.",
                "model_id": model_id,
                "citations": [],
                "latency_ms": 0.0
            }

        context_text, citations = self.format_context_blocks(chunks, max_chunks=5)

        system_prompt = (
            "You are a factual, concise, and helpful AI assistant specialized in document-grounded question answering.\n"
            "Your objective is to answer the user's question accurately based SOLELY on the provided retrieved context chunks.\n\n"
            "Strict Guidelines:\n"
            "1. Rely only on clear facts stated directly in the retrieved context. Do not speculate, extrapolate, or bring in outside knowledge.\n"
            "2. If the context does not contain enough information to answer the question, clearly state: "
            "'The provided document excerpts do not contain sufficient information to answer this question.'\n"
            "3. Keep your explanation structured, crisp, and direct. Use bullet points for lists, specifications, or pricing where appropriate.\n"
            "4. When mentioning facts, cite the source number (e.g., [Source #1]) or section name."
        )

        user_content = (
            f"Retrieved Document Context Chunks:\n\n"
            f"{context_text}\n\n"
            f"==================================================\n"
            f"User Question: {query}\n\n"
            f"Grounded Answer:"
        )

        # Dispatch to provider
        if model_id.startswith("@cf/") or "llama" in model_id.lower() or "mistral" in model_id.lower():
            return self._call_cloudflare(model_id, system_prompt, user_content, citations, max_tokens, temperature)
        elif "gemini" in model_id.lower():
            return self._call_gemini(model_id, system_prompt, user_content, citations)
        else:
            # Fallback to default Cloudflare Llama 3.1 8B
            return self._call_cloudflare("@cf/meta/llama-3.1-8b-instruct", system_prompt, user_content, citations, max_tokens, temperature)

    def _call_cloudflare(
        self,
        model_id: str,
        system_prompt: str,
        user_content: str,
        citations: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Calls Cloudflare Workers AI Text Generation API."""
        if not self.account_id or not self.api_token:
            return {
                "status": "error",
                "error": "Cloudflare Account ID or API Token missing in .env",
                "answer": "Unable to generate AI answer: Cloudflare API credentials are not configured.",
                "model_id": model_id,
                "citations": citations,
                "latency_ms": 0.0
            }

        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{model_id}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            latency_ms = round((time.time() - t0) * 1000, 1)

            if resp.status_code == 200:
                data = resp.json()
                result = data.get("result", {})
                # Some CF models return 'response', others return OpenAI-compatible 'choices'
                answer_text = (
                    result.get("response") or
                    (result.get("choices") and result["choices"][0].get("message", {}).get("content")) or
                    ""
                ).strip()

                if not answer_text and "choices" in data:
                    answer_text = data["choices"][0].get("message", {}).get("content", "").strip()

                return {
                    "status": "success",
                    "answer": answer_text or "No text generated by model.",
                    "model_id": model_id,
                    "provider": "Cloudflare Workers AI",
                    "latency_ms": latency_ms,
                    "citations": citations,
                    "tokens_used": result.get("usage", {}).get("total_tokens", "N/A")
                }
            else:
                return {
                    "status": "error",
                    "error": f"Cloudflare HTTP {resp.status_code}: {resp.text[:200]}",
                    "answer": f"⚠️ Cloudflare AI returned HTTP {resp.status_code}. Please check credentials or try another model.",
                    "model_id": model_id,
                    "provider": "Cloudflare Workers AI",
                    "latency_ms": latency_ms,
                    "citations": citations
                }
        except Exception as e:
            latency_ms = round((time.time() - t0) * 1000, 1)
            return {
                "status": "error",
                "error": str(e),
                "answer": f"Error communicating with Cloudflare AI: {str(e)}",
                "model_id": model_id,
                "provider": "Cloudflare Workers AI",
                "latency_ms": latency_ms,
                "citations": citations
            }

    def _call_gemini(
        self,
        model_id: str,
        system_prompt: str,
        user_content: str,
        citations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calls Google Gemini API via REST."""
        if not self.gemini_api_key:
            return {
                "status": "error",
                "error": "GEMINI_API_KEY not configured in .env",
                "answer": "Unable to generate AI answer: Google Gemini API Key is not configured.",
                "model_id": model_id,
                "citations": citations,
                "latency_ms": 0.0
            }

        candidate_models = [model_id, "gemini-3.8-flash", "gemini-2.5-flash"]
        t0 = time.time()
        last_error = ""

        for m in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.gemini_api_key}"
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": user_content}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600}
            }
            try:
                r = requests.post(url, json=payload, timeout=20)
                latency_ms = round((time.time() - t0) * 1000, 1)
                if r.status_code == 200:
                    data = r.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts).strip()
                        return {
                            "status": "success",
                            "answer": text,
                            "model_id": m,
                            "provider": "Google Gemini",
                            "latency_ms": latency_ms,
                            "citations": citations
                        }
                last_error = f"HTTP {r.status_code}: {r.text[:120]}"
            except Exception as e:
                last_error = str(e)

        latency_ms = round((time.time() - t0) * 1000, 1)
        # If Gemini is busy (503/404), fall back to Cloudflare Llama 3.1
        fallback_res = self._call_cloudflare("@cf/meta/llama-3.1-8b-instruct", system_prompt, user_content, citations)
        if fallback_res.get("status") == "success":
            fallback_res["fallback_from"] = f"Gemini ({last_error})"
            return fallback_res

        return {
            "status": "error",
            "error": last_error,
            "answer": f"Gemini API unavailable ({last_error}).",
            "model_id": model_id,
            "provider": "Google Gemini",
            "latency_ms": latency_ms,
            "citations": citations
        }
