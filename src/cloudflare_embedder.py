"""
High-Performance Cloudflare Workers AI Embedder Client with Connection Pooling & Warmup.
"""

import os
import time
import hashlib
from typing import List, Dict, Any, Tuple, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np

from src.cloudflare_config import (
    ALL_CLOUDFLARE_MODELS,
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    AVAILABLE_MODEL_IDS
)


class CloudflareEmbedder:
    """
    Client for generating embeddings using Cloudflare Workers AI REST API.
    Utilizes HTTP persistent connection pooling and supports latency benchmarking.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: float = 30.0,
        enable_mock_fallback: bool = True
    ):
        self.account_id = (account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID") or CLOUDFLARE_ACCOUNT_ID).strip()
        self.api_token = (api_token or os.getenv("CLOUDFLARE_API_TOKEN") or CLOUDFLARE_API_TOKEN).strip()
        self.timeout = timeout
        self.enable_mock_fallback = enable_mock_fallback
        self.last_fetch_meta: Dict[str, Any] = {}
        self._last_used_mock = False

        # Build persistent connection pool
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(
            pool_connections=16,
            pool_maxsize=32,
            max_retries=retries
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def is_configured(self) -> bool:
        """Returns True if valid Cloudflare credentials are provided."""
        return bool(self.account_id and self.api_token)

    def get_api_url(self, model_id: str) -> str:
        """Construct the REST inference endpoint for a model."""
        clean_model = model_id.strip()
        return f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{clean_model}"

    def warmup(self, model_id: str) -> float:
        """
        Executes a warmup ping to pre-establish the TLS connection and wake up the model container.
        Returns warmup latency in ms.
        """
        if not self.is_configured():
            return 0.5  # Simulated local warmup

        t0 = time.perf_counter()
        try:
            _, _ = self.embed_texts(["warmup connection"], model_id=model_id, batch_size=1, is_warmup=True)
            return (time.perf_counter() - t0) * 1000.0
        except Exception:
            return (time.perf_counter() - t0) * 1000.0

    def warmup_all(self, model_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Pings all models in parallel or series to prime the connection pool."""
        m_list = model_ids or AVAILABLE_MODEL_IDS
        results = {}
        for mid in m_list:
            results[mid] = self.warmup(mid)
        return results

    def embed_texts(
        self,
        texts: List[str],
        model_id: str,
        batch_size: int = 16,
        is_warmup: bool = False
    ) -> Tuple[List[List[float]], float]:
        """
        Generate embeddings for a list of texts using Cloudflare Workers AI.
        Returns:
            Tuple of (embeddings_list, total_api_latency_ms)
        """
        if not texts:
            return [], 0.0

        if "gemini" in model_id.lower():
            return self._embed_with_gemini(texts, model_id)

        model_cfg = ALL_CLOUDFLARE_MODELS.get(model_id)
        target_dims = model_cfg.dimensions if model_cfg else 1024

        # If credentials are not configured, fallback to deterministic simulation
        if not self.is_configured():
            if self.enable_mock_fallback:
                self._last_used_mock = True
                vectors, sim_ms = self._generate_synthetic_embeddings(texts, target_dims)
                self.last_fetch_meta[model_id] = {
                    "source": "offline_simulation",
                    "is_mock": True,
                    "status_code": 200,
                    "dimensions": target_dims,
                    "vector_sample": [round(x, 4) for x in vectors[0][:5]] if vectors else [],
                    "norm": 1.0
                }
                return vectors, sim_ms
            else:
                raise ValueError("Cloudflare Account ID and API Token must be configured.")

        self._last_used_mock = False
        all_embeddings: List[List[float]] = []
        total_api_time_ms = 0.0

        url = self.get_api_url(model_id)
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"text": batch}

            start_t = time.perf_counter()
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                batch_latency_ms = (time.perf_counter() - start_t) * 1000.0
                total_api_time_ms += batch_latency_ms

                if response.status_code == 200:
                    res_json = response.json()
                    batch_vectors = self._extract_vectors_from_response(res_json, len(batch), target_dims)
                    all_embeddings.extend(batch_vectors)
                    self.last_fetch_meta[model_id] = {
                        "source": "live_cloudflare_api",
                        "is_mock": False,
                        "status_code": 200,
                        "dimensions": len(batch_vectors[0]) if batch_vectors else target_dims,
                        "vector_sample": [round(x, 4) for x in batch_vectors[0][:5]] if batch_vectors else [],
                        "norm": round(float(np.linalg.norm(batch_vectors[0])), 4) if batch_vectors else 0.0
                    }
                else:
                    error_msg = f"Cloudflare API returned status {response.status_code}: {response.text}"
                    if self.enable_mock_fallback and not is_warmup:
                        self._last_used_mock = True
                        vectors, _ = self._generate_synthetic_embeddings(batch, target_dims)
                        all_embeddings.extend(vectors)
                        self.last_fetch_meta[model_id] = {
                            "source": "fallback_simulation",
                            "is_mock": True,
                            "status_code": response.status_code,
                            "error": response.text[:200],
                            "dimensions": target_dims,
                            "vector_sample": [round(x, 4) for x in vectors[0][:5]] if vectors else [],
                            "norm": 1.0
                        }
                    else:
                        raise RuntimeError(error_msg)

            except Exception as ex:
                if self.enable_mock_fallback and not is_warmup:
                    self._last_used_mock = True
                    vectors, _ = self._generate_synthetic_embeddings(batch, target_dims)
                    all_embeddings.extend(vectors)
                    total_api_time_ms += 15.0
                    self.last_fetch_meta[model_id] = {
                        "source": "fallback_simulation",
                        "is_mock": True,
                        "error": str(ex),
                        "dimensions": target_dims,
                        "vector_sample": [round(x, 4) for x in vectors[0][:5]] if vectors else [],
                        "norm": 1.0
                    }
                else:
                    raise RuntimeError(f"Error calling Cloudflare Workers AI for model '{model_id}': {ex}")

        return all_embeddings, total_api_time_ms

    def embed_single_query(self, query: str, model_id: str) -> Tuple[List[float], float]:
        """
        Generate embedding for a single question or search query.
        Returns:
            Tuple of (vector, api_latency_ms)
        """
        vectors, latency_ms = self.embed_texts([query], model_id=model_id, batch_size=1)
        if not vectors:
            raise RuntimeError(f"Failed to obtain embedding vector for query with model {model_id}")
        return vectors[0], latency_ms

    def _extract_vectors_from_response(
        self,
        res_json: Dict[str, Any],
        expected_count: int,
        target_dims: int
    ) -> List[List[float]]:
        """Parse vector arrays from Cloudflare Workers AI response formats."""
        result = res_json.get("result", {})
        data = result.get("data", [])
        if not data and "data" in res_json:
            data = res_json.get("data", [])

        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                return [[float(x) for x in v] for v in data]
            elif isinstance(data[0], dict) and "embedding" in data[0]:
                return [[float(x) for x in item["embedding"]] for item in data]
            elif isinstance(data[0], (int, float)):
                return [[float(x) for x in data]]

        raise ValueError(f"Could not parse valid vector data from response: {res_json}")

    def _generate_synthetic_embeddings(
        self,
        texts: List[str],
        dimension: int
    ) -> Tuple[List[List[float]], float]:
        """
        Generate deterministic unit-normalized pseudo-embeddings for testing
        when credentials are not yet set or in offline simulation mode.
        """
        start_t = time.perf_counter()
        embeddings: List[List[float]] = []

        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.RandomState(seed)
            raw_vec = rng.randn(dimension).astype(np.float32)
            norm = np.linalg.norm(raw_vec)
            if norm > 0:
                raw_vec = raw_vec / norm
            embeddings.append(raw_vec.tolist())

        sim_latency_ms = (time.perf_counter() - start_t) * 1000.0 + 12.0
        return embeddings, sim_latency_ms

    def _embed_with_gemini(self, texts: List[str], model_id: str) -> Tuple[List[List[float]], float]:
        """Generate high-precision embeddings using Google Gemini API."""
        t0 = time.perf_counter()
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured in .env")

        from google import genai
        client = genai.Client(api_key=gemini_api_key)

        vectors = []
        for t in texts:
            r = client.models.embed_content(model="gemini-embedding-001", contents=t)
            v = np.array(r.embeddings[0].values, dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors.append(v.tolist())

        lat_ms = (time.perf_counter() - t0) * 1000.0
        self.last_fetch_meta[model_id] = {
            "source": "live_gemini_api",
            "is_mock": False,
            "status_code": 200,
            "dimensions": 3072,
            "vector_sample": [round(x, 4) for x in vectors[0][:5]] if vectors else [],
            "norm": 1.0
        }
        return vectors, lat_ms
