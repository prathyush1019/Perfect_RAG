"""
Local On-Disk Qdrant Vector Storage Manager with Multi-Document Filtering & HNSW Search.
"""

import os
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.cloudflare_config import (
    ALL_CLOUDFLARE_MODELS,
    DEFAULT_QDRANT_STORAGE_PATH,
    EmbeddingModelConfig
)


from src.sparse_encoder import compute_sparse_vector


class LocalQdrantManager:
    """
    Manages local on-disk Qdrant storage with dedicated collections
    for each Cloudflare Workers AI embedding model and supports document filtering.
    Implements:
      - Fix #1: Exact flat scan (exact: True) for 100% recall
      - Fix #3: Native Qdrant Hybrid Search with Reciprocal Rank Fusion (RRF)
      - Fix #4: High-precision token/entity price-boosted lexical matching
    """

    def __init__(self, storage_path: str = DEFAULT_QDRANT_STORAGE_PATH):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        try:
            self.client = QdrantClient(path=self.storage_path)
        except Exception as e:
            if "already accessed" in str(e).lower() or "permission denied" in str(e).lower() or "locked" in str(e).lower():
                print(f"\n[!] Qdrant Lock Error: Storage directory '{self.storage_path}' is currently locked by another active server process.")
                print("[!] Please make sure no other instance of server.py or python is running.\n")
            raise e
        self._ensure_all_collections()

    def close(self):
        """Cleanly releases locks and closes local storage."""
        try:
            if hasattr(self, "client") and self.client is not None:
                self.client.close()
        except Exception:
            pass

    def _ensure_all_collections(self):
        """Initializes collections for all Cloudflare models."""
        for model_id, model_cfg in ALL_CLOUDFLARE_MODELS.items():
            self.ensure_collection(model_cfg)

    def ensure_collection(self, model_cfg: EmbeddingModelConfig) -> bool:
        """
        Ensure collection exists with hybrid schema:
        - Named dense vector with Cosine metric
        - Named sparse vector for exact lexical/number/price matching
        """
        col_name = model_cfg.collection_name
        try:
            if self.client.collection_exists(collection_name=col_name):
                info = self.client.get_collection(collection_name=col_name)
                vectors_param = info.config.params.vectors
                is_valid_hybrid = (
                    isinstance(vectors_param, dict)
                    and "dense" in vectors_param
                    and vectors_param["dense"].size == model_cfg.dimensions
                    and info.config.params.sparse_vectors is not None
                    and "sparse" in info.config.params.sparse_vectors
                )
                if not is_valid_hybrid:
                    # Upgrade legacy schema to native Hybrid schema (dense + sparse)
                    self.client.delete_collection(collection_name=col_name)
                    self.client.create_collection(
                        collection_name=col_name,
                        vectors_config={
                            "dense": qmodels.VectorParams(
                                size=model_cfg.dimensions,
                                distance=qmodels.Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "sparse": qmodels.SparseVectorParams()
                        }
                    )
                    return True
                return False
            else:
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config={
                        "dense": qmodels.VectorParams(
                            size=model_cfg.dimensions,
                            distance=qmodels.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": qmodels.SparseVectorParams()
                    }
                )
                return True
        except Exception as e:
            print(f"Error ensuring collection {col_name}: {e}")
            return False

    def upsert_document_chunks(
        self,
        doc_id: str,
        doc_name: str,
        chunks: List[Dict[str, Any]],
        embeddings_by_model: Dict[str, List[List[float]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Upserts embedded chunks into each model's collection with dual vectors:
        1. Dense semantic vector
        2. Sparse lexical vector (BM25 term hashing with exact price/number boost)
        """
        results = {}
        for model_id, vectors in embeddings_by_model.items():
            model_cfg = ALL_CLOUDFLARE_MODELS.get(model_id)
            if not model_cfg or len(vectors) != len(chunks):
                continue

            self.ensure_collection(model_cfg)
            col_name = model_cfg.collection_name

            points = []
            for chunk, vec in zip(chunks, vectors):
                if isinstance(chunk, dict):
                    chunk_id = chunk.get("chunk_id", 1)
                    raw_text = chunk.get("text") or chunk.get("clean_text") or chunk.get("raw_text", "")
                    meta_dict = chunk.get("metadata", {})
                    context_grounding = chunk.get("context_grounding") or meta_dict.get("context_grounding", "")
                else:
                    chunk_id = getattr(chunk, "chunk_id", 1)
                    raw_text = getattr(chunk, "text", "") or getattr(chunk, "raw_text", "")
                    meta_dict = getattr(chunk, "metadata", {})
                    context_grounding = getattr(chunk, "context_grounding", "") or meta_dict.get("context_grounding", "")

                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{model_id}_{chunk_id}"))
                sparse_vec = compute_sparse_vector(raw_text)

                payload = {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_id": chunk_id,
                    "text": raw_text,
                    "context_grounding": context_grounding,
                    "char_count": len(raw_text),
                    "metadata": meta_dict,
                    "model_id": model_id,
                    "timestamp": time.time()
                }

                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector={
                            "dense": vec,
                            "sparse": sparse_vec
                        },
                        payload=payload
                    )
                )

            t0 = time.perf_counter()
            self.client.upsert(
                collection_name=col_name,
                points=points,
                wait=True
            )
            upsert_ms = (time.perf_counter() - t0) * 1000.0

            results[model_id] = {
                "upserted_count": len(points),
                "latency_ms": round(upsert_ms, 2)
            }

        return results

    def search(
        self,
        query_vector: List[float],
        model_id: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 4,
        raw_query: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Performs vector similarity search against the model's collection.
        Features:
          - Fix #1: Exact Flat Scan (exact: True) to eliminate HNSW graph approximation error
          - Fix #3: Native Qdrant RRF Fusion combining Dense semantic + Sparse lexical/price matching
          - Document filtering when doc_ids are provided
        """
        model_cfg = ALL_CLOUDFLARE_MODELS.get(model_id)
        if not model_cfg:
            raise ValueError(f"Unknown model_id: {model_id}")

        col_name = model_cfg.collection_name
        self.ensure_collection(model_cfg)

        # Build document filter if doc_ids are specified
        query_filter = None
        if doc_ids and len(doc_ids) > 0:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id",
                        match=qmodels.MatchAny(any=doc_ids)
                    )
                ]
            )

        t0 = time.perf_counter()
        try:
            info = self.client.get_collection(collection_name=col_name)
            is_hybrid = (
                info.config.params.sparse_vectors is not None
                and "sparse" in info.config.params.sparse_vectors
            )

            if is_hybrid and raw_query:
                # Fix #3: Native Qdrant Reciprocal Rank Fusion (RRF)
                sparse_query_vec = compute_sparse_vector(raw_query)
                prefetch = [
                    qmodels.Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=top_k * 3,
                        filter=query_filter,
                        params=qmodels.SearchParams(exact=True)  # Fix #1: Disable Approximation ➔ exact: True
                    ),
                    qmodels.Prefetch(
                        query=sparse_query_vec,
                        using="sparse",
                        limit=top_k * 3,
                        filter=query_filter
                    )
                ]
                res = self.client.query_points(
                    collection_name=col_name,
                    prefetch=prefetch,
                    query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                    search_params=qmodels.SearchParams(exact=True),
                    limit=top_k
                )
            else:
                # Fallback to exact flat scan dense search
                res = self.client.query_points(
                    collection_name=col_name,
                    query=query_vector,
                    using="dense" if isinstance(info.config.params.vectors, dict) else None,
                    query_filter=query_filter,
                    search_params=qmodels.SearchParams(exact=True),  # Fix #1: exact: True
                    limit=top_k
                )

            search_ms = (time.perf_counter() - t0) * 1000.0

            retrieved = []
            for p in res.points:
                payload = p.payload or {}
                retrieved.append({
                    "score": round(float(p.score), 4),
                    "doc_id": payload.get("doc_id", ""),
                    "doc_name": payload.get("doc_name", ""),
                    "chunk_id": payload.get("chunk_id", 0),
                    "chunk_type": payload.get("chunk_type", "text"),
                    "text": payload.get("text", ""),
                    "char_count": payload.get("char_count", 0),
                    "metadata": payload.get("metadata", {}),
                    "model_id": model_id
                })

            return retrieved, search_ms
        except Exception as ex:
            search_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Error searching Qdrant collection {col_name}: {ex}")
            return [], search_ms

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Returns all distinct documents stored in Qdrant across collections.
        """
        doc_map = {}
        # Sample points from primary flagship collection
        primary_col = ALL_CLOUDFLARE_MODELS["@cf/baai/bge-m3"].collection_name
        try:
            if self.client.collection_exists(primary_col):
                points, _ = self.client.scroll(
                    collection_name=primary_col,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False
                )
                for pt in points:
                    payload = pt.payload or {}
                    d_id = payload.get("doc_id")
                    if d_id and d_id not in doc_map:
                        doc_map[d_id] = {
                            "doc_id": d_id,
                            "doc_name": payload.get("doc_name", d_id),
                            "chunk_count": 0,
                            "timestamp": payload.get("timestamp", 0)
                        }
                    if d_id in doc_map:
                        doc_map[d_id]["chunk_count"] += 1
        except Exception as e:
            print(f"Error listing documents: {e}")

        return list(doc_map.values())

    def delete_document(self, doc_id: str) -> bool:
        """Deletes all chunks for a given document across all model collections."""
        deleted = False
        filt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_id",
                    match=qmodels.MatchValue(value=doc_id)
                )
            ]
        )
        for model_cfg in ALL_CLOUDFLARE_MODELS.values():
            col_name = model_cfg.collection_name
            try:
                if self.client.collection_exists(col_name):
                    self.client.delete(
                        collection_name=col_name,
                        points_selector=filt
                    )
                    deleted = True
            except Exception:
                pass
        return deleted

    def clear_all_documents(self) -> bool:
        """
        Deletes all stored points and vectors across all Cloudflare collections,
        completely purging all indexed documents from Qdrant.
        """
        for model_id, model_cfg in ALL_CLOUDFLARE_MODELS.items():
            col_name = model_cfg.collection_name
            try:
                if self.client.collection_exists(col_name):
                    self.client.delete(
                        collection_name=col_name,
                        points_selector=qmodels.FilterSelector(filter=qmodels.Filter()),
                        wait=True
                    )
            except Exception as e:
                print(f"Error clearing points in collection {col_name}: {e}")
        return True

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Returns statistics for each Cloudflare model collection."""
        stats = {}
        for mid, model_cfg in ALL_CLOUDFLARE_MODELS.items():
            col_name = model_cfg.collection_name
            try:
                if self.client.collection_exists(col_name):
                    info = self.client.get_collection(col_name)
                    stats[mid] = {
                        "name": col_name,
                        "display_name": model_cfg.display_name,
                        "dimensions": model_cfg.dimensions,
                        "points_count": info.points_count or 0,
                        "is_multilingual": model_cfg.is_multilingual,
                        "badge_color": model_cfg.badge_color
                    }
                else:
                    stats[mid] = {
                        "name": col_name,
                        "display_name": model_cfg.display_name,
                        "dimensions": model_cfg.dimensions,
                        "points_count": 0,
                        "is_multilingual": model_cfg.is_multilingual,
                        "badge_color": model_cfg.badge_color
                    }
            except Exception:
                stats[mid] = {
                    "name": col_name,
                    "display_name": model_cfg.display_name,
                    "dimensions": model_cfg.dimensions,
                    "points_count": 0,
                    "is_multilingual": model_cfg.is_multilingual,
                    "badge_color": model_cfg.badge_color
                }
        return stats
