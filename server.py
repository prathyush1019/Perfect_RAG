import os
import sys
import re
import json
import io
import time
import uuid
import concurrent.futures
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIO_DIR = os.path.join(BASE_DIR, "chunking_studio")
if STUDIO_DIR not in sys.path:
    sys.path.insert(0, STUDIO_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv()

from chunking_studio.src.chunkers import (
    detect_document_strategy,
    get_chunker,
    apply_contextual_chunk_grounding
)
from chunking_studio.src.chunkers.base import count_units, split_text_to_ceiling, ChunkResult
from chunking_studio.src.auditor import VoiceChunkAuditor
from chunking_studio.src.normalizer import GeminiDocumentNormalizer
from chunking_studio.src.document_parsers import parse_uploaded_file

# Cloudflare & Qdrant Managers
from src.cloudflare_config import ALL_CLOUDFLARE_MODELS, AVAILABLE_MODEL_IDS
from src.cloudflare_embedder import CloudflareEmbedder
from src.qdrant_manager import LocalQdrantManager

embedder = CloudflareEmbedder()
qdrant_manager = LocalQdrantManager()


class ChunkingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            html_path = os.path.join(BASE_DIR, "index.html")
            if os.path.exists(html_path):
                with open(html_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        elif parsed.path == "/api/models":
            models_list = [
                {
                    "id": cfg.id,
                    "display_name": cfg.display_name,
                    "dimensions": cfg.dimensions,
                    "languages": cfg.languages,
                    "max_tokens": cfg.max_tokens,
                    "description": cfg.description,
                    "badge_color": cfg.badge_color,
                    "is_multilingual": cfg.is_multilingual
                }
                for cfg in ALL_CLOUDFLARE_MODELS.values()
            ]
            self._send_json({
                "models": models_list,
                "is_configured": embedder.is_configured(),
                "account_id": embedder.account_id
            })
            return

        elif parsed.path == "/api/documents":
            docs = qdrant_manager.list_documents()
            stats = qdrant_manager.get_collection_stats()
            self._send_json({"documents": docs, "stats": stats})
            return

        elif parsed.path == "/api/logs":
            token_display = "Not set"
            if embedder.api_token:
                token_display = f"{embedder.api_token[:7]}...{embedder.api_token[-4:]}"
            elif embedder.global_api_key:
                token_display = f"Global Key ({embedder.auth_email})"

            self._send_json({
                "status": "ok",
                "cloudflare_configured": embedder.is_configured(),
                "account_id": f"{embedder.account_id[:6]}...{embedder.account_id[-4:]}" if len(embedder.account_id) > 10 else embedder.account_id,
                "token_masked": token_display,
                "is_mock_fallback_active": embedder._last_used_mock,
                "logs": embedder.get_recent_logs(limit=100)
            })
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/chunk":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as e:
                self._send_json({"error": "Invalid JSON: " + str(e)}, 400)
                return

            text = payload.get("text", "").strip()
            unit = payload.get("unit", "tokens").lower()
            default_chunk = 512 if unit in ["tokens", "token"] else 2000
            default_overlap = 50 if unit in ["tokens", "token"] else 150

            chunk_size = int(payload.get("chunk_size", default_chunk))
            overlap = int(payload.get("overlap", default_overlap))
            model_name = payload.get("model", "none")

            if not text:
                self._send_json({"error": "Document text cannot be empty"}, 400)
                return

            active_text = text
            llm_info = {"enabled": False, "status": "skipped", "model": model_name}
            # Gemini Stage 1
            if model_name and model_name.lower() != "none" and "gemini" in model_name.lower():
                api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
                normalizer = GeminiDocumentNormalizer(api_key=api_key, model_name=model_name)
                norm_text, norm_meta = normalizer.normalize(active_text)
                llm_info = {
                    "enabled": True,
                    "status": norm_meta.get("status"),
                    "model": norm_meta.get("model", model_name),
                    "error": norm_meta.get("error"),
                    "message": norm_meta.get("message")
                }
                if norm_meta.get("status") == "server_busy":
                    self._send_json({
                        "error": norm_meta.get("error", "Server is busy. Please try again in a few moments."),
                        "llm_info": llm_info
                    }, 503)
                    return
                elif norm_meta.get("status") == "success":
                    active_text = norm_text

            faq_mode = payload.get("faq_mode", "consolidated").lower()
            consolidate_faq = (faq_mode != "atomic")
            doc_name = payload.get("doc_name", "Document").strip() or "Document"
            enable_grounding = payload.get("context_grounding", True)
            if isinstance(enable_grounding, str):
                enable_grounding = enable_grounding.lower() in ["true", "1", "yes", "on"]

            # If Contextual Grounding is enabled, reserve budget for the breadcrumb anchor
            # (avg 40 tokens / 160 chars) so raw chunking leaves room without awkward splits
            grounding_reserve = (40 if unit in ["tokens", "token"] else 160) if enable_grounding else 0
            effective_chunk_size = max(50, chunk_size - grounding_reserve)

            strategy, _ = detect_document_strategy(active_text)
            chunker = get_chunker(
                strategy,
                unit=unit,
                chunk_size=effective_chunk_size,
                overlap=overlap,
                target_words=effective_chunk_size,
                overlap_words=overlap,
                consolidate_faq=consolidate_faq
            )
            metadata = {"doc_title": doc_name, "currency": "INR"}
            raw_chunks = chunker.chunk(active_text, metadata=metadata)

            # Apply Contextual Chunk Grounding (if enabled)
            if enable_grounding:
                raw_chunks = apply_contextual_chunk_grounding(
                    raw_chunks,
                    full_text=active_text,
                    doc_name=doc_name,
                    enabled=True
                )

            # Enforce strict hard ceiling across all chunks on the FINAL text (including anchor)
            final_chunks = []
            curr_id = 1
            for c in raw_chunks:
                c_text = (c.text or c.searchable_augmented_text or c.raw_text or "").strip()
                # Discard degenerate / solitary symbol chunks (e.g., solitary '#')
                if len(re.findall(r"\w", c_text, re.UNICODE)) < 2:
                    continue

                if count_units(c_text, unit) > chunk_size:
                    anchor = c.context_grounding or ""
                    anchor_units = count_units(anchor + "\n", unit) if anchor else 0
                    body_budget = max(20, chunk_size - anchor_units)

                    # Extract raw body by stripping anchor prefix if present
                    raw_body = c.raw_text or c_text
                    if anchor and raw_body.startswith(anchor):
                        raw_body = raw_body[len(anchor):].lstrip("\n")

                    sub_pieces = split_text_to_ceiling(raw_body, body_budget, unit)
                    total_sub = len(sub_pieces)
                    for sub_idx, sub in enumerate(sub_pieces, 1):
                        if len(re.findall(r"\w", sub.strip(), re.UNICODE)) < 2:
                            continue
                        grounded_sub = f"{anchor}\n{sub}" if anchor else sub
                        final_chunks.append(
                            ChunkResult(
                                chunk_id=curr_id,
                                start_char=c.start_char,
                                end_char=c.end_char,
                                text=grounded_sub,
                                raw_text=sub,
                                searchable_augmented_text=grounded_sub,
                                chunk_type=c.chunk_type,
                                context_grounding=anchor,
                                metadata={
                                    **c.metadata,
                                    "sub_chunk": True,
                                    "part": sub_idx,
                                    "total_parts": total_sub,
                                    "hard_ceiling": chunk_size
                                }
                            )
                        )
                        curr_id += 1
                else:
                    c.chunk_id = curr_id
                    c.metadata["hard_ceiling"] = chunk_size
                    final_chunks.append(c)
                    curr_id += 1


            auditor = VoiceChunkAuditor()
            audit_summary = auditor.audit_chunks(final_chunks)

            chunks_data = [c.to_dict() for c in final_chunks]
            self._send_json({
                "chunks": chunks_data,
                "stats": audit_summary,
                "strategy": strategy,
                "llm_info": llm_info,
                "normalized_text": active_text if llm_info.get("status") == "success" else None
            })

        elif parsed.path == "/api/embed_and_save":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as e:
                self._send_json({"error": "Invalid JSON: " + str(e)}, 400)
                return

            doc_name = payload.get("doc_name", "Untitled Document").strip() or "Untitled Document"
            doc_id = payload.get("doc_id") or f"doc_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            chunks = payload.get("chunks", [])
            model_ids = payload.get("model_ids") or AVAILABLE_MODEL_IDS

            if not chunks:
                self._send_json({"error": "No chunks provided to embed"}, 400)
                return

            chunk_texts = [c.get("text") or c.get("clean_text") or c.get("raw_text") or str(c) for c in chunks]

            # If clear_previous is True (default: True to only keep the current document), wipe old vectors
            clear_previous = payload.get("clear_previous", True)
            if clear_previous:
                qdrant_manager.clear_all_documents()

            # Generate embeddings for each model in parallel
            embeddings_by_model = {}
            api_latencies = {}

            def _embed_model(mid):
                vectors, lat_ms = embedder.embed_texts(chunk_texts, model_id=mid)
                return mid, vectors, lat_ms

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(model_ids), 7)) as executor:
                results = list(executor.map(_embed_model, model_ids))

            for mid, vectors, lat_ms in results:
                embeddings_by_model[mid] = vectors
                api_latencies[mid] = round(lat_ms, 2)

            # Upsert into Qdrant collections
            upsert_results = qdrant_manager.upsert_document_chunks(
                doc_id=doc_id,
                doc_name=doc_name,
                chunks=chunks,
                embeddings_by_model=embeddings_by_model
            )

            self._send_json({
                "status": "success",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "chunks_count": len(chunks),
                "api_latencies": api_latencies,
                "upsert_results": upsert_results,
                "models_count": len(embeddings_by_model),
                "recent_logs": embedder.get_recent_logs(limit=30)
            })

        elif parsed.path == "/api/search":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as e:
                self._send_json({"error": "Invalid JSON: " + str(e)}, 400)
                return

            query = payload.get("query", "").strip()
            selected_doc_ids = payload.get("selected_doc_ids") or payload.get("doc_ids") or []

            # Support selecting a single model or multiple models or 'all'
            single_model = payload.get("model_id")
            model_ids = payload.get("model_ids")
            if single_model and single_model.lower() != "all":
                model_ids = [single_model]
            elif not model_ids or (isinstance(model_ids, str) and model_ids.lower() == "all"):
                model_ids = AVAILABLE_MODEL_IDS
            elif isinstance(model_ids, str):
                model_ids = [model_ids]

            # Filter valid models only
            valid_model_ids = [mid for mid in model_ids if mid in ALL_CLOUDFLARE_MODELS]
            if not valid_model_ids:
                valid_model_ids = AVAILABLE_MODEL_IDS

            top_k = int(payload.get("top_k", 4))

            if not query:
                self._send_json({"error": "Query cannot be empty"}, 400)
                return

            # Query search across chosen model(s)
            def _search_single_model(mid):
                m_cfg = ALL_CLOUDFLARE_MODELS.get(mid)
                if not m_cfg:
                    return None
                q_vec, api_lat = embedder.embed_single_query(query, model_id=mid)
                hits, qdrant_lat = qdrant_manager.search(
                    query_vector=q_vec,
                    model_id=mid,
                    doc_ids=selected_doc_ids,
                    top_k=top_k,
                    raw_query=query
                )
                meta = embedder.last_fetch_meta.get(mid, {})
                sc = meta.get("status_code", 200)
                err = meta.get("error")
                ep = embedder.get_api_url(mid) if "gemini" not in mid else "Google GenAI API"
                top_score = hits[0]["score"] if hits else 0.0
                log_line = f"[Search] {m_cfg.display_name} ({mid}) -> Cloudflare: HTTP {sc} ({api_lat:.1f}ms, {len(q_vec)}d) | Qdrant: {len(hits)} hits ({qdrant_lat:.1f}ms, top: {top_score:.4f}) | Filter: {selected_doc_ids}"
                if err:
                    log_line += f" | Error: {err}"
                print(log_line)

                # Record Qdrant search retrieval step in embedder logs
                qdrant_status_msg = f"Retrieved {len(hits)} chunk(s) (Top: {top_score:.4f}) in {qdrant_lat:.1f}ms | Filter: {selected_doc_ids}" if hits else f"0 matching chunks found in Qdrant for filter {selected_doc_ids}"
                level = "SUCCESS" if hits else "WARN"
                embedder._record_log(
                    level=level,
                    model_id=mid,
                    endpoint=f"Qdrant HNSW ('{m_cfg.collection_name}')",
                    status_code=200 if hits else 404,
                    latency_ms=qdrant_lat,
                    message=qdrant_status_msg,
                    dims=len(q_vec),
                    count=len(hits)
                )

                return {
                    "model_id": mid,
                    "display_name": m_cfg.display_name,
                    "dimensions": m_cfg.dimensions,
                    "languages": m_cfg.languages,
                    "badge_color": m_cfg.badge_color,
                    "is_multilingual": m_cfg.is_multilingual,
                    "is_mock": meta.get("is_mock", False),
                    "api_source": meta.get("source", "live_cloudflare_api" if "gemini" not in mid else "live_gemini_api"),
                    "api_error": err,
                    "api_status_code": sc,
                    "api_latency_ms": round(api_lat, 2),
                    "qdrant_latency_ms": round(qdrant_lat, 2),
                    "total_latency_ms": round(api_lat + qdrant_lat, 2),
                    "hits": hits,
                    "matches": hits,
                    "log_entry": {
                        "time": time.strftime("%H:%M:%S"),
                        "model": m_cfg.display_name,
                        "model_id": mid,
                        "endpoint": ep,
                        "status": sc,
                        "api_latency_ms": round(api_lat, 1),
                        "qdrant_latency_ms": round(qdrant_lat, 1),
                        "dims": len(q_vec),
                        "hits_count": len(hits),
                        "filtered_docs": selected_doc_ids,
                        "error": err
                    }
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(valid_model_ids), 7)) as executor:
                model_results = list(executor.map(_search_single_model, valid_model_ids))

            results_dict = {r["model_id"]: r for r in model_results if r is not None}

            self._send_json({
                "query": query,
                "selected_doc_ids": selected_doc_ids,
                "selected_model_ids": valid_model_ids,
                "is_single_model": len(valid_model_ids) == 1,
                "results_by_model": results_dict,
                "results": results_dict,
                "recent_logs": embedder.get_recent_logs(limit=40)
            })

        elif parsed.path == "/api/delete_document":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as e:
                self._send_json({"error": "Invalid JSON: " + str(e)}, 400)
                return

            doc_id = payload.get("doc_id")
            if not doc_id:
                self._send_json({"error": "doc_id is required"}, 400)
                return

            success = qdrant_manager.delete_document(doc_id)
            self._send_json({"status": "success", "deleted_doc_id": doc_id, "success": success})

        elif parsed.path == "/api/clear_all":
            qdrant_manager.clear_all_documents()
            self._send_json({"status": "success", "message": "All documents removed from Qdrant vector space"})

        elif parsed.path == "/api/save_credentials":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as e:
                self._send_json({"error": "Invalid JSON: " + str(e)}, 400)
                return

            account_id = payload.get("account_id", "").strip()
            api_token = payload.get("api_token", "").strip()

            if account_id and api_token:
                embedder.account_id = account_id
                embedder.api_token = api_token
                # Write to .env
                env_path = os.path.join(BASE_DIR, ".env")
                env_lines = []
                if os.path.exists(env_path):
                    with open(env_path, "r", encoding="utf-8") as f:
                        env_lines = [l for l in f.readlines() if not l.startswith("CLOUDFLARE_")]
                env_lines.append(f"CLOUDFLARE_ACCOUNT_ID={account_id}\n")
                env_lines.append(f"CLOUDFLARE_API_TOKEN={api_token}\n")
                with open(env_path, "w", encoding="utf-8") as f:
                    f.writelines(env_lines)

                # Also sync with chunking_studio/.env
                studio_env_path = os.path.join(BASE_DIR, "chunking_studio", ".env")
                if os.path.exists(os.path.dirname(studio_env_path)):
                    s_lines = []
                    if os.path.exists(studio_env_path):
                        with open(studio_env_path, "r", encoding="utf-8") as f:
                            s_lines = [l for l in f.readlines() if not l.startswith("CLOUDFLARE_")]
                    s_lines.append(f"CLOUDFLARE_ACCOUNT_ID={account_id}\n")
                    s_lines.append(f"CLOUDFLARE_API_TOKEN={api_token}\n")
                    with open(studio_env_path, "w", encoding="utf-8") as f:
                        f.writelines(s_lines)

                self._send_json({"status": "success", "is_configured": True})
            else:
                self._send_json({"error": "Both Account ID and API Token are required"}, 400)

        elif parsed.path == "/api/parse_file":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_json({"error": "Expected multipart/form-data"}, 400)
                return

            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)

            boundary = content_type.split("boundary=")[-1].strip()
            parts = body.split(f"--{boundary}".encode())

            extracted_text = ""
            filename = "uploaded_file.txt"

            for part in parts:
                if b'filename="' in part:
                    header_end = part.find(b"\r\n\r\n")
                    headers = part[:header_end].decode("utf-8", errors="replace")
                    file_content = part[header_end + 4 : -2]

                    if 'filename="' in headers:
                        filename = headers.split('filename="')[1].split('"')[0]

                    try:
                        extracted_text, _ = parse_uploaded_file(io.BytesIO(file_content), filename)
                    except Exception as e:
                        self._send_json({"error": str(e)}, 500)
                        return
                    break

            self._send_json({"text": extracted_text, "filename": filename})

        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), ChunkingHandler)
    print(f"Server running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            qdrant_manager.close()
        except Exception:
            pass


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
