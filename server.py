import os
import sys
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

from chunking_studio.src.chunkers import detect_document_strategy, get_chunker
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
                if norm_meta.get("status") == "success":
                    active_text = norm_text

            strategy, _ = detect_document_strategy(active_text)
            chunker = get_chunker(
                strategy,
                unit=unit,
                chunk_size=chunk_size,
                overlap=overlap,
                target_words=chunk_size,
                overlap_words=overlap
            )
            metadata = {"doc_title": payload.get("doc_name", "Document"), "currency": "INR"}
            raw_chunks = chunker.chunk(active_text, metadata=metadata)

            # Enforce hard ceiling across all chunks
            final_chunks = []
            curr_id = 1
            for c in raw_chunks:
                c_text = (c.text or c.searchable_augmented_text or c.raw_text or "").strip()
                # Discard degenerate / solitary symbol chunks (e.g., solitary '#')
                if len(re.findall(r"\w", c_text, re.UNICODE)) < 2:
                    continue

                if count_units(c_text, unit) > chunk_size:
                    entity_title = c.metadata.get("branch_name") or (c_text.splitlines()[0].strip() if c_text.splitlines() else "Entity")
                    sub_pieces = split_text_to_ceiling(c_text, chunk_size, unit)
                    total_sub = len(sub_pieces)
                    for sub_idx, sub in enumerate(sub_pieces, 1):
                        if len(re.findall(r"\w", sub.strip(), re.UNICODE)) < 2:
                            continue
                        anchored_sub = sub
                        if sub_idx > 1 and entity_title and entity_title not in sub[:120]:
                            anchored_sub = f"[{entity_title} (Part {sub_idx}/{total_sub})]\n" + sub

                        final_chunks.append(
                            ChunkResult(
                                chunk_id=curr_id,
                                start_char=c.start_char,
                                end_char=c.end_char,
                                text=anchored_sub,
                                raw_text=anchored_sub,
                                searchable_augmented_text=anchored_sub,
                                chunk_type=c.chunk_type,
                                metadata={
                                    **c.metadata,
                                    "parent_entity": entity_title,
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
                "models_count": len(embeddings_by_model)
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
                    top_k=top_k
                )
                meta = embedder.last_fetch_meta.get(mid, {})
                return {
                    "model_id": mid,
                    "display_name": m_cfg.display_name,
                    "dimensions": m_cfg.dimensions,
                    "languages": m_cfg.languages,
                    "badge_color": m_cfg.badge_color,
                    "is_multilingual": m_cfg.is_multilingual,
                    "is_mock": meta.get("is_mock", not embedder.is_configured()),
                    "api_source": meta.get("source", "live_cloudflare_api" if embedder.is_configured() else "offline_simulation"),
                    "api_latency_ms": round(api_lat, 2),
                    "qdrant_latency_ms": round(qdrant_lat, 2),
                    "total_latency_ms": round(api_lat + qdrant_lat, 2),
                    "hits": hits,
                    "matches": hits
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
                "results": results_dict
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
