import os
import sys
import json
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIO_DIR = os.path.join(BASE_DIR, "chunking_studio")
if STUDIO_DIR not in sys.path:
    sys.path.insert(0, STUDIO_DIR)

from dotenv import load_dotenv
load_dotenv()

from chunking_studio.src.chunkers import detect_document_strategy, get_chunker
from chunking_studio.src.chunkers.base import count_units, split_text_to_ceiling, ChunkResult
from chunking_studio.src.auditor import VoiceChunkAuditor
from chunking_studio.src.normalizer import GeminiDocumentNormalizer
from chunking_studio.src.document_parsers import parse_uploaded_file


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
            unit = payload.get("unit", "characters").lower()
            default_chunk = 500 if unit in ["characters", "chars"] else 150
            default_overlap = 100 if unit in ["characters", "chars"] else 25

            chunk_size = int(payload.get("chunk_size", default_chunk))
            overlap = int(payload.get("overlap", default_overlap))
            model_name = payload.get("model", "none")

            if not text:
                self._send_json({"error": "Document text cannot be empty"}, 400)
                return

            active_text = text
            # Gemini Stage 1
            if model_name and model_name.lower() != "none" and "gemini" in model_name.lower():
                api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
                normalizer = GeminiDocumentNormalizer(api_key=api_key, model_name=model_name)
                norm_text, norm_meta = normalizer.normalize(active_text)
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
            metadata = {"doc_title": "Document", "currency": "INR"}
            raw_chunks = chunker.chunk(active_text, metadata=metadata)

            # Enforce hard ceiling across all chunks
            final_chunks = []
            curr_id = 1
            for c in raw_chunks:
                c_text = c.text or c.searchable_augmented_text or c.raw_text
                if count_units(c_text, unit) > chunk_size:
                    entity_title = c.metadata.get("branch_name") or (c_text.splitlines()[0].strip() if c_text.splitlines() else "Entity")
                    sub_pieces = split_text_to_ceiling(c_text, chunk_size, unit)
                    total_sub = len(sub_pieces)
                    for sub_idx, sub in enumerate(sub_pieces, 1):
                        anchored_sub = sub
                        # Prepend contextual anchor header to subsequent parts if missing
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
                "stats": audit_summary
            })

        elif parsed.path == "/api/parse_file":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_json({"error": "Expected multipart/form-data"}, 400)
                return

            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            
            # Simple multipart boundary parsing
            boundary = content_type.split("boundary=")[-1].strip()
            parts = body.split(f"--{boundary}".encode())

            extracted_text = ""
            filename = "uploaded_file.txt"

            for part in parts:
                if b'filename="' in part:
                    header_end = part.find(b"\r\n\r\n")
                    headers = part[:header_end].decode("utf-8", errors="replace")
                    file_content = part[header_end + 4 : -2]  # strip trailing \r\n
                    
                    # Extract filename
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
    server.serve_forever()


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
