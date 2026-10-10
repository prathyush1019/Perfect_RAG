"""
Root launcher for Voice AI Context-AG Chunking Studio.
Delegates to chunking_studio/app.py
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIO_DIR = os.path.join(CURRENT_DIR, "chunking_studio")
if STUDIO_DIR not in sys.path:
    sys.path.insert(0, STUDIO_DIR)

app_path = os.path.join(STUDIO_DIR, "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()
exec(code, {"__name__": "__main__", "__file__": app_path})
