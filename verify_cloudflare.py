import os
import sys
import requests
from dotenv import load_dotenv

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "chunking_studio", ".env"))

acct = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
auth_email = os.getenv("CLOUDFLARE_AUTH_EMAIL", "").strip()
global_key = os.getenv("CLOUDFLARE_GLOBAL_API_KEY", "").strip()

print("=" * 60)
print("🔍 Cloudflare Workers AI Credential Verification")
print("=" * 60)
print(f"Account ID     : {acct or 'MISSING'}")

if global_key and auth_email:
    auth_type = "Global API Key"
    headers = {
        "X-Auth-Email": auth_email,
        "X-Auth-Key": global_key,
        "Content-Type": "application/json"
    }
    print(f"Auth Method    : Global API Key ({auth_email})")
    print(f"Key Prefix     : {global_key[:4]}...{global_key[-4:]} (Len: {len(global_key)})")
elif token:
    auth_type = "Bearer API Token"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    print(f"Auth Method    : Bearer Token")
    print(f"Token Prefix   : {token[:6]}...{token[-4:] if len(token) > 10 else ''} (Len: {len(token)})")
else:
    print("\n❌ Error: Neither CLOUDFLARE_API_TOKEN nor (CLOUDFLARE_AUTH_EMAIL + CLOUDFLARE_GLOBAL_API_KEY) found in .env")
    sys.exit(1)

if not acct:
    print("\n❌ Error: CLOUDFLARE_ACCOUNT_ID is missing in .env")
    sys.exit(1)

# Test live inference on BGE-M3
print("\nTesting live inference on flagship model @cf/baai/bge-m3...")
model_url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/baai/bge-m3"

try:
    r_inf = requests.post(model_url, headers=headers, json={"text": ["Testing Cloudflare BGE-M3 embedding"]}, timeout=15)
    if r_inf.status_code == 200:
        res = r_inf.json()
        data = res.get("result", {}).get("data", [])
        dims = len(data[0]) if data and isinstance(data[0], list) else 0
        print(f"  🎉 SUCCESS! @cf/baai/bge-m3 returned live {dims}d embedding vector!")
        print(f"  Authenticated successfully via {auth_type}!")
    else:
        print(f"  ❌ Inference failed! HTTP {r_inf.status_code}:")
        print(f"     {r_inf.text}")
        if r_inf.status_code == 401:
            print("\n  👉 401 Authentication Error: Invalid token or key.")
except Exception as e:
    print(f"  ❌ Inference error: {e}")

print("=" * 60)
