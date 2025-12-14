import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://genexahealth.onrender.com/api/v1")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

def wake_server():
    # health endpoint is outside /api/v1 in your swagger screenshot
    url = "https://genexahealth.onrender.com/health"
    try:
        r = requests.get(url, timeout=30)
        print("✅ Wake:", r.status_code)
    except Exception as e:
        print("⚠️ Wake failed (ok to continue):", e)

def post_with_retries(url, data, tries=6, timeout=120):
    last_err = None
    for i in range(tries):
        try:
            r = requests.post(url, data=data, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            wait = 2 ** i
            print(f"⚠️ Attempt {i+1}/{tries} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise last_err

def get_token():
    url = f"{BASE_URL}/token"
    # FastAPI OAuth2PasswordRequestForm expects form data
    data = {"username": USERNAME, "password": PASSWORD}
    r = post_with_retries(url, data=data, tries=6, timeout=120)
    return r.json()

if __name__ == "__main__":
    wake_server()
    token = get_token()
    print("✅ Token response keys:", list(token.keys()))
    print("ACCESS_TOKEN:", token.get("access_token"))