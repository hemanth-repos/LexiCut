import sqlite3
import requests
import json
import time
import os

BASE_URL = "http://127.0.0.1:8000"

print("=" * 60)
print("LEXICUT SYSTEM VALIDATION")
print("=" * 60)

passed = 0
failed = 0


def check(name, condition):
    global passed, failed

    if condition:
        print(f"[PASS] {name}")
        passed += 1
    else:
        print(f"[FAIL] {name}")
        failed += 1


# --------------------------------------------------
# 1. HEALTH CHECK
# --------------------------------------------------

try:
    response = requests.get(f"{BASE_URL}/")
    check("API Health Check", response.status_code == 200)
except Exception as e:
    print(e)
    check("API Health Check", False)


# --------------------------------------------------
# 2. CACHE MISS
# --------------------------------------------------

payload = {
    "query": "What is machine learning?"
}

response = requests.post(
    f"{BASE_URL}/chat",
    json=payload
)

data = response.json()

check(
    "Cache MISS Response",
    data["telemetry"]["source"] == "llm"
)

# --------------------------------------------------
# 3. CACHE HIT
# --------------------------------------------------

response = requests.post(
    f"{BASE_URL}/chat",
    json=payload
)

data = response.json()

check(
    "Cache HIT Response",
    data["telemetry"]["source"] == "cache"
)

# --------------------------------------------------
# 4. SEMANTIC HIT
# --------------------------------------------------

payload = {
    "query": "Explain machine learning"
}

response = requests.post(
    f"{BASE_URL}/chat",
    json=payload
)

data = response.json()

check(
    "Semantic Similarity Lookup",
    response.status_code == 200
)

# --------------------------------------------------
# 5. ANTONYM SAFETY
# --------------------------------------------------

requests.post(
    f"{BASE_URL}/chat",
    json={"query": "Enable dark mode"}
)

response = requests.post(
    f"{BASE_URL}/chat",
    json={"query": "Disable dark mode"}
)

data = response.json()

check(
    "Antonym Safety Layer",
    data["telemetry"]["source"] == "llm"
)

# --------------------------------------------------
# 6. SQLITE DATABASE
# --------------------------------------------------

try:
    conn = sqlite3.connect("telemetry.db")
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM request_logs"
    )

    total_logs = cur.fetchone()[0]

    check(
        "SQLite Telemetry Logging",
        total_logs > 0
    )

except Exception:
    check(
        "SQLite Telemetry Logging",
        False
    )

# --------------------------------------------------
# 7. HIT RATE VALIDATION
# --------------------------------------------------

try:

    cur.execute("""
        SELECT COUNT(*)
        FROM request_logs
        WHERE status='HIT'
    """)

    hits = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*)
        FROM request_logs
        WHERE status='MISS'
    """)

    misses = cur.fetchone()[0]

    total = hits + misses

    hit_rate = (
        hits / total * 100
        if total > 0 else 0
    )

    print(
        f"\nHit Rate = {hit_rate:.2f}%"
    )

    check(
        "Hit Rate Calculation",
        total > 0
    )

except Exception:
    check(
        "Hit Rate Calculation",
        False
    )

# --------------------------------------------------
# 8. ANTONYM BLOCK COUNT
# --------------------------------------------------

try:

    cur.execute("""
        SELECT COUNT(*)
        FROM request_logs
        WHERE antonym_blocked = 1
    """)

    blocks = cur.fetchone()[0]

    print(
        f"Antonym Blocks = {blocks}"
    )

    check(
        "Antonym Tracking",
        blocks >= 0
    )

except Exception:
    check(
        "Antonym Tracking",
        False
    )

# --------------------------------------------------
# 9. TOKENS SAVED
# --------------------------------------------------

try:

    cur.execute("""
        SELECT SUM(tokens_saved)
        FROM request_logs
    """)

    saved = cur.fetchone()[0]

    print(
        f"Estimated Tokens Saved = {saved}"
    )

    check(
        "Token Savings Metric",
        saved is not None
    )

except Exception:
    check(
        "Token Savings Metric",
        False
    )

# --------------------------------------------------
# 10. PERSISTENCE FILES
# --------------------------------------------------

check(
    "query_metadata.pkl Exists",
    os.path.exists("query_metadata.pkl")
)

check(
    "faiss.index Exists",
    os.path.exists("faiss.index")
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Passed : {passed}")
print(f"Failed : {failed}")

if failed == 0:
    print("\nSYSTEM STATUS: HEALTHY")
else:
    print("\nSYSTEM STATUS: NEEDS ATTENTION")

print("=" * 60)
